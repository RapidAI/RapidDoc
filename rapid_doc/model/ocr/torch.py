from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from importlib.metadata import version
rapidocr_version = version("rapidocr")

if rapidocr_version >= "3.4.3":
    from rapidocr.inference_engine.pytorch.networks.backbones.rec_hgnet import ConvBNAct
else:
    from rapidocr.networks.backbones.rec_hgnet import ConvBNAct
from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.log import logger
from rapidocr.inference_engine.base import FileInfo, InferSession
import rapidocr as rapidocr_pkg

from rapid_doc.model.ocr.ppocrv6_pytorch.modeling.architectures.base_model import BaseModel

root_dir = Path(rapidocr_pkg.__path__[0])
RAPID_DOC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = RAPID_DOC_ROOT / "resources" / "arch_config.yaml"
ARCH_NAME_ALIASES = {
    "ch_PP-OCRv6_rec_small": "ch_PP-OCRv6_small_rec_infer",
    "ch_PP-OCRv6_det_small": "ch_PP-OCRv6_det_small",
}

def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        model_path = self._init_model_path(cfg)
        arch_config = self._load_arch_config(model_path)

        self.predictor = self._build_and_load_model(arch_config, model_path)

        self._setup_device(cfg)

        self.predictor.eval()

    def _init_model_path(self, cfg) -> Path:
        model_path = cfg.get("model_path", None)
        if model_path is None:
            model_info = self.get_model_url(
                FileInfo(
                    engine_type=cfg.engine_type,
                    ocr_version=cfg.ocr_version,
                    task_type=cfg.task_type,
                    lang_type=cfg.lang_type,
                    model_type=cfg.model_type,
                )
            )
            default_model_url = model_info["model_dir"]
            model_path = self.DEFAULT_MODEL_PATH / Path(default_model_url).name
            DownloadFile.run(
                DownloadFileInput(
                    file_url=default_model_url,
                    sha256=model_info["SHA256"],
                    save_path=model_path,
                    logger=logger,
                )
            )

        logger.info(f"Using {model_path}")
        self._verify_model(model_path)
        return Path(model_path)

    def _load_arch_config(self, model_path: Path):
        all_arch_config = OmegaConf.load(DEFAULT_CFG_PATH)

        file_name = ARCH_NAME_ALIASES.get(model_path.stem, model_path.stem)
        if file_name not in all_arch_config:
            raise ValueError(f"architecture {file_name} is not in arch_config.yaml")

        return all_arch_config.get(file_name)

    def _build_and_load_model(self, arch_config, model_path: Path):
        state_dict = self._load_state_dict(model_path)
        kwargs = {}
        out_channels = self._get_rec_out_channels(state_dict)
        if out_channels is not None:
            kwargs["out_channels"] = out_channels

        model = BaseModel(arch_config, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def _load_state_dict(self, model_path: Path):
        if model_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise TorchInferError(
                    "safetensors is required for PP-OCRv6 PyTorch OCR weights. "
                    "Please install rapid-doc[gpu] or `pip install safetensors`."
                ) from exc
            state_dict = load_file(str(model_path), device="cpu")
        else:
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(model_path, map_location="cpu")

        if any(key.startswith("model.") for key in state_dict):
            state_dict = {
                key.removeprefix("model."): value
                for key, value in state_dict.items()
            }
        return state_dict

    @staticmethod
    def _get_rec_out_channels(state_dict):
        if "head.head.weight" in state_dict:
            return int(state_dict["head.head.weight"].shape[0])
        return None

    def _setup_device(self, cfg):
        self.device, self.use_gpu, self.use_npu = self._resolve_device_config(cfg)

        if self.use_npu:
            self._config_npu()

        self._move_model_to_device()

    def _resolve_device_config(self, cfg):
        if cfg.engine_cfg.use_cuda:
            return torch.device(f"cuda:{cfg.engine_cfg.gpu_id}"), True, False

        if cfg.engine_cfg.get('use_npu'):
            return torch.device(f"npu:{cfg.engine_cfg.npu_id}"), False, True

        return torch.device("cpu"), False, False

    def _config_npu(self):
        try:
            import torch_npu

            kernel_meta_dir = (root_dir / "kernel_meta").resolve()
            mkdir(kernel_meta_dir)

            options = {
                "ACL_OP_COMPILER_CACHE_MODE": "enable",
                "ACL_OP_COMPILER_CACHE_DIR": str(kernel_meta_dir),
            }
            torch_npu.npu.set_option(options)
        except ImportError:
            logger.warning(
                "torch_npu is not installed, options with ACL setting failed. \n"
                "Please refer to https://github.com/Ascend/pytorch to see how to install."
            )

            self.device = torch.device("cpu")
            self.use_npu = False

    def _move_model_to_device(self):
        self.predictor.to(self.device)

        for module in self.predictor.modules():
            # det
            if hasattr(module, 'rep'):
                module.rep()
            # rec
            if isinstance(module, ConvBNAct):
                if module.use_act:
                    torch.quantization.fuse_modules(module, ['conv', 'bn', 'act'], inplace=True)
                else:
                    torch.quantization.fuse_modules(module, ['conv', 'bn'], inplace=True)


    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu or self.use_npu:
                inp = inp.to(self.device)

            # 适配跟onnx对齐取值逻辑
            outputs = self.predictor(inp)
            return self._to_numpy_output(outputs)

    @staticmethod
    def _to_numpy_output(outputs):
        if isinstance(outputs, dict):
            if "maps" in outputs:
                outputs = outputs["maps"]
            elif "ctc_logits" in outputs:
                outputs = torch.softmax(outputs["ctc_logits"], dim=2)
            elif "res" in outputs:
                outputs = outputs["res"]
            else:
                outputs = next(iter(outputs.values()))
        return outputs.cpu().numpy()

    def have_key(self, key: str = "character") -> bool:
        return False

    def get_character_list(self, key: str = "character"):
        return []


class TorchInferError(Exception):
    pass
