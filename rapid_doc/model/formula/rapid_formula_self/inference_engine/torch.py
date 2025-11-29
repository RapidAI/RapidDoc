# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from ..model_handler import ModelProcessor

from ..networks.architectures.base_model import BaseModel
from ..utils.logger import Logger
from ..utils.utils import mkdir
from .base import InferSession

root_dir = Path(__file__).resolve().parent.parent
# DEFAULT_CFG_PATH = root_dir / "networks" / "arch_config.yaml"
DEFAULT_CFG_PATH = root_dir / "networks" / "pp_formulanet_arch_config.yaml"


class TorchInferSession(InferSession):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.logger = Logger(logger_name=__name__).get_log()
        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type, cfg.engine_type)
        else:
            model_path = Path(cfg.model_dir_or_path)
        arch_config = self._load_arch_config(model_path)

        self.predictor = self._build_and_load_model(arch_config, model_path)

        self._setup_device(cfg)

        self.predictor.eval()

    def _load_arch_config(self, model_path: Path):
        all_cfg = OmegaConf.load(DEFAULT_CFG_PATH)
        name = model_path.stem.lower().replace("-", "_")

        for k in all_cfg.keys():
            if k.lower().replace("-", "_") == name:
                return all_cfg[k]

        raise ValueError(f"architecture {model_path.stem} is not in arch_config.yaml")

    def _build_and_load_model(self, arch_config, model_path: Path):
        os.environ['RAPID_FORMULA_DEVICE_MODE'] = self.get_device(self.cfg)
        model = BaseModel(arch_config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        return model

    def _setup_device(self, cfg):
        self.device, self.use_gpu, self.use_npu = self._resolve_device_config(cfg)

        if self.use_npu:
            self._config_npu()

        self._move_model_to_device()

    def _resolve_device_config(self, cfg):
        if cfg.engine_cfg.get("use_cuda", False):
            return torch.device(f"cuda:{cfg.engine_cfg['gpu_id']}"), True, False

        if cfg.engine_cfg.get("use_npu", False):
            return torch.device(f"npu:{cfg.engine_cfg['npu_id']}"), False, True

        return torch.device("cpu"), False, False

    def get_device(self, cfg):
        if cfg.engine_cfg.get("use_cuda", False):
            return f"cuda:{cfg.engine_cfg['gpu_id']}"

        if cfg.engine_cfg.get("use_npu", False):
            return f"npu:{cfg.engine_cfg['npu_id']}"

        return "cpu"

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
            self.logger.warning(
                "torch_npu is not installed, options with ACL setting failed. \n"
                "Please refer to https://github.com/Ascend/pytorch to see how to install."
            )

            self.device = torch.device("cpu")
            self.use_npu = False

    def _move_model_to_device(self):
        self.predictor.to(self.device)

    def __call__(self, img: np.ndarray):
        with torch.no_grad():
            inp = torch.from_numpy(img)
            if self.use_gpu or self.use_npu:
                inp = inp.to(self.device)

            # 适配跟onnx对齐取值逻辑
            outputs = self.predictor(inp).cpu().numpy()
            return [outputs]

    def have_key(self, key: str = "character") -> bool:
        return False

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> Dict[str, Any]:
        dict_path = self.cfg.dict_keys_path
        if not dict_path or (not Path(dict_path).exists()):
            dict_path = ModelProcessor.get_character_path(self.cfg.model_type, self.cfg.engine_type)
        with open(dict_path, "r", encoding="utf-8") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
            return data["PostProcess"]["character_dict"]


class TorchInferError(Exception):
    pass
