import os
from copy import deepcopy
import traceback
from pathlib import Path
from typing import Any, List

import numpy as np
from omegaconf import DictConfig
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from ...model_handler.utils import ModelProcessor
from ...utils.logger import Logger
from ...utils.typings import RapidLayoutInput, ModelType
from ..base import InferSession
from .provider_config import ProviderConfig


class OrtInferSession(InferSession):
    def __init__(self, cfg: RapidLayoutInput):
        self.logger = Logger(logger_name=__name__).get_log()

        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type)
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        self.logger.info(f"Using {model_path}")

        engine_cfg = self.update_params(
            deepcopy(self.engine_cfg[cfg.engine_type.value]), cfg.engine_cfg
        )

        sess_opt = self._init_sess_opts(engine_cfg)

        provider_cfg = ProviderConfig(engine_cfg=engine_cfg)
        providers = provider_cfg.get_ep_list()
        cuda_requested = any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider"
                             for p in providers)
        source = model_path
        self.mask_uint8 = False
        if cuda_requested and cfg.model_type == ModelType.PP_DOCLAYOUTV3 and engine_cfg.get("mask_uint8", True):
            try:
                import onnx
                from .optimization import compact_binary_mask

                model = compact_binary_mask(onnx.load(model_path))
                if model is not None:
                    onnx.external_data_helper.convert_model_from_external_data(model)
                    source = model.SerializeToString()
                    self.mask_uint8 = True
                else:
                    self.logger.info("Layout binary mask compression skipped: unsupported graph")
            except Exception as error:
                self.logger.warning("Layout mask compression disabled: %s", error)
        try:
            self.session = InferenceSession(source, sess_options=sess_opt, providers=providers)
        except Exception:
            if not self.mask_uint8:
                raise
            self.logger.warning("Layout compact session failed; loading original model", exc_info=True)
            self.mask_uint8 = False
            self.session = InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        if self.mask_uint8 and "CUDAExecutionProvider" not in self.session.get_providers():
            self.mask_uint8 = False
            self.session = InferenceSession(model_path, sess_options=sess_opt, providers=providers)
        provider_cfg.verify_providers(self.session.get_providers())
        self.io_binding = (cuda_requested and "CUDAExecutionProvider" in self.session.get_providers()
                           and cfg.model_type in (ModelType.PP_DOCLAYOUTV2, ModelType.PP_DOCLAYOUTV3)
                           and engine_cfg.get("io_binding", True))
        self.cuda_device_id = int(self.session.get_provider_options().get("CUDAExecutionProvider", {}).get("device_id", 0))
        self.logger.info("Layout ONNX optimization: mask_uint8=%s, io_binding=%s", self.mask_uint8, self.io_binding)

    @staticmethod
    def _init_sess_opts(cfg: DictConfig) -> SessionOptions:
        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = cfg.enable_cpu_mem_arena
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        cpu_nums = os.cpu_count()
        intra_op_num_threads = cfg.get("intra_op_num_threads", -1)
        if intra_op_num_threads != -1 and 1 <= intra_op_num_threads <= cpu_nums:
            sess_opt.intra_op_num_threads = intra_op_num_threads

        inter_op_num_threads = cfg.get("inter_op_num_threads", -1)
        if inter_op_num_threads != -1 and 1 <= inter_op_num_threads <= cpu_nums:
            sess_opt.inter_op_num_threads = inter_op_num_threads

        return sess_opt

    def __call__(self, input_content: np.ndarray, scale_factor: np.ndarray = None) -> Any:
        if scale_factor is not None:
            input_names = self.get_input_names()
            input_dict = {}
            if "image" in input_names:
                input_dict["image"] = input_content
            if "scale_factor" in input_names:
                input_dict["scale_factor"] = scale_factor
            if "im_shape" in input_names:
                h, w = input_content.shape[-2:]
                input_dict["im_shape"] = np.tile(np.array([[h, w]], dtype=np.float32), (input_content.shape[0], 1))
        else:
            input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            if self.io_binding:
                # Per-call bindings support changing batch sizes and concurrent callers.
                binding = self.session.io_binding()
                for name, value in input_dict.items():
                    binding.bind_cpu_input(name, np.ascontiguousarray(value))
                for name in self.get_output_names():
                    binding.bind_output(name, "cuda", self.cuda_device_id)
                self.session.run_with_iobinding(binding)
                binding.synchronize_outputs()
                return binding.copy_outputs_to_cpu()
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        return [v.name for v in self.session.get_outputs()]

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in meta_dict.keys():
            return True
        return False


class ONNXRuntimeError(Exception):
    pass
