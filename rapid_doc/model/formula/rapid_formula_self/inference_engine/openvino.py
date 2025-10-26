import os
import onnx
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from openvino.runtime import Core

from ..model_handler.utils import ModelProcessor
from ..utils.logger import Logger
from ..utils.typings import RapidFormulaInput
from .base import InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: RapidFormulaInput):
        super().__init__(cfg)
        self.logger = Logger(logger_name=__name__).get_log()

        core = Core()

        if cfg.model_dir_or_path is None:
            model_path = ModelProcessor.get_model_path(cfg.model_type, cfg.engine_type)
        else:
            model_path = Path(cfg.model_dir_or_path)

        self._verify_model(model_path)
        self.model_path = model_path
        self.logger.info(f"Using {model_path}")

        self.onnx_model = onnx.load(self.model_path)  # 读取 ONNX 文件

        config = self._init_config(cfg)
        core.set_property("CPU", config)

        model_onnx = core.read_model(model_path)

        compile_model = core.compile_model(model=model_onnx, device_name="CPU")
        # self.session = compile_model.create_infer_request()
        self.session = compile_model

    def _init_config(self, cfg: RapidFormulaInput) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.engine_cfg

        infer_num_threads = engine_cfg.get("inference_num_threads", -1)
        if infer_num_threads != -1 and 1 <= infer_num_threads <= os.cpu_count():
            config["INFERENCE_NUM_THREADS"] = str(infer_num_threads)

        performance_hint = engine_cfg.get("performance_hint", None)
        if performance_hint is not None:
            config["PERFORMANCE_HINT"] = str(performance_hint)

        performance_num_requests = engine_cfg.get("performance_num_requests", -1)
        if performance_num_requests != -1:
            config["PERFORMANCE_HINT_NUM_REQUESTS"] = str(performance_num_requests)

        enable_cpu_pinning = engine_cfg.get("enable_cpu_pinning", None)
        if enable_cpu_pinning is not None:
            config["ENABLE_CPU_PINNING"] = str(enable_cpu_pinning)

        num_streams = engine_cfg.get("num_streams", -1)
        if num_streams != -1:
            config["NUM_STREAMS"] = str(num_streams)

        enable_hyper_threading = engine_cfg.get("enable_hyper_threading", None)
        if enable_hyper_threading is not None:
            config["ENABLE_HYPER_THREADING"] = str(enable_hyper_threading)

        scheduling_core_type = engine_cfg.get("scheduling_core_type", None)
        if scheduling_core_type is not None:
            config["SCHEDULING_CORE_TYPE"] = str(scheduling_core_type)

        self.logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray) -> Any:
        input_dict = dict(zip(self.get_input_names(), [input_content]))
        try:
            result = self.session(input_dict)
            return [result[out] for out in self.session.outputs]
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [inp.any_name for inp in self.session.inputs]

    def get_output_names(self) -> List[str]:
        return [inp.any_name for inp in self.session.outputs]

    @property
    def characters(self) -> List[str]:
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:

        meta_dict = {p.key: p.value for p in self.onnx_model.metadata_props}
        if key not in meta_dict:
            return []
        return meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        meta_dict = {p.key: p.value for p in self.onnx_model.metadata_props}
        return key in meta_dict


class OpenVIONError(Exception):
    pass
