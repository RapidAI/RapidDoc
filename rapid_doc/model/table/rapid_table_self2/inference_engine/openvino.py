import os
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from openvino.runtime import Core

try:
    import openvino as ov
except ImportError:
    raise ImportError(
        "openvino is not installed. Please install it with: pip install openvino"
    )


from ..model_processor.main import ModelProcessor
from ..utils.logger import Logger
from .base import InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger = Logger(logger_name=__name__).get_log()

        core = Core()

        if cfg["model_dir_or_path"] is None:
            model_path = ModelProcessor.get_model_path(cfg['model_type'])
        else:
            model_path = Path(cfg["model_dir_or_path"])

        self._verify_model(model_path)
        self.model_path = model_path
        self.logger.info(f"Using {model_path}")

        self.model = core.read_model(model=str(model_path))
        self.input_tensor = self.model.inputs[0]
        self.output_tensors = self.model.outputs

        device = cfg.get('device', 'CPU')
        ov_config = self._init_config(cfg)
        self.compiled_model = core.compile_model(
            self.model,
            device,
            ov_config,
        )
        self.infer_request = self.compiled_model.create_infer_request()

    def _init_config(self, cfg) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg['engine_cfg']

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
        try:
            input_tensor_name = self.input_tensor.get_any_name()
            self.infer_request.set_tensor(input_tensor_name, ov.Tensor(input_content))
            self.infer_request.infer()

            outputs = []
            for output_tensor in self.output_tensors:
                output_tensor_name = output_tensor.get_any_name()
                output = self.infer_request.get_tensor(output_tensor_name).data
                outputs.append(output)

            return outputs

        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def get_input_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.inputs]

    def get_output_names(self) -> List[str]:
        return [tensor.get_any_name() for tensor in self.model.outputs]

    @property
    def characters(self):
        return self.get_character_list()

    def get_character_list(self, key: str = "character") -> List[str]:
        val = self.model.get_rt_info()["framework"][key]
        return val.value.splitlines()

    def have_key(self, key: str = "character") -> bool:
        try:
            rt_info = self.model.get_rt_info()
            return key in rt_info
        except:
            return False


class OpenVIONError(Exception):
    pass
