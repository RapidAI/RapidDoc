# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import os
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig

try:
    import openvino as ov
    from openvino.runtime import Core
except ImportError:
    raise ImportError(
        "openvino is not installed. Please install it with: pip install openvino"
    )

from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.log import logger
from rapidocr.inference_engine.base import FileInfo, InferSession


class OpenVINOInferSession(InferSession):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.logger = logger

        core = Core()

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
            model_path = self.DEFAULT_MODEL_PATH / Path(model_info["model_dir"]).name
            download_params = DownloadFileInput(
                file_url=model_info["model_dir"],
                sha256=model_info["SHA256"],
                save_path=model_path,
                logger=self.logger,
            )
            DownloadFile.run(download_params)

        self.logger.info(f"Using {model_path}")
        model_path = Path(model_path)
        self._verify_model(model_path)

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

    def _init_config(self, cfg: DictConfig) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.get("engine_cfg", {})

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

            # self.infer_request.infer()
            # 使用异步推理替代同步 infer()
            self.infer_request.start_async()
            self.infer_request.wait()  # 等待推理完成

            output_tensor_name = self.output_tensors[0].get_any_name()
            output = self.infer_request.get_tensor(output_tensor_name).data
            return output
        except Exception as e:
            error_info = traceback.format_exc()
            raise OpenVIONError(error_info) from e

    def have_key(self, key: str = "character") -> bool:
        return False


class OpenVIONError(Exception):
    pass
