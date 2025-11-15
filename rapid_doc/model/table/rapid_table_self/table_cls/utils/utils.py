import traceback
from io import BytesIO
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_pybind11_state import (
    SessionOptions,
    GraphOptimizationLevel,
)

from ...utils import is_url

InputType = Union[str, np.ndarray, bytes, Path, Image.Image]


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





class OpenVINOInferSession():
    def __init__(self, model_path: Union[str, Path], num_threads: int = -1):
        # super().__init__(cfg)
        # self.logger = get_logger("OpenVINOInferSession")

        core = Core()

        # model_path = cfg.get("model_path", None)
        # self._verify_model(model_path)

        self.model_path = model_path
        # self.logger.info(f"Using {model_path}")

        self.model = core.read_model(model=str(model_path))
        self.input_tensor = self.model.inputs[0]
        self.output_tensors = self.model.outputs

        device = 'CPU'
        ov_config = {}
        self.compiled_model = core.compile_model(
            self.model,
            device,
            ov_config,
        )
        self.infer_request = self.compiled_model.create_infer_request()

    def _init_config(self, cfg) -> Dict[Any, Any]:
        config = {}
        engine_cfg = cfg.get('engine_cfg', {})
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

        # self.logger.info(f"Using OpenVINO config: {config}")
        return config

    def __call__(self, input_content: np.ndarray) -> Any:
        try:
            input_content = np.squeeze(input_content, axis=1)
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

class OrtInferSession:
    def __init__(self, model_path: Union[str, Path], num_threads: int = -1):
        self.verify_exist(model_path)

        self.num_threads = num_threads
        self._init_sess_opt()

        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }
        EP_list = [(cpu_ep, cpu_provider_options)]
        try:
            self.session = InferenceSession(
                str(model_path), sess_options=self.sess_opt, providers=EP_list
            )
        except TypeError:
            # 这里兼容ort 1.5.2
            self.session = InferenceSession(str(model_path), sess_options=self.sess_opt)

    def _init_sess_opt(self):
        self.sess_opt = SessionOptions()
        self.sess_opt.log_severity_level = 4
        self.sess_opt.enable_cpu_mem_arena = False

        if self.num_threads != -1:
            self.sess_opt.intra_op_num_threads = self.num_threads

        self.sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(None, input_dict)
        except Exception as e:
            error_info = traceback.format_exc()
            raise ONNXRuntimeError(error_info) from e

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_name(self, output_idx=0):
        return self.session.get_outputs()[output_idx].name

    def get_metadata(self):
        meta_dict = self.session.get_modelmeta().custom_metadata_map
        return meta_dict

    @staticmethod
    def verify_exist(model_path: Union[Path, str]):
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist!")

        if not model_path.is_file():
            raise FileExistsError(f"{model_path} must be a file")


class ONNXRuntimeError(Exception):
    pass


class LoadImageError(Exception):
    pass


class LoadImage:
    def __init__(self):
        pass

    def __call__(self, img: InputType) -> np.ndarray:
        if not isinstance(img, InputType.__args__):
            raise LoadImageError(
                f"The img type {type(img)} does not in {InputType.__args__}"
            )

        origin_img_type = type(img)
        img = self.load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            if is_url(img):
                img = Image.open(requests.get(img, stream=True, timeout=60).raw)
            else:
                self.verify_exist(img)
                img = Image.open(img)

            try:
                img = self.img_to_ndarray(img)
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → BGR"""
        r, g, b, a = cv2.split(img)
        new_img = cv2.merge((b, g, r))

        not_a = cv2.bitwise_not(a)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(new_img, new_img, mask=a)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")