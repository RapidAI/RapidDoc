import time
from enum import Enum
from pathlib import Path
from typing import Union, Dict

import cv2
import numpy as np

from .utils.download_model import DownloadModel
from .utils.utils import InputType, LoadImage, OrtInferSession


class ModelType(Enum):
    PADDLE_CLS = "paddle"


ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/"
KEY_TO_MODEL_URL = {
    ModelType.PADDLE_CLS.value: f"{ROOT_URL}/table_cls/paddle_cls.onnx",
}


class TableCls:
    def __init__(self, model_type=ModelType.PADDLE_CLS.value, model_path=None):
        model_path = self.get_model_path(model_type, model_path)
        if model_type == ModelType.PADDLE_CLS.value:
            self.table_engine = PaddleCls(model_path)
        self.load_img = LoadImage()

    def __call__(self, content: InputType):
        ss = time.perf_counter()
        img = self.load_img(content)
        img = self.table_engine.preprocess(img)
        predict_cla = self.table_engine([img])
        table_elapse = time.perf_counter() - ss
        return predict_cla, table_elapse

    @staticmethod
    def get_model_path(
        model_type: str, model_path: Union[str, Path, None]
    ) -> Union[str, Dict[str, str]]:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(model_type, None)
        if isinstance(model_url, str):
            model_path = DownloadModel.download(model_url)
            return model_path

        if isinstance(model_url, dict):
            model_paths = {}
            for k, url in model_url.items():
                model_paths[k] = DownloadModel.download(
                    url, save_model_name=f"{model_type}_{Path(url).name}"
                )
            return model_paths

        raise ValueError(f"Model URL: {type(model_url)} is not between str and dict.")


class PaddleCls:
    def __init__(self, model_path):
        self.table_cls = OrtInferSession(model_path)
        self.inp_h = 224
        self.inp_w = 224
        self.resize_short = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls = {0: "wired", 1: "wireless"}

    def preprocess(self, img):
        # short resize
        img_h, img_w = img.shape[:2]
        percent = float(self.resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LANCZOS4)
        # center crop
        img_h, img_w = img.shape[:2]
        w_start = (img_w - self.inp_w) // 2
        h_start = (img_h - self.inp_h) // 2
        w_end = w_start + self.inp_w
        h_end = h_start + self.inp_h
        img = img[h_start:h_end, w_start:w_end, :]
        # normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img -= self.mean
        img /= self.std
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        # Add batch dimension, only one image
        img = np.expand_dims(img, axis=0)
        return img

    def __call__(self, img):
        pred_output = self.table_cls(img)[0]
        pred_idxs = list(np.argmax(pred_output, axis=1))
        predict_cla = max(set(pred_idxs), key=pred_idxs.count)
        return self.cls[predict_cla]
