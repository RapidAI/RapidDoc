import time
from typing import Optional, Union, List

import cv2
import numpy as np
from dataclasses import asdict

from tqdm import tqdm

from ..utils import RapidTableInput, ModelType, EngineType
from ..inference_engine.base import get_engine
from ..model_processor.main import ModelProcessor
from ..utils import LoadImage, InputType


class TableCls:
    def __init__(self, cfg: Optional[RapidTableInput] = None):
        if cfg is None:
            cfg = RapidTableInput()
        cfg.model_type = ModelType.PADDLE_CLS

        if not cfg.model_dir_or_path and cfg.model_type is not None:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.table_engine = PaddleCls(asdict(cfg))
        self.load_img = LoadImage()

    def __call__(self, img_contents: Union[List[InputType], InputType], batch_size: int = 1, tqdm_enable=False):
        ss = time.perf_counter()
        label_res = []
        if not isinstance(img_contents, list):
            img_contents = [img_contents]
        total_nums = len(img_contents)
        for start_i in tqdm(range(0, total_nums, batch_size), desc="Table-cls predict", disable=not tqdm_enable):
            end_i = min(total_nums, start_i + batch_size)
            imgs = self._load_imgs(img_contents[start_i:end_i])
            x = self.table_engine.batch_preprocess(imgs)
            predict_cla = self.table_engine(x)
            label_res.extend(predict_cla)
        table_elapse = time.perf_counter() - ss
        return label_res, table_elapse

    def _load_imgs(
        self, img_content: Union[List[InputType], InputType]
    ) -> List[np.ndarray]:
        img_contents = img_content if isinstance(img_content, list) else [img_content]
        return [self.load_img(img) for img in img_contents]

class PaddleCls:
    def __init__(self, cfg):
        if cfg["engine_type"] is None:
            cfg["engine_type"] = EngineType.ONNXRUNTIME
        self.session = get_engine(cfg["engine_type"])(cfg)
        # self.table_cls = OpenVINOInferSession(model_path)
        self.inp_h = 224
        self.inp_w = 224
        self.resize_short = 256
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls = {0: "wired", 1: "wireless"}

    def batch_preprocess(self, imgs):
        res_imgs = []
        for img in imgs:
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
            res_imgs.append(img)
        x = np.stack(res_imgs, axis=0).astype(dtype=np.float32, copy=False)
        return x

    def __call__(self, img):
        pred_output = self.session(img)[0]
        pred_idxs = list(np.argmax(pred_output, axis=1))
        return [self.cls[idx] for idx in pred_idxs]
