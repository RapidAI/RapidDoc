from collections import Counter
import re
import time
from typing import List

import cv2
import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidFormulaOutput
from ..base import BaseModelHandler
from .post_process import PPPostProcess
from .pre_process import PPPreProcess


def _has_repetitive_output(latex: str) -> bool:
    # 长公式本身可以有效，只重试被少量重复命令或字符支配的输出。
    tokens = re.findall(r"\\[A-Za-z]+|[A-Za-z0-9]+", latex)
    return len(tokens) >= 128 and max(Counter(tokens).values()) * 3 >= len(tokens)


def _clean_speckled_crop(image: np.ndarray) -> np.ndarray | None:
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 32:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((gray < 135).astype(np.uint8))
    areas = stats[1:, cv2.CC_STAT_AREA]
    if np.count_nonzero(areas < 5) < 16:
        return None
    keep = stats[:, cv2.CC_STAT_AREA] >= 5
    keep[0] = False
    if not keep.any():
        return None
    # 保留深色笔画附近的浅色边缘。只有浅灰散点遍布各区才尝试去背景。
    foreground = cv2.dilate(keep[labels].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    background_ink = (gray < 220) & ~foreground
    if any(tile.mean() < 0.025 for row in np.array_split(background_ink, 3)
           for tile in np.array_split(row, 3, axis=1)):
        return None
    cleaned = image.copy()
    cleaned[~foreground] = 255
    return cleaned


class PPFormulaNetPlusModelHandler(BaseModelHandler):
    def __init__(self, character_dict, session: InferSession, target_size):
        self.img_size = target_size
        self.pp_preprocess = PPPreProcess(img_size=self.img_size)
        self.pp_postprocess = PPPostProcess(character_dict)

        self.session = session

    def __call__(self, ori_img_list: List[np.ndarray]) -> List[RapidFormulaOutput]:
        s1 = time.perf_counter()
        rec_formula_list = self._recognize(ori_img_list)
        result_list = []
        for image, rec_formula in zip(ori_img_list, rec_formula_list):
            if _has_repetitive_output(rec_formula):
                cleaned = _clean_speckled_crop(image)
                if cleaned is not None:
                    retry = self._recognize([cleaned])[0]
                    if retry.strip() and not _has_repetitive_output(retry):
                        rec_formula = retry
            elapse = time.perf_counter() - s1
            result = RapidFormulaOutput(img=image, rec_formula=rec_formula, elapse=elapse)
            result_list.append(result)
        return result_list

    def _recognize(self, images: List[np.ndarray]) -> List[str]:
        inputs = np.concatenate(self.preprocess(images), axis=0)
        predictions = self.session(inputs)[0]
        return self.pp_postprocess([prediction.reshape(-1) for prediction in predictions])

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self.pp_preprocess(image)

    def postprocess(self, preds):
        return self.pp_postprocess(preds)
