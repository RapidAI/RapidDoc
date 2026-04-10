"""
ocr_patch.py
------------
此模块用于优化和修正 RapidOCR 的推理逻辑。

包含：
1. 修复 TextDetector 的预处理逻辑。
2. 优化 PyTorch 推理性能，替换内部方法以支持更高效的推理。
3. openvino异步获取结果
4. 支持印章识别
"""
import sys
import types

import cv2
import pyclipper
import numpy as np
from typing import List, Tuple, Dict, Any
from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_det.utils import DetPreProcess, DBPostProcess
from rapidocr.inference_engine.base import get_engine
from rapid_doc.utils.model_utils import import_package
from importlib.metadata import version
rapidocr_version = version("rapidocr")


def patch_text_detector():
    """修复 TextDetector 的 get_preprocess 方法"""
    def new_get_preprocess(self, max_wh: int) -> DetPreProcess:
        limit_side_len = self.limit_side_len
        return DetPreProcess(limit_side_len, self.limit_type, self.mean, self.std)

    # 覆盖原始方法
    TextDetector.get_preprocess = new_get_preprocess


def patch_torch_ocr():
    """优化 PyTorch OCR 推理性能"""
    torch_ = import_package("torch")
    if not torch_:
        return  # 未安装 PyTorch，跳过
    if rapidocr_version >= "3.4.3":
        from rapidocr.inference_engine.pytorch.networks.backbones.rec_lcnetv3 import LearnableRepLayer, ConvBNLayer
    else:
        from rapidocr.networks.backbones.rec_lcnetv3 import LearnableRepLayer, ConvBNLayer

    from torch import nn
    import torch

    def _fuse_bn_tensor2(self, branch):
        """替换 LearnableRepLayer._fuse_bn_tensor 方法"""
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    LearnableRepLayer._fuse_bn_tensor = _fuse_bn_tensor2

    # 替换 rapidocr 的 TorchInferSession
    if rapidocr_version >= "3.4.3":
        from rapidocr.inference_engine.pytorch import main as rapidocr_torch
    else:
        from rapidocr.inference_engine import torch as rapidocr_torch
    from rapid_doc.model.ocr.torch import TorchInferSession
    rapidocr_torch.TorchInferSession = TorchInferSession

def patch_openvino_ocr():
    """优化 openvino OCR 推理，使用异步推理替代同步，修复并发识别报错"""
    torch_ = import_package("openvino")
    if not torch_:
        return  # 未安装 openvino，跳过
    try:
        import openvino
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    else:
        if "openvino.runtime" not in sys.modules:
            runtime_mod = types.ModuleType("openvino.runtime")
            runtime_mod.Core = openvino.Core
            sys.modules["openvino.runtime"] = runtime_mod

    # 替换 rapidocr 的 openvino
    from rapidocr import inference_engine
    from rapid_doc.model.ocr import openvino as self_openvino
    inference_engine.openvino = self_openvino

def patch_seal_det():
    # 1、为 DBPostProcess 添加 box_type 参数
    _old_init = DBPostProcess.__init__
    def patched_init(self, *args, **kwargs):
        self.box_type = kwargs.get("box_type", "quad")
        kwargs.pop('box_type', None)
        _old_init(self, *args, **kwargs)
    DBPostProcess.__init__ = patched_init

    # 2、TextDetector 把 cfg 里的 box_type 传到 DBPostProcess 里
    def textdetector__init__(self, cfg: Dict[str, Any]):
        self.limit_side_len = cfg.get("limit_side_len")
        self.limit_type = cfg.get("limit_type")
        self.mean = cfg.get("mean")
        self.std = cfg.get("std")
        self.preprocess_op = None

        post_process = {
            "thresh": cfg.get("thresh", 0.3),
            "box_thresh": cfg.get("box_thresh", 0.5),
            "max_candidates": cfg.get("max_candidates", 1000),
            "unclip_ratio": cfg.get("unclip_ratio", 1.6),
            "use_dilation": cfg.get("use_dilation", True),
            "score_mode": cfg.get("score_mode", "fast"),
            "box_type": cfg.get("box_type", "quad"),
        }
        self.postprocess_op = DBPostProcess(**post_process)

        self.session = get_engine(cfg.engine_type)(cfg)

    TextDetector.__init__ = textdetector__init__

    # 3、DBPostProcess.__call__ 支持大于四个点的 bbox
    def unclip(box, unclip_ratio):
        """unclip"""
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        distance = area * unclip_ratio / length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        try:
            expanded = np.array(offset.Execute(distance))
        except ValueError:
            expanded = np.array(offset.Execute(distance)[0])
        return expanded

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height, ):
        """_bitmap: single map with shape (1, H, W), whose values are binarized as {0, 1}"""

        bitmap = _bitmap
        height, width = bitmap.shape
        width_scale = dest_width / width
        height_scale = dest_height / height
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            if len(box) > 0:
                _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < self.min_size + 2:
                    continue
            else:
                continue

            box = np.array(box)
            for i in range(box.shape[0]):
                box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
                box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))

            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def __call__(
        self, pred: np.ndarray, ori_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[float]]:
        src_h, src_w = ori_shape
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        mask = segmentation[0]
        if self.dilation_kernel is not None:
            mask = cv2.dilate(
                np.array(segmentation[0]).astype(np.uint8), self.dilation_kernel
            )
        if self.box_type == "poly":
            boxes, scores = polygons_from_bitmap(self, pred[0], mask, src_w, src_h)
        else:
            boxes, scores = self.boxes_from_bitmap(pred[0], mask, src_w, src_h)
            boxes, scores = self.filter_det_res(boxes, scores, src_h, src_w)
        return boxes, scores
    DBPostProcess.__call__ = __call__

    _original_sorted_boxes = TextDetector.sorted_boxes
    def sorted_boxes(self, dt_boxes: np.ndarray) -> np.ndarray:
        """
        Equivalent NumPy implementation of the original bubble-adjusted sort.
        """
        if len(dt_boxes) == 0:
            return dt_boxes
        # 判断是否存在任意一个 box 的点数 > 4
        if any(len(box) > 4 for box in dt_boxes):
            return dt_boxes

        return _original_sorted_boxes(dt_boxes)

    TextDetector.sorted_boxes = sorted_boxes

def apply_ocr_patch():
    """统一入口：应用所有 OCR 相关补丁"""
    patch_text_detector()
    patch_torch_ocr()
    patch_openvino_ocr()
    patch_seal_det()
