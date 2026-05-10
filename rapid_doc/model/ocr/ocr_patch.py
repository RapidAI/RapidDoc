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

from rapidocr.cal_rec_boxes import CalRecBoxes
from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_det.utils import DetPreProcess, DBPostProcess
from rapidocr.ch_ppocr_rec.typings import WordInfo, WordType
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.inference_engine.base import get_engine
from rapidocr.utils.utils import has_chinese_char, quads_to_rect_bbox

from rapid_doc.utils.model_utils import import_package
from importlib.metadata import version
rapidocr_version = version("rapidocr")
# print(f"rapidocr_version=={rapidocr_version}")

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


def patch_word_box():
    """return_word_box 的时候，修复丢失空格问题"""
    def cal_ocr_word_box(
            self,
            rec_txt: str,
            bbox: np.ndarray,
            word_info: WordInfo,
            return_single_char_box: bool = False,
    ) -> Tuple[List[str], List[List[List[float]]], List[float]]:
        """Calculate the detection frame for each word based on the results of recognition and detection of ocr
        汉字坐标是单字的
        英语坐标是单词级别的
        三种情况：
        1. 全是汉字
        2. 全是英文
        3. 中英混合
        """
        if not rec_txt or word_info.line_txt_len == 0:
            return [], [], []

        bbox_points = quads_to_rect_bbox(bbox[None, ...])
        avg_col_width = (bbox_points[2] - bbox_points[0]) / word_info.line_txt_len

        is_all_en_num = all(v is WordType.EN_NUM for v in word_info.word_types)
        col_confs = get_col_confs(word_info)

        line_cols, char_widths, word_contents, content_confs = [], [], [], []
        for word, word_col in zip(word_info.words, word_info.word_cols):
            if is_all_en_num and not return_single_char_box:
                line_cols.append(word_col)
                word_contents.append("".join(word))
                content_confs.append(calc_word_conf(word_col, col_confs))
            else:
                line_cols.extend(word_col)
                word_contents.extend(word)
                content_confs.extend(calc_char_confs(word_col, col_confs))

            if len(word_col) == 1:
                continue

            avg_width = self.calc_avg_char_width(word_col, avg_col_width)
            char_widths.append(avg_width)

        avg_char_width = self.calc_all_char_avg_width(
            char_widths, bbox_points[0], bbox_points[2], len(rec_txt)
        )

        if is_all_en_num and not return_single_char_box:
            word_boxes = self.calc_en_num_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        else:
            word_boxes = self.calc_box(
                line_cols, avg_char_width, avg_col_width, bbox_points
            )
        return word_contents, word_boxes, content_confs

    def get_col_confs(word_info: WordInfo) -> dict:
        cols = [col for word_col in word_info.word_cols for col in word_col]
        return dict(zip(cols, word_info.confs))

    def calc_word_conf(word_col: List[int], col_confs: dict) -> float:
        confs = [col_confs[col] for col in word_col if col in col_confs]
        if not confs:
            return 0.0

        return round(float(np.mean(confs)), 5)

    def calc_char_confs(word_col: List[int], col_confs: dict) -> List[float]:
        return [calc_word_conf([col], col_confs) for col in word_col]
    # 替换 cal_ocr_word_box 方法
    CalRecBoxes.cal_ocr_word_box = cal_ocr_word_box


    def get_word_info(self, text: str, selection: np.ndarray) -> WordInfo:
        """
        Group the decoded characters and record the corresponding decoded positions.
        from https://github.com/PaddlePaddle/PaddleOCR/blob/fbba2178d7093f1dffca65a5b963ec277f1a6125/ppocr/postprocess/rec_postprocess.py#L70
        """
        word_list = []
        word_col_list = []
        state_list = []

        word_content = []
        word_col_content = []

        valid_col = np.where(selection)[0]
        if len(valid_col) <= 0:
            return WordInfo()

        col_width = np.zeros(valid_col.shape)
        col_width[1:] = valid_col[1:] - valid_col[:-1]
        col_width[0] = min(3 if has_chinese_char(text[0]) else 2, int(valid_col[0]))

        def flush_word():
            nonlocal state, word_content, word_col_content
            if not word_content:
                return

            word_list.append(word_content)
            word_col_list.append(word_col_content)
            state_list.append(state)
            word_content = []
            word_col_content = []

        state = None
        for c_i, char in enumerate(text):
            if char.isspace():
                flush_word()
                word_list.append([char])
                word_col_list.append([int(valid_col[c_i])])
                state_list.append(WordType.EN_NUM)
                state = None
                continue

            c_state = WordType.CN if has_chinese_char(char) else WordType.EN_NUM
            if state is None:
                state = c_state

            if state != c_state or col_width[c_i] > 5:
                flush_word()
                state = c_state

            word_content.append(char)
            word_col_content.append(int(valid_col[c_i]))

        flush_word()

        return WordInfo(words=word_list, word_cols=word_col_list, word_types=state_list)
    # 替换 get_word_info 方法
    CTCLabelDecode.get_word_info = get_word_info

def apply_ocr_patch():
    """统一入口：应用所有 OCR 相关补丁"""
    patch_text_detector()
    patch_torch_ocr()
    patch_openvino_ocr()
    patch_seal_det()
    patch_word_box()
