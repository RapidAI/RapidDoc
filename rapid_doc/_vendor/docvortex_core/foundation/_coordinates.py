"""不依赖 PDF 运行时或图像编解码的坐标原语。"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from loguru import logger

from ..schema import BBox


def bbox_to_quad(bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    """将轴对齐矩形转换为按边界顺序排列的四点坐标。"""
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """计算 bbox 中心点，用于判断图片或公式应归属哪个表格。"""
    return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0


def normalize_quarter_turn_angle(angle: Any) -> int:
    """规范视觉块角度为 0/90/180/270，无法识别的角度按 0 处理。"""
    try:
        normalized_angle = int(float(angle or 0)) % 360
    except (TypeError, ValueError):
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    if normalized_angle not in {0, 90, 180, 270}:
        logger.warning(f"Unsupported visual block angle: {angle}, using 0")
        return 0
    return normalized_angle


def rotate_bbox(
    bbox: BBox,
    image_width: float,
    image_height: float,
    angle: int,
) -> BBox:
    """把原表格裁图中的 bbox 同步转换到旋转后裁图坐标系。"""
    x0, y0, x1, y1 = [float(value) for value in bbox]
    if angle == 270:
        # 顺时针旋转 90 度后，新 x 轴来自原 y 轴的反方向。
        return (image_height - y1, x0, image_height - y0, x1)
    if angle == 90:
        # 逆时针旋转 90 度后，新 y 轴来自原 x 轴的反方向。
        return (y0, image_width - x1, y1, image_width - x0)
    if angle == 180:
        return (image_width - x1, image_height - y1, image_width - x0, image_height - y0)
    return (x0, y0, x1, y1)


def convert_bbox(
    bbox: BBox | None,
    *,
    source_space: Literal["unit", "pixel", "point"],
    target_space: Literal["unit", "pixel", "point"],
    page_size: tuple[float, float],
    render_scale: float = 1.0,
    clip: bool = False,
) -> BBox | None:
    """显式转换坐标空间；page_size 使用 PDF point，render_scale 表示每 point 像素数。"""
    spaces = {"unit", "pixel", "point"}
    if source_space not in spaces or target_space not in spaces:
        raise ValueError("Unknown coordinate space")
    if bbox is None or len(bbox) != 4 or min(page_size) <= 0 or render_scale <= 0:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    width, height = page_size
    sizes = {"unit": (1.0, 1.0), "point": (width, height), "pixel": (width * render_scale, height * render_scale)}
    source_width, source_height = sizes[source_space]
    target_width, target_height = sizes[target_space]
    if source_space == "pixel" and target_space == "point":
        x0, x1, y0, y1 = x0 / render_scale, x1 / render_scale, y0 / render_scale, y1 / render_scale
    elif target_space == "unit":
        x0, x1, y0, y1 = x0 / source_width, x1 / source_width, y0 / source_height, y1 / source_height
    else:
        scale_x, scale_y = target_width / source_width, target_height / source_height
        x0, x1 = x0 * scale_x, x1 * scale_x
        y0, y1 = y0 * scale_y, y1 * scale_y
    if clip:
        x0, x1 = max(0.0, min(target_width, x0)), max(0.0, min(target_width, x1))
        y0, y1 = max(0.0, min(target_height, y0)), max(0.0, min(target_height, y1))
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def normalize_bbox(raw_bbox: Any) -> BBox | None:
    """校验模型 block 的四点框，返回可用于面积包含判断的浮点坐标。"""
    try:
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox
