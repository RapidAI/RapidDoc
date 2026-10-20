"""几何载荷与独立矩形类型之间的纯值转换。"""

from __future__ import annotations

from ._contracts import Bbox


def char_bbox_values(bbox: object) -> tuple[float, float, float, float] | None:
    """读取固定四元坐标，拒绝结构不完整的几何对象。"""
    if isinstance(bbox, Bbox):
        bbox = bbox.bbox
        # 原生提取生成的内置浮点框已完成数值转换，直接读取以免重复调用 float。
        if (
            type(bbox) is list
            and len(bbox) == 4
            and type(bbox[0]) is float
            and type(bbox[1]) is float
            and type(bbox[2]) is float
            and type(bbox[3]) is float
        ):
            return bbox[0], bbox[1], bbox[2], bbox[3]
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        return tuple(float(value) for value in bbox)
    return None


__all__ = ["char_bbox_values"]
