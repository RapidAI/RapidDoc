"""提供表格恢复与文本投影共享的局部几何，不改变各业务层校验策略。"""

from __future__ import annotations

from ....document.pdf.text._contracts import Bbox
from ....schema import BBox


def normalize_bbox(value: object) -> BBox | None:
    """把任意四元组规范为有效浮点 bbox，异常或退化框返回空。"""

    # 自有 Bbox 的下标协议仅转发底层数组，直接读取可避免每个框五次 Python 调用。
    # 只匹配精确类型，第三方可迭代对象及覆盖下标行为的子类仍走原协议。
    if type(value) is Bbox:
        value = value.bbox
    try:
        x0, y0, x1, y1 = [float(item) for item in value]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    # 与两个元素的稳定排序使用相同比较方向，保留相等值及 NaN 的顺序。
    left, right = (x1, x0) if x1 < x0 else (x0, x1)
    top, bottom = (y1, y0) if y1 < y0 else (y0, y1)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def rotate_local_bbox(
    bbox: BBox,
    width: float,
    height: float,
    angle: int,
) -> BBox:
    """把表格裁剪框内 bbox 转换到正向表格局部坐标。"""

    x0, y0, x1, y1 = bbox
    if angle == 270:
        return height - y1, x0, height - y0, x1
    if angle == 90:
        return y0, width - x1, y1, width - x0
    if angle == 180:
        return width - x1, height - y1, width - x0, height - y0
    return bbox


__all__ = ["normalize_bbox", "rotate_local_bbox"]
