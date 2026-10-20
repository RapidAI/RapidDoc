# Portions derived from pdftext 0.7.1, Copyright Vik Paruchuri, Apache-2.0.
# Changed in DocVortex: owned character types and source-index mappings replace upstream containers.
"""DocVortex 自有 PDF 字符与几何数据，不携带 PDFium 句柄。"""

from __future__ import annotations

from typing import Any, TypedDict


class Bbox:
    __slots__ = ("bbox", "ensure_nonzero_area")

    def __init__(self, bbox: list[float], ensure_nonzero_area: bool = False) -> None:
        """建立独立矩形对象并按需保证面积。"""
        if ensure_nonzero_area:
            bbox = list(bbox)
            bbox[2] = max(bbox[0], bbox[2] + 1)
            bbox[3] = max(bbox[1], bbox[3] + 1)
        self.bbox = bbox
        self.ensure_nonzero_area = ensure_nonzero_area

    def __getitem__(self, item: int | slice) -> float | list[float]:
        """读取矩形坐标。"""
        return self.bbox[item]

    def __repr__(self) -> str:
        """返回矩形的可读表示。"""
        return f"Bbox({self.bbox})"

    def __reduce__(self) -> tuple:
        # ensure_nonzero_area is already applied at construction; don't re-apply on unpickle
        """保存矩形数值以支持跨进程序列化。"""
        return (Bbox, (self.bbox,))

    def copy(self) -> Bbox:
        """复制矩形避免共享累加状态。"""
        return Bbox(list(self.bbox))

    @property
    def height(self) -> float:
        """计算矩形的 height 几何属性。"""
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self) -> float:
        """计算矩形的 width 几何属性。"""
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self) -> float:
        """计算矩形的 area 几何属性。"""
        return self.width * self.height

    @property
    def center(self) -> list[float]:
        """计算矩形的 center 几何属性。"""
        return [(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2]

    @property
    def size(self) -> list[float]:
        """计算矩形的 size 几何属性。"""
        return [self.width, self.height]

    @property
    def x_start(self) -> float:
        """计算矩形的 x_start 几何属性。"""
        return self.bbox[0]

    @property
    def y_start(self) -> float:
        """计算矩形的 y_start 几何属性。"""
        return self.bbox[1]

    @property
    def x_end(self) -> float:
        """计算矩形的 x_end 几何属性。"""
        return self.bbox[2]

    @property
    def y_end(self) -> float:
        """计算矩形的 y_end 几何属性。"""
        return self.bbox[3]

    def merge(self, other: Bbox) -> Bbox:
        """返回覆盖两个矩形的新对象。"""
        self_bbox = self.bbox
        other_bbox = other.bbox
        return Bbox(
            [
                min(self_bbox[0], other_bbox[0]),
                min(self_bbox[1], other_bbox[1]),
                max(self_bbox[2], other_bbox[2]),
                max(self_bbox[3], other_bbox[3]),
            ]
        )

    def merge_inplace(self, other: Bbox) -> Bbox:
        # Mutates this bbox; only safe on accumulator bboxes that aren't shared
        """仅修改当前累加矩形。"""
        self_bbox = self.bbox
        other_bbox = other.bbox
        if other_bbox[0] < self_bbox[0]:
            self_bbox[0] = other_bbox[0]
        if other_bbox[1] < self_bbox[1]:
            self_bbox[1] = other_bbox[1]
        if other_bbox[2] > self_bbox[2]:
            self_bbox[2] = other_bbox[2]
        if other_bbox[3] > self_bbox[3]:
            self_bbox[3] = other_bbox[3]
        return self

    def overlap_x(self, other: Bbox) -> float:
        """计算矩形的 overlap_x 几何属性。"""
        return max(0, min(self.bbox[2], other.bbox[2]) - max(self.bbox[0], other.bbox[0]))

    def overlap_y(self, other: Bbox) -> float:
        """计算矩形的 overlap_y 几何属性。"""
        return max(0, min(self.bbox[3], other.bbox[3]) - max(self.bbox[1], other.bbox[1]))

    def intersection_area(self, other: Bbox) -> float:
        """计算矩形的 intersection_area 几何属性。"""
        return self.overlap_x(other) * self.overlap_y(other)

    def intersection_pct(self, other: Bbox) -> float:
        """计算矩形的 intersection_pct 几何属性。"""
        if self.area <= 0:
            return 0

        intersection = self.intersection_area(other)
        return intersection / self.area

    def rotate(self, page_width: float, page_height: float, rotation: int) -> Bbox:
        """将矩形转换到旋转后的页面坐标。"""
        if rotation not in [0, 90, 180, 270]:
            raise ValueError("Rotation must be one of [0, 90, 180, 270] degrees.")

        x_min, y_min, x_max, y_max = self.bbox

        if rotation == 0:
            return Bbox(list(self.bbox))
        elif rotation == 90:
            new_x_min = page_height - y_max
            new_y_min = x_min
            new_x_max = page_height - y_min
            new_y_max = x_max
        elif rotation == 180:
            new_x_min = page_width - x_max
            new_y_min = page_height - y_max
            new_x_max = page_width - x_min
            new_y_max = page_height - y_min
        elif rotation == 270:
            new_x_min = y_min
            new_y_min = page_width - x_max
            new_x_max = y_max
            new_y_max = page_width - x_min

        # Ensure that x_min < x_max and y_min < y_max; must stay a list so
        # merge_inplace can mutate it
        rotated_bbox = [
            min(new_x_min, new_x_max),
            min(new_y_min, new_y_max),
            max(new_x_min, new_x_max),
            max(new_y_min, new_y_max),
        ]

        return Bbox(rotated_bbox)


class _CharValue(TypedDict):
    bbox: Bbox
    char: str
    rotation: float
    font: dict[str, Any]
    char_idx: int


class Char(_CharValue, total=False):
    """字符及原始索引映射，几何缺失保留为显式空值。"""

    source_indices: tuple[int, ...]
    raw_code: int
    is_generated: bool  # PDFium 的生成字符证据；未确认时省略，不删除原字符。
    text_object_id: int | None
    text_render_mode: int | None
    text_is_visible: bool
    writing_angle: float
    loose_bbox: tuple[float, float, float, float] | None
    tight_bbox: tuple[float, float, float, float] | None
    origin: tuple[float, float] | None


class Span(TypedDict):
    """基础字体片段，包含已解码文本及原始字符引用。"""

    bbox: Bbox
    text: str
    font: dict[str, Any]
    chars: list[Char]
    char_start_idx: int
    char_end_idx: int
    rotation: float
    url: str
    superscript: bool
    subscript: bool


class Line(TypedDict):
    """基础文本行，几何与片段顺序均保持可追溯。"""

    spans: list[Span]
    bbox: Bbox
    rotation: float


Spans = list[Span]
Lines = list[Line]
__all__ = ["Bbox", "Char", "Span", "Line", "Spans", "Lines"]
