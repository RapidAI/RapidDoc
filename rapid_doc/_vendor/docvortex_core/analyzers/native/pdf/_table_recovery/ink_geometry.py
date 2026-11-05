"""以独立墨迹带证明布局框误合行，再统一修正表格内部纵向几何。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import statistics
from typing import TYPE_CHECKING

from .....schema import BBox
from .contracts import NativeTableInput
from .geometry import bbox_union, normalize_angle, normalize_bbox, page_bbox_to_table_local

if TYPE_CHECKING:
    from .text import _PendingGlyph


@dataclass(frozen=True, slots=True)
class _InkAnchor:
    """保存与语言、字体名称无关的字母数字墨迹及原始基线证据。"""

    source_index: int
    baseline: float
    bbox: BBox
    layout_bbox: BBox
    inflation: float


@dataclass(frozen=True, slots=True)
class _InkBand:
    """汇总具有至少两个锚字支撑的独立基线带。"""

    baseline: float
    height: float
    bbox: BBox
    layout_bbox: BBox
    inflation: float


def _body_ink_height(heights: Sequence[float]) -> float:
    """按最近秩取得第七十五百分位高度，避免低矮字形主导主体尺度。"""
    return sorted(heights)[math.ceil(0.75 * len(heights)) - 1]


def _baseline_bands(anchors: list[_InkAnchor], body_height: float) -> list[_InkBand]:
    """按基线中位数聚带，只保留具有独立复数字形支撑的组。"""
    groups: list[list[_InkAnchor]] = []
    for anchor in sorted(anchors, key=lambda item: (item.baseline, item.bbox[0], item.source_index)):
        if groups and abs(anchor.baseline - statistics.median(item.baseline for item in groups[-1])) <= 0.15 * body_height:
            groups[-1].append(anchor)
        else:
            groups.append([anchor])
    return [
        _InkBand(
            baseline=float(statistics.median(item.baseline for item in group)),
            height=_body_ink_height([item.bbox[3] - item.bbox[1] for item in group]),
            bbox=bbox_union(item.bbox for item in group),
            layout_bbox=(
                min(item.layout_bbox[0] for item in group),
                float(statistics.median(item.layout_bbox[1] for item in group)),
                max(item.layout_bbox[2] for item in group),
                float(statistics.median(item.layout_bbox[3] for item in group)),
            ),
            inflation=float(statistics.median(item.inflation for item in group)),
        )
        for group in groups
        if len(group) >= 2
    ]


def _bands_prove_false_merge(upper: _InkBand, lower: _InkBand, median_height: float) -> bool:
    """联合大小、基线、墨迹留白和横向交叠，排除上下标及异列错位。"""
    smaller_height = min(upper.height, lower.height)
    larger_height = max(upper.height, lower.height)
    if (
        min(upper.inflation, lower.inflation) < 2.0
        or smaller_height < (2.0 / 3.0) * larger_height
        or lower.baseline - upper.baseline < larger_height
        or lower.bbox[1] - upper.bbox[3] < 0.10 * smaller_height
    ):
        return False
    overlap = min(upper.bbox[2], lower.bbox[2]) - max(upper.bbox[0], lower.bbox[0])
    smaller_width = min(upper.bbox[2] - upper.bbox[0], lower.bbox[2] - lower.bbox[0])
    if overlap < 0.50 * smaller_width:
        return False
    from ....._compute_backend import get_native

    # 排除异列居中内容扩张整行外框形成的桥接；两带本身也必须被既有组行规则误并。
    # 纵向取字形边界中位数，不能让孤立大字或上下标代表整条正文基线。
    rows = get_native().table_visual_rows([upper.layout_bbox, lower.layout_bbox], [0, 1], median_height)
    return len(rows) == 1


def _selected_ink_boxes(table_input: NativeTableInput, pending: Sequence[_PendingGlyph]) -> dict[int, BBox] | None:
    """完整校验当前选中字形的墨迹来源，任一缺失或歧义都放弃整表校准。"""
    source_indices = {glyph.source_index for glyph in pending}
    if len(source_indices) != len(pending) or any(glyph.writing_angle != 0 for glyph in pending):
        return None
    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None or not all(math.isfinite(value) for value in table_bbox):
        return None
    angle = normalize_angle(table_input.angle)
    boxes: dict[int, BBox] = {}
    for index, char in enumerate(table_input.chars):
        try:
            source_index = int(char.get("char_idx", index))
        except (TypeError, ValueError, OverflowError):
            source_index = index
        if source_index not in source_indices:
            continue
        if source_index in boxes:
            return None
        try:
            ink = normalize_bbox(char.get("tight_bbox"))
        except OverflowError:
            return None
        if ink is None or not all(math.isfinite(value) for value in ink):
            return None
        local = page_bbox_to_table_local(ink, table_bbox, angle)
        if local is None:
            return None
        boxes[source_index] = local
    return boxes if len(boxes) == len(pending) else None


def calibrate_inflated_table_glyphs(
    table_input: NativeTableInput,
    pending: list[_PendingGlyph],
    original_rows: list[list[_PendingGlyph]],
    *,
    median_height: float,
) -> list[_PendingGlyph] | None:
    """仅在旧视觉行合并了独立正文墨迹带时，为整表创建纵向校准副本。"""
    if not any(len(row) >= 4 for row in original_rows):
        return None
    ink_boxes = _selected_ink_boxes(table_input, pending)
    if ink_boxes is None:
        return None
    anchors: dict[int, _InkAnchor] = {}
    for glyph in pending:
        if (
            not any(char.isalnum() for char in glyph.text)
            or glyph.origin is None
            or not all(math.isfinite(value) for value in (*glyph.bbox, *glyph.origin))
        ):
            continue
        ink = ink_boxes[glyph.source_index]
        anchors[glyph.source_index] = _InkAnchor(
            source_index=glyph.source_index,
            baseline=glyph.origin[1],
            bbox=ink,
            layout_bbox=glyph.bbox,
            inflation=(glyph.bbox[3] - glyph.bbox[1]) / (ink[3] - ink[1]),
        )
    if len(anchors) < 4:
        return None
    body_height = _body_ink_height([anchor.bbox[3] - anchor.bbox[1] for anchor in anchors.values()])
    for row in original_rows:
        bands = _baseline_bands([anchors[g.source_index] for g in row if g.source_index in anchors], body_height)
        if not any(
            _bands_prove_false_merge(upper, lower, median_height)
            for index, upper in enumerate(bands)
            for lower in bands[index + 1 :]
        ):
            continue
        # 字符已按原始几何筛选完毕；薄负号等墨迹不能再次经过高度过滤。
        # 横向保留排版 advance，避免把字母之间的正常墨迹留白误认为词间空格。
        return [
            replace(
                glyph,
                bbox=(glyph.bbox[0], ink_boxes[glyph.source_index][1], glyph.bbox[2], ink_boxes[glyph.source_index][3]),
            )
            for glyph in pending
        ]
    return None


__all__ = ["calibrate_inflated_table_glyphs"]
