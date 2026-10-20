"""PDF 表格视觉行几何；保留原有认领顺序与判定规则。"""

from __future__ import annotations
from ....schema import BBox
from .models import _VisualRow
from .geometry import (
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union_many,
)


def _clip_visual_row_to_corridor(
    row: _VisualRow,
    corridor_bbox: BBox,
    *,
    margin: float,
) -> _VisualRow | None:
    """仅保留横向走廊内的片段，避免同基线的另一栏文本污染表格区域。"""

    fragments = [
        fragment
        for fragment in row.fragments
        if corridor_bbox[0] - margin <= _bbox_center_x(fragment.local_bbox) <= corridor_bbox[2] + margin
    ]
    if not fragments:
        return None
    fragments.sort(key=lambda fragment: fragment.local_bbox[0])
    return _VisualRow(
        fragments=fragments,
        center_y=sum(_bbox_center_y(fragment.local_bbox) for fragment in fragments) / len(fragments),
        bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
        visual_row_id=row.visual_row_id,
    )
