"""恢复横线宽度不同、末端合并单元格未画底边的物理网格。"""

from __future__ import annotations

import statistics

from .geometry import _bbox_center_x, _bbox_center_y, _bbox_overlap_in_first, _bbox_overlap_in_smaller
from .models import _AxisLine, _PageSource, _TableCandidate


def detect_partial_rule_grids(source: _PageSource, excluded_bboxes: list) -> list[_TableCandidate]:
    """两条连续外竖框、共端列轨及重复行线确认整体范围，只补页内末端横边。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    tolerance = max(.75, .15 * em)
    vertical = [rule.bbox for rule in source.drawing_lines if rule.orientation == "vertical"]
    horizontal = [rule.bbox for rule in source.drawing_lines if rule.orientation == "horizontal"]
    output = []
    for top_rule in sorted(horizontal, key=lambda box: (box[1], box[0])):
        left, right = top_rule[0], top_rule[2]
        top = _bbox_center_y(top_rule)
        if right - left < 10 * em:
            continue
        outer = [next((box for box in vertical
                       if abs(_bbox_center_x(box) - x) <= tolerance
                       and abs(box[1] - top) <= tolerance
                       and box[3] - box[1] >= 5 * em), None) for x in (left, right)]
        if any(box is None for box in outer) or abs(outer[0][3] - outer[1][3]) > tolerance:
            continue
        bottom = min(box[3] for box in outer)
        bounds = (left, top_rule[1], right, max(box[3] for box in outer))
        if any(_bbox_overlap_in_smaller(bounds, box) >= .5 for box in excluded_bboxes):
            continue
        if any(_bbox_overlap_in_first(bounds, form) >= .9 for form in source.form_bboxes):
            continue
        if any(_bbox_overlap_in_first(bounds, item.core_bbox) >= .9 for item in output):
            continue
        tracks = [box for box in vertical
                  if left - tolerance <= _bbox_center_x(box) <= right + tolerance
                  and top - tolerance <= box[1] <= top + .4 * (bottom - top)
                  and abs(box[3] - bottom) <= tolerance]
        positions = []
        for x in sorted(_bbox_center_x(box) for box in tracks):
            if not positions or x - positions[-1] > tolerance:
                positions.append(x)
        if len(positions) < 4:
            continue
        # 所有行线端点必须落在同一列轨；缺失中间分隔线保持跨行单元格。
        partial = [box for box in horizontal
                   if top + tolerance < _bbox_center_y(box) <= bottom + tolerance
                   and left + tolerance < box[0] < right - tolerance
                   and abs(box[2] - right) <= tolerance
                   and any(abs(box[0] - x) <= tolerance for x in positions)]
        if len(partial) < 2:
            continue
        members = [line for line in lines if _bbox_overlap_in_first(line.bbox, bounds) >= .99]
        rows = []
        for line in sorted(members, key=lambda item: (_bbox_center_y(item.bbox), item.bbox[0])):
            if rows and abs(_bbox_center_y(line.bbox) - _bbox_center_y(rows[-1][0].bbox)) <= .5 * em:
                rows[-1].append(line)
            else:
                rows.append([line])
        repeated = 0
        for row in rows:
            occupied = set()
            for line in row:
                occupied.update(i for i, (a, b) in enumerate(zip(positions, positions[1:]))
                                if a < _bbox_center_x(line.bbox) < b)
            if len(occupied) >= 3:
                repeated += 1
        if repeated < 3:
            continue
        # 本页只补共同竖线终点处的底边；不推断文本行、内部列线或跨页内容。
        inferred = []
        if not any(abs(_bbox_center_y(box) - bottom) <= tolerance
                   and abs(box[0] - left) <= tolerance and abs(box[2] - right) <= tolerance
                   for box in horizontal):
            inferred.append(_AxisLine((left, bottom - .05, right, bottom + .05), .1, "horizontal"))
        output.append(_TableCandidate(
            bbox=bounds, local_bbox=bounds, core_bbox=bounds, angle=0,
            score=200, line_indices={line.source_index for line in members},
            # Drawing 已包含所有真实描边。底色矩形可能跨多个实列，不能与完整线网格争夺拓扑。
            inferred_grid=[*source.drawing_lines, *inferred], inferred_grid_authoritative=True,
        ))
    return output
