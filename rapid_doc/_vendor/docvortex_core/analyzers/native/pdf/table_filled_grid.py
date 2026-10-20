"""PDF 填充单元格候选；保留原有认领顺序与判定规则。"""

from __future__ import annotations
import re
import statistics

from ....document.pdf._document import PDFPathInfo
from ....schema import BBox
from .geometry import (
    _bbox_area,
    _bbox_overlap_in_smaller,
    _bbox_union_many,
    _bbox_axis_overlap_ratio,
)
from .models import _TableCandidate, _PageSource, _TableAnnotation, _AxisLine
from .table_constants import (
    _FILLED_GRID_MAX_PAGE_AREA_RATIO,
    _FILLED_GRID_MIN_PAGE_AREA_RATIO,
    _FILLED_GRID_MIN_PAGE_HEIGHT_RATIO,
    _FILLED_GRID_MIN_PAGE_WIDTH_RATIO,
)


def _detect_tiled_fill_table_candidates(source: _PageSource, excluded: list[BBox]) -> list[_TableCandidate]:
    """相接的填充单元格与贯穿列线共同证明完整网格，保留跨行单元及无底纹表头。"""
    width, height = source.page_size
    if not has_inflated_native_cjk_metrics(source.chars):
        return []
    tolerance = max(0.5, min(width, height) * 0.001)
    rectangles = [
        path.bbox
        for path in source.path_infos
        if path.form_depth == 0
        and path.fill_visible
        and len(path.rectangle_bboxes) == 1
        and path.segment_count == 5
        and path.bbox[2] - path.bbox[0] >= 6 * tolerance
        and path.bbox[3] - path.bbox[1] >= 4 * tolerance
        and _bbox_area(path.bbox) < 0.65 * width * height
        and not any(_bbox_overlap_in_smaller(path.bbox, box) >= 0.5 for box in excluded)
    ]
    pending = set(range(len(rectangles)))
    result = []
    while pending:
        members = [pending.pop()]
        for index in members:
            box = rectangles[index]
            touching = [
                other
                for other in pending
                if any(
                    min(abs(box[axis] - rectangles[other][axis + 2]), abs(box[axis + 2] - rectangles[other][axis])) <= tolerance
                    and _bbox_axis_overlap_ratio(box, rectangles[other], axis="y" if axis == 0 else "x") >= 0.8
                    for axis in (0, 1)
                )
            ]
            pending.difference_update(touching)
            members.extend(touching)
        if len(members) < 6:
            continue
        bounds = _bbox_union_many([rectangles[index] for index in members])
        columns = [
            rule
            for rule in source.drawing_lines
            if rule.orientation == "vertical"
            and bounds[0] + tolerance < (rule.bbox[0] + rule.bbox[2]) / 2 < bounds[2] - tolerance
            and _bbox_axis_overlap_ratio(rule.bbox, bounds, axis="y") >= 0.75
        ]
        rows = [
            rule
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and bounds[1] + tolerance < (rule.bbox[1] + rule.bbox[3]) / 2 < bounds[3] - tolerance
            and _bbox_axis_overlap_ratio(rule.bbox, bounds, axis="x") >= 0.8
        ]
        if not columns or len(rows) < 2:
            continue
        # 连续列轨可越过未填充的最后一行，但绝不越过另一张表的独立表头。
        bottom = max(rule.bbox[3] for rule in columns)
        if bottom > bounds[3] and bottom - bounds[3] <= 0.3 * (bounds[3] - bounds[1]):
            bounds = (bounds[0], bounds[1], bounds[2], bottom)
        line_members = []
        above = []
        for line in source.lines:
            ink = [char.get("tight_bbox") for char in line.chars or [] if char.get("tight_bbox")]
            box = _bbox_union_many(ink) if ink else line.bbox
            center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if bounds[0] - tolerance <= center_x <= bounds[2] + tolerance:
                if bounds[1] <= center_y <= bounds[3] + tolerance:
                    line_members.append(line)
                elif 0 < bounds[1] - box[3] <= 1.5 * max(box[3] - box[1], tolerance) and len(line.text) <= 16:
                    above.append((line, box))
        inferred_grid = []
        header_slack = max(2 * tolerance, 0.25 * statistics.median(box[3] - box[1] for _, box in above)) if above else 0
        if len(above) >= 2 and max(box[1] for _, box in above) - min(box[1] for _, box in above) <= header_slack:
            body_top = bounds[1]
            bounds = (bounds[0], min(box[1] for _, box in above), bounds[2], bounds[3])
            line_members.extend(line for line, _ in above)
            # 多列同基线的无底纹表头建立一个独立行带，列轨延伸到新顶边以免恢复器静默丢字。
            inferred_grid.append(
                _AxisLine((bounds[0], bounds[1] - tolerance / 2, bounds[2], bounds[1] + tolerance / 2), tolerance, "horizontal")
            )
            tracks = sorted({bounds[0], bounds[2], *((rule.bbox[0] + rule.bbox[2]) / 2 for rule in columns)})
            inferred_grid.extend(
                _AxisLine((x - tolerance / 2, bounds[1], x + tolerance / 2, body_top), tolerance, "vertical") for x in tracks
            )
            inferred_grid.extend(
                rule
                for rule in source.drawing_lines
                if bounds[0] - tolerance <= (rule.bbox[0] + rule.bbox[2]) / 2 <= bounds[2] + tolerance
                and bounds[1] - tolerance <= (rule.bbox[1] + rule.bbox[3]) / 2 <= bounds[3] + tolerance
            )
        result.append(
            _TableCandidate(
                bounds,
                bounds,
                0,
                130 + len(members),
                core_bbox=bounds,
                line_indices={line.source_index for line in line_members},
                inferred_grid=inferred_grid,
                inferred_grid_authoritative=bool(inferred_grid),
            )
        )
    result.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    for before, after in zip(result, result[1:]):
        # 相邻独立网格间的短整行题名不能进入上一表的无底纹尾部。
        em = statistics.median(
            char["bbox"][2] - char["bbox"][0] for char in source.chars if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", ""))
        )
        if not 0 <= after.bbox[1] - before.bbox[3] <= 2 * em:
            continue
        captions = [
            line
            for line in source.lines
            if line.source_index in before.line_indices
            and re.fullmatch(r"[\u4e00-\u9fff]{8,35}", line.text.strip())
            and abs(line.bbox[0] - before.bbox[0]) <= em
            and line.bbox[2] - line.bbox[0] < 0.6 * (before.bbox[2] - before.bbox[0])
            and before.bbox[3] - 2 * em <= line.bbox[1] < after.bbox[1]
        ]
        if not captions:
            continue
        caption = max(captions, key=lambda line: line.bbox[1])
        body = [line for line in source.lines if line.source_index in before.line_indices and line is not caption]
        bottom = max(line.bbox[3] for line in body if line.bbox[1] < caption.bbox[1]) + tolerance
        bounds = (before.bbox[0], before.bbox[1], before.bbox[2], min(bottom, caption.bbox[1] - tolerance))
        before.bbox = before.core_bbox = before.local_bbox = bounds
        before.line_indices.discard(caption.source_index)
        after.annotations.append(
            _TableAnnotation("caption", caption.bbox, {caption.source_index}, {caption.source_index: caption.bbox})
        )
    return result


def has_inflated_native_cjk_metrics(chars: list) -> bool:
    """汉字墨迹正常而宽松框高度超过两倍字宽时，允许用路径补偿失真的文本行尺度。"""
    anchors = [
        char
        for char in chars
        if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", "")) and char.get("tight_bbox") and char.get("bbox")
    ]
    if len(anchors) < 20:
        return False
    width = statistics.median(char["bbox"][2] - char["bbox"][0] for char in anchors)
    loose = statistics.median(char["bbox"][3] - char["bbox"][1] for char in anchors)
    ink = statistics.median(char["tight_bbox"][3] - char["tight_bbox"][1] for char in anchors)
    return width > 0 and loose >= 2.2 * width and 0.5 * width <= ink <= 1.4 * width


def _detect_filled_grid_table_candidates(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    excluded_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """仅按根层填充矩形的嵌套行带识别精确表格外框。"""

    page_width, page_height = page_size
    page_area = page_width * page_height
    if page_area <= 0:
        return []
    excluded_bboxes = excluded_bboxes or []
    rectangles = [
        path_info.bbox
        for path_info in path_infos
        if path_info.form_depth == 0
        and path_info.fill_visible
        and path_info.segment_count == 5
        and _bbox_area(path_info.bbox) > 0
    ]
    evidence: list[tuple[_TableCandidate, int, float, float]] = []
    for outer_bbox in rectangles:
        outer_width = outer_bbox[2] - outer_bbox[0]
        outer_height = outer_bbox[3] - outer_bbox[1]
        outer_area = _bbox_area(outer_bbox)
        if not (
            _FILLED_GRID_MIN_PAGE_AREA_RATIO <= outer_area / page_area <= _FILLED_GRID_MAX_PAGE_AREA_RATIO
            and outer_width >= _FILLED_GRID_MIN_PAGE_WIDTH_RATIO * page_width
            and outer_height >= _FILLED_GRID_MIN_PAGE_HEIGHT_RATIO * page_height
        ):
            continue
        if any(_bbox_overlap_in_smaller(outer_bbox, excluded_bbox) >= 0.5 for excluded_bbox in excluded_bboxes):
            continue

        cells = _select_maximal_filled_grid_cells(rectangles, outer_bbox)
        bands = _group_filled_grid_cells_into_bands(cells, outer_bbox)
        accepted_bands = [band for band in bands if _filled_grid_band_covers_outer_width(band, outer_bbox)]
        if len(accepted_bands) < 4:
            continue
        accepted_bands.sort(
            key=lambda band: (
                min(cell[1] for cell in band),
                min(cell[0] for cell in band),
            )
        )
        covered_height = sum(max(cell[3] for cell in band) - min(cell[1] for cell in band) for band in accepted_bands)
        vertical_coverage = covered_height / outer_height
        if vertical_coverage < 0.75:
            continue
        if any(
            min(cell[1] for cell in next_band) - max(cell[3] for cell in current_band) > 0.05 * outer_height
            for current_band, next_band in zip(
                accepted_bands,
                accepted_bands[1:],
            )
        ):
            continue
        candidate = _TableCandidate(
            bbox=outer_bbox,
            local_bbox=outer_bbox,
            angle=0,
            score=float(100.0 + len(accepted_bands) + vertical_coverage),
            core_bbox=outer_bbox,
            line_indices=set(),
        )
        evidence.append(
            (
                candidate,
                len(accepted_bands),
                vertical_coverage,
                outer_area,
            )
        )

    output: list[_TableCandidate] = []
    for candidate, _band_count, _coverage, _outer_area in sorted(
        evidence,
        key=lambda item: (item[1], item[2], item[3]),
        reverse=True,
    ):
        if any(_bbox_overlap_in_smaller(candidate.bbox, accepted.bbox) >= 0.9 for accepted in output):
            continue
        output.append(candidate)
    return sorted(output, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]))


def _select_maximal_filled_grid_cells(
    rectangles: list[BBox],
    outer_bbox: BBox,
) -> list[BBox]:
    """移除边缘细条、重复 Path 和同跨度半行底纹，保留最大单元格。"""

    outer_width = outer_bbox[2] - outer_bbox[0]
    outer_height = outer_bbox[3] - outer_bbox[1]
    outer_area = _bbox_area(outer_bbox)
    x_tolerance = 0.01 * outer_width
    y_tolerance = 0.02 * outer_height
    nested_rectangles = [
        rectangle
        for rectangle in rectangles
        if rectangle != outer_bbox
        and _bbox_is_contained_with_tolerance(
            rectangle,
            outer_bbox,
            x_tolerance,
            y_tolerance,
        )
        and _bbox_area(rectangle) <= 0.8 * outer_area
        and rectangle[2] - rectangle[0] >= 0.1 * outer_width
        and rectangle[3] - rectangle[1] >= 0.06 * outer_height
    ]
    output: list[BBox] = []
    for rectangle in sorted(
        nested_rectangles,
        key=lambda bbox: (-_bbox_area(bbox), bbox[1], bbox[0]),
    ):
        if any(
            rectangle != other
            and _bbox_is_contained_with_tolerance(
                rectangle,
                other,
                x_tolerance,
                y_tolerance,
            )
            and _bbox_area(other) >= 1.08 * _bbox_area(rectangle)
            and _bbox_area(other) <= 0.8 * outer_area
            and abs((rectangle[2] - rectangle[0]) - (other[2] - other[0])) <= 0.01 * outer_width
            for other in nested_rectangles
        ):
            continue
        if any(_bbox_overlap_in_smaller(rectangle, accepted) >= 0.95 for accepted in output):
            continue
        output.append(rectangle)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _bbox_is_contained_with_tolerance(
    inner_bbox: BBox,
    outer_bbox: BBox,
    x_tolerance: float,
    y_tolerance: float,
) -> bool:
    """按横纵独立容差判断一个矩形是否完整位于另一个矩形内。"""

    return (
        inner_bbox[0] >= outer_bbox[0] - x_tolerance
        and inner_bbox[1] >= outer_bbox[1] - y_tolerance
        and inner_bbox[2] <= outer_bbox[2] + x_tolerance
        and inner_bbox[3] <= outer_bbox[3] + y_tolerance
    )


def _group_filled_grid_cells_into_bands(
    cells: list[BBox],
    outer_bbox: BBox,
) -> list[list[BBox]]:
    """按相近上下边界把最大填充单元格聚成水平行带。"""

    y_tolerance = 0.02 * (outer_bbox[3] - outer_bbox[1])
    bands: list[list[BBox]] = []
    for cell in sorted(cells, key=lambda bbox: (bbox[1], bbox[0], bbox[3])):
        target = next(
            (band for band in bands if abs(cell[1] - band[0][1]) <= y_tolerance and abs(cell[3] - band[0][3]) <= y_tolerance),
            None,
        )
        if target is None:
            bands.append([cell])
        else:
            target.append(cell)
    return bands


def _filled_grid_band_covers_outer_width(
    band: list[BBox],
    outer_bbox: BBox,
) -> bool:
    """校验单个填充行带的单元格数量、横向覆盖、间隙和端点。"""

    if len(band) < 2:
        return False
    outer_width = outer_bbox[2] - outer_bbox[0]
    segments = sorted((cell[0], cell[2]) for cell in band)
    current_left, current_right = segments[0]
    covered_width = 0.0
    maximum_gap = 0.0
    for left, right in segments[1:]:
        if left <= current_right:
            current_right = max(current_right, right)
            continue
        covered_width += current_right - current_left
        maximum_gap = max(maximum_gap, left - current_right)
        current_left, current_right = left, right
    covered_width += current_right - current_left
    return (
        covered_width >= 0.9 * outer_width
        and maximum_gap <= 0.01 * outer_width
        and abs(min(cell[0] for cell in band) - outer_bbox[0]) <= 0.02 * outer_width
        and abs(max(cell[2] for cell in band) - outer_bbox[2]) <= 0.02 * outer_width
    )
