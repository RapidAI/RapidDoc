"""PDF 表格检测编排与续表标题；保留原有认领顺序与判定规则。"""

from __future__ import annotations
import re
import math
import statistics
from dataclasses import replace
from ....schema import BBox
from .models import _AxisLine, _LineItem, _PageSource, _TableAnnotation, _TableCandidate
from . import geometry as _overlap_geometry
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _expand_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)

from .table_constants import _TABLE_CONTINUATION_RE
from .table_annotations import _table_caption_candidates
from .inline.detection import _font_styles_from_metadata
from .table_filled_grid import _detect_filled_grid_table_candidates, _detect_tiled_fill_table_candidates
from .table_rules import (
    _RuleCandidateDraft,
    _build_closed_rule_grid_candidates,
    _build_fragments,
    _build_rule_table_candidates,
    _cluster_fragment_rows,
    _connected_rule_grid_bboxes,
    _median_fragment_height,
    _merge_owned_table_candidates,
)


def _detect_table_candidates(
    source: _PageSource,
    excluded_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """按文本方向融合横线区间、规则文本和填充行带，生成表格候选。"""

    excluded_bboxes = excluded_bboxes or []
    filled_grid_candidates = _detect_filled_grid_table_candidates(
        source.path_infos,
        source.page_size,
        excluded_bboxes,
    )
    tiled = _detect_tiled_fill_table_candidates(source, excluded_bboxes)
    filled_grid_candidates = [
        candidate
        for candidate in filled_grid_candidates
        if not any(_bbox_overlap_in_smaller(candidate.bbox, item.bbox) >= 0.9 for item in tiled)
    ] + tiled
    filled_grid_bboxes = [candidate.bbox for candidate in filled_grid_candidates]
    rule_candidates: list[_TableCandidate] = []
    angles = sorted({line.angle for line in source.lines})
    for angle in angles:
        angle_lines = [line for line in source.lines if line.angle == angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(angle_lines, source.page_size)
        if not fragments:
            continue
        median_height = _median_fragment_height(fragments)
        rows = _cluster_fragment_rows(fragments, median_height)
        local_axis_lines = _transform_axis_lines(
            source.drawing_lines,
            source.page_size,
            angle,
        )
        caption_candidates = _table_caption_candidates(
            angle_lines,
            source.page_size,
            angle,
            median_height,
        )
        local_excluded_bboxes = [
            _expand_bbox(
                _rotate_bbox_to_upright(bbox, source.page_size, angle),
                min(2.5 * median_height, 0.01 * source.page_size[0])
                if any(bbox == item.bbox for item in tiled)
                else 2.5 * median_height,
            )
            for bbox in [*excluded_bboxes, *filled_grid_bboxes]
        ]
        angle_rule_candidates = _build_rule_table_candidates(
            rows,
            angle_lines,
            source.page_size,
            angle,
            median_height,
            local_axis_lines,
            path_infos=source.path_infos,
            excluded_bboxes=local_excluded_bboxes,
            caption_candidates=caption_candidates,
            defer_materialization=True,
        )
        rule_candidates.extend(angle_rule_candidates)
        # 规则候选与闭合网格消费同一批横线/竖轨；复用上下文冻结的分量，
        # 避免 pollutant 这类多表页面连续两次重建相同连通关系。
        shared_grid_components = angle_rule_candidates[0].context.grid_components if angle_rule_candidates else None
        rule_candidates.extend(
            _build_closed_rule_grid_candidates(
                rows,
                angle_lines,
                source.page_size,
                angle,
                median_height,
                local_axis_lines,
                local_excluded_bboxes,
                caption_candidates,
                grid_components=shared_grid_components,
                previous_table_grid=source.previous_table_grid,
                page_index=source.page_index,
                form_bboxes=[_rotate_bbox_to_upright(bbox, source.page_size, angle) for bbox in source.form_bboxes],
                leading_obstacles=[
                    _rotate_bbox_to_upright(box, source.page_size, angle)
                    for box in [*excluded_bboxes, *source.image_bboxes, *source.signature_bboxes, *source.form_bboxes]
                ],
            )
        )
    merged_rule_candidates = [
        candidate
        for candidate in _merge_owned_table_candidates(_exclude_prose_spanning_rule_drafts(source, rule_candidates))
        if not any(_bbox_overlap_in_smaller(candidate.bbox, filled_bbox) >= 0.2 for filled_bbox in filled_grid_bboxes)
    ]
    base = [*filled_grid_candidates, *merged_rule_candidates, *_detect_shaded_header_tables(source, excluded_bboxes)]
    from .table_open_grid import detect_partial_rule_grids

    recovered = [
        item
        for item in [*detect_partial_rule_grids(source, excluded_bboxes),
                     *_detect_open_vertical_tables(source, excluded_bboxes)]
        if not any(_bbox_overlap_in_first(item.core_bbox, candidate.core_bbox or candidate.bbox) >= 0.98 for candidate in base)
    ]
    # 长列线确认的完整表优先于其内部横线生成的片段；完整旧候选保持自己的物理边界和注释。
    base = [
        candidate
        for candidate in base
        if not any(_bbox_overlap_in_first(candidate.core_bbox or candidate.bbox, item.core_bbox) >= 0.9 for item in recovered)
    ]
    year_tables = _detect_year_header_numeric_tables(source, excluded_bboxes)
    numeric_tables = _detect_aligned_numeric_tables(source, excluded_bboxes)
    # 新检测只补缺失区域或明显缺列的局部表；完整旧外框交给结构恢复检查欠分割。
    numeric_tables = [
        candidate
        for candidate in numeric_tables
        if not any(
            _bbox_overlap_in_smaller(old.core_bbox or old.bbox, candidate.bbox) >= 0.9
            and (old.bbox[2] - old.bbox[0]) >= 0.95 * (candidate.bbox[2] - candidate.bbox[0])
            and (old.bbox[3] - old.bbox[1]) >= 0.85 * (candidate.bbox[3] - candidate.bbox[1])
            for old in [*base, *recovered]
        )
    ]
    metric_tables = _detect_ruled_metric_pair_tables(source, excluded_bboxes)
    # 由独立年份表头证明的上下表优先于跨越它们的长候选，数据列仍逐项验证。
    replacements = year_tables + metric_tables + numeric_tables
    base, recovered, replacements = _prefer_complete_table_replacements(base, recovered, replacements)
    base.extend(replacements)
    existing_bounds = [candidate.core_bbox or candidate.bbox for candidate in [*base, *recovered]]
    candidates = _join_caption_supported_table_groups(
        source,
        [
            *base,
            *recovered,
            *_detect_short_caption_ruled_tables(source, excluded_bboxes),
            *_detect_captioned_blank_answer_tables(source, excluded_bboxes),
            *_detect_unruled_numeric_column_tables(source, [*excluded_bboxes, *existing_bounds]),
            *_detect_captioned_mixed_numeric_tables(source, [*excluded_bboxes, *existing_bounds]),
            *_detect_captioned_pair_text_tables(source, [*excluded_bboxes, *existing_bounds]),
            *_detect_page_clipped_blank_form_tables(source, [*excluded_bboxes, *existing_bounds]),
        ],
    )
    for candidate in candidates:
        _trim_unruled_table_prose_tail(source, candidate)
    candidates = [candidate for candidate in candidates if not any(
        _bbox_overlap_in_first(candidate.core_bbox or candidate.bbox, form) >= 0.9
        for form in source.form_bboxes
    )]
    _externalize_table_continuation_captions(
        source,
        candidates,
    )
    _bind_short_native_grid_captions(source, candidates)
    return sorted(
        candidates,
        key=lambda candidate: (candidate.bbox[1], candidate.bbox[0]),
    )


def _bind_short_native_grid_captions(source: _PageSource, candidates: list[_TableCandidate]) -> None:
    """异常字框页把紧贴网格的独立短中文题名绑定到表，拒绝同基线多列表头。"""
    from .table_filled_grid import has_inflated_native_cjk_metrics

    if not has_inflated_native_cjk_metrics(source.chars):
        return
    em = statistics.median(
        char["bbox"][2] - char["bbox"][0] for char in source.chars if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", ""))
    )
    boxes = {
        line.source_index: _bbox_union_many([char["tight_bbox"] for char in line.chars or [] if char.get("tight_bbox")])
        or line.bbox
        for line in source.lines
    }
    owned = {index for candidate in candidates for annotation in candidate.annotations for index in annotation.line_indices}
    for candidate in candidates:
        if candidate.angle:
            continue
        bounds = candidate.core_bbox or candidate.bbox
        for line in source.lines:
            box = boxes[line.source_index]
            if (
                line.source_index in owned
                or line.angle
                or not re.fullmatch(r"[\u4e00-\u9fff]{8,35}", line.text.strip())
                or not 0 < bounds[1] - box[3] <= 1.25 * em
                or abs(box[0] - bounds[0]) > em
                or box[2] - box[0] >= 0.6 * (bounds[2] - bounds[0])
            ):
                continue
            if any(
                other is not line
                and abs(_bbox_center_y(boxes[other.source_index]) - _bbox_center_y(box)) <= 0.5 * em
                and bounds[0] <= _bbox_center_x(boxes[other.source_index]) <= bounds[2]
                for other in source.lines
            ):
                continue
            candidate.annotations.append(_TableAnnotation("caption", box, {line.source_index}, {line.source_index: box}))
            candidate.line_indices.add(line.source_index)
            owned.add(line.source_index)


def _detect_aligned_numeric_tables(source: _PageSource, excluded: list[BBox]) -> list[_TableCandidate]:
    """重复数值列及二维表头补足稀疏续表，完整区域必须通过原生字符网格恢复。"""
    from ._table_recovery.aligned_numeric import (
        numeric_row_groups,
        numeric_region_bounds,
        regroup_numeric_text,
        build_aligned_numeric_candidate,
    )
    from ._table_recovery.contracts import NativeTableInput
    from ._table_recovery.text import build_native_table_text
    from ._table_recovery.sparse_common import _local_rules
    from ._table_recovery.engine import coerce_native_table_rules, coerce_native_table_rectangles

    if sum(bool(re.fullmatch(r"[+−-]?\(?\d+(?:[.,]\d+)*\)?|[-–—]", line.text.strip())) for line in source.lines) < 9:
        return []
    width, height = source.page_size
    table_input = NativeTableInput(
        (0.0, 0.0, width, height),
        source.page_size,
        0,
        tuple(source.chars),
        coerce_native_table_rules(source.drawing_lines),
        coerce_native_table_rectangles(source.path_infos),
    )
    text = build_native_table_text(table_input)
    if text is None:
        return []
    text = regroup_numeric_text(text)
    rules = _local_rules(table_input, width, height)
    output = []
    for rows, anchors in numeric_row_groups(text):
        if len(rows) < 3 or len(anchors) < 3:
            continue
        bounds = numeric_region_bounds(text, rows, anchors, rules)
        if bounds is None or any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded):
            continue
        regional_input = replace(table_input, table_bbox=bounds)
        regional_text = build_native_table_text(regional_input)
        if (
            regional_text is None
            or build_aligned_numeric_candidate(regional_input, regroup_numeric_text(regional_text)) is None
        ):
            continue
        members = {line.source_index for line in source.lines if _bbox_overlap_in_first(line.bbox, bounds) >= 0.8}
        output.append(
            _TableCandidate(bounds, bounds, 0, 120, core_bbox=bounds, line_indices=members, preserve_inline_font_styles=True)
        )
    return output


_PROSE_RULE_REFERENCE_FUNCTIONS = (_bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many)


def _prose_rule_native_box(value):
    """只接受内置有限框，异常或自定义数值不触发转换回调而直接回退。"""
    return (
        type(value) in (tuple, list)
        and len(value) == 4
        and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2**50 for v in value)
    )


def _prose_rule_native_flags(source, candidates):
    """文字特征按输入行计算一次；把完整成员集合和几何传给批量筛选。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None or any(
        actual is not expected
        for actual, expected in zip(
            (_bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many), _PROSE_RULE_REFERENCE_FUNCTIONS
        )
    ):
        return None
    grids = []
    drafts = []
    indices = []
    for i, candidate in enumerate(candidates):
        if isinstance(candidate, _TableCandidate) and candidate.angle == 0:
            box = candidate.core_bbox or candidate.bbox
            if type(candidate) is not _TableCandidate or not _prose_rule_native_box(box):
                return None
            grids.append(box)
        elif isinstance(candidate, _RuleCandidateDraft):
            if type(candidate) is not _RuleCandidateDraft:
                return None
            if candidate.context.angle or candidate.caption is not None:
                continue
            em = candidate.context.median_height
            if type(em) not in (int, float) or not math.isfinite(em) or abs(em) > 2**50:
                return None
            if em <= 0 or not candidate.boundaries:
                continue
            boxes = [rule.bbox for rule in candidate.boundaries]
            if not all(_prose_rule_native_box(box) for box in boxes):
                return None
            drafts.append((em, min(box[1] for box in boxes), max(box[3] for box in boxes), _bbox_union_many(boxes), []))
            indices.append(i)
    if not grids or not drafts:
        return set()
    # 只在全部必要几何条件成立后读取成员，避免为密集弱线组合复制整个表体。
    eligible = native.prose_rule_eligible(drafts, grids)
    active_drafts, active_indices = [], []
    for draft, index, accepted in zip(drafts, indices, eligible):
        if not accepted:
            continue
        candidate = candidates[index]
        members = [fragment.line_index for row in candidate.rows for fragment in row.fragments]
        if any(type(member) is not int or not -(2**63) <= member < 2**63 for member in members):
            return None
        active_drafts.append((*draft[:4], members))
        active_indices.append(index)
    drafts, indices = active_drafts, active_indices
    if not drafts:
        return set()
    rules = []
    for rule in source.drawing_lines:
        if (
            type(rule) is not _AxisLine
            or rule.orientation not in ("vertical", "horizontal")
            or not _prose_rule_native_box(rule.bbox)
        ):
            return None
        rules.append((rule.bbox, rule.orientation == "vertical"))
    lines = []
    for line in source.lines:
        if type(line) is not _LineItem or type(line.text) is not str or not _prose_rule_native_box(line.bbox):
            return None
        if type(line.source_index) is not int or not -(2**63) <= line.source_index < 2**63:
            return None
        weight = line.dominant_font_weight
        if weight is not None and (type(weight) not in (int, float) or not math.isfinite(weight)):
            return None
        eligible = (
            line.angle == 0
            and not line.caption_start
            and not line.formula_candidate_only
            and not line.semantic_type
            and (weight or 400) < 600
            and len(re.findall(r"\b[A-Za-z]{2,}\b", line.text)) >= 6
        )
        lines.append((line.bbox, line.source_index, bool(eligible)))
    flags = native.prose_rule_conflicts(drafts, grids, rules, lines)
    return None if flags is None else {index for index, conflict in zip(indices, flags) if conflict}


def _exclude_prose_spanning_rule_drafts(source, candidates):
    """批量筛选只移除原规则确认的冲突候选，特殊输入保留完整 Python 判断。"""
    conflicts = _prose_rule_native_flags(source, candidates)
    if conflicts is None:
        return _exclude_prose_spanning_rule_drafts_python(source, candidates)
    return [candidate for i, candidate in enumerate(candidates) if i not in conflicts]


def _exclude_prose_spanning_rule_drafts_python(source: _PageSource, candidates: list) -> list:
    """闭合列线和上下边界确认表顶时，跨越上方独立正文的弱横线区间不能扩大表体。"""
    grids = [candidate for candidate in candidates if isinstance(candidate, _TableCandidate) and candidate.angle == 0]
    if not grids:
        return candidates
    result = []
    for draft in candidates:
        if not isinstance(draft, _RuleCandidateDraft) or draft.context.angle or draft.caption is not None:
            result.append(draft)
            continue
        em = draft.context.median_height
        if em <= 0 or not draft.boundaries:
            result.append(draft)
            continue
        first = min(rule.bbox[1] for rule in draft.boundaries)
        last = max(rule.bbox[3] for rule in draft.boundaries)
        boundary_box = _bbox_union_many([rule.bbox for rule in draft.boundaries])
        members = {fragment.line_index for row in draft.rows for fragment in row.fragments}
        conflict = False
        for grid in grids:
            core = grid.core_bbox or grid.bbox
            width, height = core[2] - core[0], core[3] - core[1]
            if width < 8 * em or height < 4 * em or core[1] - first < 5 * em:
                continue
            if not core[1] - em <= last <= core[3] + em:
                continue
            if _bbox_axis_overlap_ratio(boundary_box, core, axis="x") < 0.9:
                continue
            verticals = [
                rule
                for rule in source.drawing_lines
                if rule.orientation == "vertical"
                and core[0] - 0.5 * em <= _bbox_center_x(rule.bbox) <= core[2] + 0.5 * em
                and rule.bbox[1] <= core[1] + 0.5 * em
                and rule.bbox[3] >= core[3] - 0.5 * em
            ]
            tracks = []
            for x in sorted(_bbox_center_x(rule.bbox) for rule in verticals):
                if not tracks or x - tracks[-1] > 0.3 * em:
                    tracks.append(x)
            horizontals = [
                rule
                for rule in source.drawing_lines
                if rule.orientation == "horizontal" and _bbox_axis_overlap_ratio(rule.bbox, core, axis="x") >= 0.9
            ]
            if len(tracks) < 3 or not all(
                any(abs(_bbox_center_y(rule.bbox) - y) <= 0.5 * em for rule in horizontals) for y in (core[1], core[3])
            ):
                continue
            prose = [
                line
                for line in source.lines
                if line.angle == 0
                and line.source_index in members
                and first + 0.5 * em < _bbox_center_y(line.bbox) < core[1] - em
                and line.bbox[2] - line.bbox[0] >= 0.65 * width
                and _bbox_axis_overlap_ratio(line.bbox, core, axis="x") >= 0.9
                and len(re.findall(r"\b[A-Za-z]{2,}\b", line.text)) >= 6
                and not line.caption_start
                and not line.formula_candidate_only
                and not line.semantic_type
                and (line.dominant_font_weight or 400) < 600
            ]
            rows = []
            for y in sorted(_bbox_center_y(line.bbox) for line in prose):
                if not rows or y - rows[-1] > 0.6 * em:
                    rows.append(y)
            if len(rows) >= 3:
                conflict = True
                break
        if not conflict:
            result.append(draft)
    return result


def _trim_unruled_table_prose_tail(source: _PageSource, candidate: _TableCandidate) -> None:
    """上下横线可能包住表后正文；重复数值列结束后，独立全宽自然段不能继续属于无列线表体。"""
    if candidate.angle:
        return
    core = candidate.core_bbox or candidate.local_bbox
    members = [
        line
        for line in source.lines
        if line.angle == 0
        and core[0] - 2 <= line.bbox[0]
        and line.bbox[2] <= core[2] + 2
        and core[1] <= _bbox_center_y(line.bbox) <= core[3]
    ]
    heights = [line.effective_height for line in members if line.effective_height > 0]
    if not heights:
        return
    em = statistics.median(heights)
    width = core[2] - core[0]
    if width < 8 * em:
        return
    rows = []
    for line in sorted(members, key=lambda item: (_bbox_center_y(item.bbox), item.bbox[0])):
        if (
            rows
            and abs(_bbox_center_y(line.bbox) - statistics.median(_bbox_center_y(item.bbox) for item in rows[-1])) <= 0.45 * em
        ):
            rows[-1].append(line)
        else:
            rows.append([line])
    numeric_rows = [
        row
        for row in rows
        if len(row) >= 2
        and max(line.bbox[0] for line in row) - min(line.bbox[0] for line in row) >= 0.3 * width
        and sum(bool(re.fullmatch(r"[+\-−$€£¥\d.,()%‰\s]+", line.text.strip())) for line in row) >= 2
    ]
    if len(numeric_rows) < 3:
        return
    last_data = max(line.bbox[3] for row in numeric_rows for line in row)
    data_em = statistics.median(line.effective_height for row in numeric_rows for line in row)
    prose = [
        line
        for line in members
        if line.bbox[1] - last_data >= 0.65 * em
        and line.bbox[2] - line.bbox[0] >= 0.8 * width
        and abs(line.bbox[0] - core[0]) <= 0.5 * em
        and len(re.findall(r"\b[A-Za-z]{2,}\b", line.text)) >= 6
        and line.effective_height >= 1.08 * data_em
    ]
    if len(prose) < 2:
        return
    first = min(prose, key=lambda line: line.bbox[1])
    from .table_annotations import _is_table_note_text

    if _is_table_note_text(first.text):
        return
    tail = sorted([line for line in members if line.bbox[1] >= first.bbox[1]], key=lambda line: line.bbox[1])
    if len(tail) < 3 or not any(b.bbox[1] - a.bbox[3] >= 0.6 * em for a, b in zip(tail, tail[1:])):
        return
    tail_bottom = max(line.bbox[3] for line in tail)
    if any(
        rule.orientation == "vertical"
        and core[0] - em <= _bbox_center_x(rule.bbox) <= core[2] + em
        and rule.bbox[1] <= first.bbox[3]
        and rule.bbox[3] >= tail_bottom - em
        for rule in source.drawing_lines
    ):
        return
    if any(
        rule.orientation == "horizontal"
        and first.bbox[1] <= rule.bbox[1] <= tail_bottom
        and _bbox_axis_overlap_ratio(rule.bbox, core, axis="x") >= 0.8
        for rule in source.drawing_lines
    ):
        return
    trimmed = (core[0], core[1], core[2], last_data + 0.25 * em)
    candidate.core_bbox = candidate.local_bbox = trimmed
    candidate.bbox = _bbox_union_many([trimmed, *(annotation.bbox for annotation in candidate.annotations)])
    candidate.line_indices.difference_update(line.source_index for line in tail)


def _bounded_table_caption_annotations(
    source: _PageSource, bounds: BBox, em: float, *, max_below_gap: float = 2.0
) -> list[_TableAnnotation]:
    """只从表体范围之外收集独立表题及同尺度续行，物理列线以内的表头不允许成为图注续行。"""
    seeds = [
        line
        for line in source.lines
        if line.angle == 0
        and re.match(r"^\s*(?:table|tab\.?|表格?)\s*(?:[:：]|(?:\d+(?:\.\d+)*|[IVX]+)[.:：\s])", line.text, re.I)
        and _bbox_axis_overlap_ratio(line.bbox, bounds, axis="x") >= 0.5
        and (0 <= bounds[1] - line.bbox[3] <= 4 * em or 0 <= line.bbox[1] - bounds[3] <= max_below_gap * em)
    ]
    if not seeds:
        # 已有物理列线确认表体时，贴近表顶、居中粗体短文字可作为无编号表题。
        seeds = [
            line
            for line in source.lines
            if line.angle == 0
            and (
                line.dominant_font_weight is not None
                and line.dominant_font_weight >= 600
                or line.font_signature is not None
                and "bold"
                in _font_styles_from_metadata(line.font_signature[0], line.font_signature[1], line.dominant_font_weight)
            )
            and 0 <= bounds[1] - line.bbox[3] <= 0.75 * em
            and 0.1 * (bounds[2] - bounds[0]) <= line.bbox[2] - line.bbox[0] <= 0.7 * (bounds[2] - bounds[0])
            and abs(_bbox_center_x(line.bbox) - _bbox_center_x(bounds)) <= 0.03 * (bounds[2] - bounds[0])
            and not re.fullmatch(r"\s*[\d.,:() -]+\s*", line.text)
        ]
        if not seeds:
            return []
    seed = min(seeds, key=lambda line: min(abs(line.bbox[3] - bounds[1]), abs(line.bbox[1] - bounds[3])))
    below = seed.bbox[1] >= bounds[3]
    members = [seed]
    for line in sorted(source.lines, key=lambda item: (item.bbox[1], item.bbox[0])):
        if line is seed or line.angle:
            continue
        if (
            abs(_bbox_center_y(line.bbox) - _bbox_center_y(seed.bbox)) <= 0.3 * em
            and 0 <= line.bbox[0] - max(item.bbox[2] for item in members) <= 4 * em
            and abs(line.effective_height - seed.effective_height) <= 0.2 * em
            and line.bbox[2] <= bounds[2] + em
        ):
            members.append(line)
            continue
        if line.bbox[1] < seed.bbox[1] + 0.5 * em:
            continue
        if not (line.bbox[1] >= bounds[3] if below else line.bbox[3] <= bounds[1]):
            continue
        if line.bbox[1] - max(item.bbox[3] for item in members) > 0.8 * em:
            break
        if abs(line.effective_height - seed.effective_height) > 0.2 * em:
            break
        if _bbox_axis_overlap_ratio(line.bbox, seed.bbox, axis="x") < 0.5:
            break
        members.append(line)
    boxes = {line.source_index: line.bbox for line in members}
    return [_TableAnnotation("caption", _bbox_union_many(list(boxes.values())), set(boxes), boxes)]


def _detect_page_clipped_blank_form_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """页面右界裁切表单须有重复横边、左框及两内轨；空白答案列和双行列名共同限定完整三列网格。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    width = source.page_size[0]
    horizontal = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and abs(rule.bbox[2] - width) <= 0.1 * em
            and rule.bbox[2] - rule.bbox[0] >= 8 * em
        ],
        key=lambda box: box[1],
    )
    if len(horizontal) < 4:
        return []
    first, last = horizontal[0], horizontal[-1]
    if not 6 * em <= last[3] - first[1] <= 20 * em or any(abs(box[0] - first[0]) > 0.25 * em for box in horizontal):
        return []
    bounds = (min(box[0] for box in horizontal), first[1], width, last[3])
    if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded_bboxes):
        return []
    vertical = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "vertical"
            and bounds[0] - 0.25 * em <= _bbox_center_x(rule.bbox) < width - em
            and min(rule.bbox[3], bounds[3]) - max(rule.bbox[1], bounds[1]) >= 0.65 * (bounds[3] - bounds[1])
        ],
        key=lambda box: box[0],
    )
    if len(vertical) != 3 or abs(_bbox_center_x(vertical[0]) - bounds[0]) > 0.25 * em:
        return []
    columns = [_bbox_center_x(box) for box in vertical] + [width]
    if any(right - left < 3 * em for left, right in zip(columns, columns[1:])):
        return []
    rows = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and bounds[1] <= _bbox_center_y(rule.bbox) <= bounds[3]
            and abs(rule.bbox[0] - bounds[0]) <= 0.25 * em
            and rule.bbox[2] >= columns[2] - 0.25 * em
        ],
        key=lambda box: box[1],
    )
    if not 6 <= len(rows) <= 12 or any(
        not 0.75 * em <= _bbox_center_y(b) - _bbox_center_y(a) <= 4 * em for a, b in zip(rows, rows[1:])
    ):
        return []
    members = [line for line in lines if _bbox_overlap_in_first(line.bbox, bounds) >= 0.9]
    header_bottom = _bbox_center_y(rows[1])
    header = [line for line in members if line.bbox[3] <= header_bottom]
    data = [line for line in members if line.bbox[1] >= header_bottom]
    if len(header) != 4 or not data or any(line.bbox[2] > columns[1] - 0.5 * em for line in data):
        return []
    left_header = sorted([line for line in header if _bbox_center_x(line.bbox) < columns[2]], key=lambda line: line.bbox[1])
    right_header = sorted([line for line in header if _bbox_center_x(line.bbox) >= columns[2]], key=lambda line: line.bbox[1])
    if len(left_header) != 2 or len(right_header) != 2:
        return []
    if (
        any(abs(a.bbox[1] - b.bbox[1]) > 0.2 * em for a, b in zip(left_header, right_header))
        or max(line.bbox[2] for line in left_header) + 0.25 * em > min(line.bbox[0] for line in right_header)
        or any(line.paragraph_terminal for line in header)
    ):
        return []
    if not all(
        any(top <= _bbox_center_y(line.bbox) <= bottom for line in data)
        for top, bottom in zip([_bbox_center_y(box) for box in rows[1:]], [_bbox_center_y(box) for box in rows[2:]])
    ):
        return []
    # 右侧表头略越过原内轨时，只在两列答案完全空白且列名有净空的前提下使用文字列界，避免切掉括号。
    columns[2] = min(columns[2], min(line.bbox[0] for line in right_header) - 0.1 * em)
    top, bottom = _bbox_center_y(rows[0]), _bbox_center_y(rows[-1])
    inferred = [_AxisLine((x - 0.05, top, x + 0.05, bottom), 0.1, "vertical") for x in columns]
    inferred.extend(
        _AxisLine((columns[0], _bbox_center_y(box) - 0.05, width, _bbox_center_y(box) + 0.05), 0.1, "horizontal")
        for box in rows
    )
    return [
        _TableCandidate(
            bbox=bounds,
            local_bbox=bounds,
            core_bbox=bounds,
            angle=0,
            score=120,
            line_indices={line.source_index for line in members},
            inferred_grid=inferred,
            inferred_grid_authoritative=True,
            preserve_inline_font_styles=True,
        )
    ]


_NUMERIC_ROW_PREPARATION_REFERENCES = (_build_fragments, _cluster_fragment_rows)


def _has_enough_numeric_fragments(lines, numeric, minimum):
    """三行数值表必须有足量独立数值片段；必要数量不足时不进行重复几何分组。"""
    if (_build_fragments, _cluster_fragment_rows) != _NUMERIC_ROW_PREPARATION_REFERENCES or any(
        type(line.text) is not str for line in lines
    ):
        return True
    count = 0
    for line in lines:
        if numeric.fullmatch(line.text.strip()):
            count += 1
            if count >= minimum:
                return True
    return False


def _detect_unruled_numeric_column_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """无框线表须有至少三行三列重复数值与独立粗体表头，列首原生字符对齐后才建立完整逻辑网格。"""
    lines = [
        line
        for line in source.lines
        if line.angle == 0
        and line.semantic_type is None
        and line.effective_height > 0
        and not line.formula_candidate_only
        and not line.compact_formula_cluster
        and not line.caption_start
    ]
    if not lines:
        return []
    numeric = re.compile(r"[+−-]?\d+(?:[.,]\d+)*(?:\s*(?:[%‰]|[A-Za-z]{1,8}))?")
    if not _has_enough_numeric_fragments(lines, numeric, 9):
        return []
    em = statistics.median(line.effective_height for line in lines)
    rows = _cluster_fragment_rows(_build_fragments(lines, source.page_size), em)
    by_source = {line.source_index: line for line in lines}
    data_rows = [
        row
        for row in rows
        if 3 <= len(row.fragments) <= 8 and all(numeric.fullmatch(fragment.text.strip()) for fragment in row.fragments)
    ]
    output = []
    consumed: set[int] = set()
    for first in data_rows:
        if any(fragment.line_index in consumed for fragment in first.fragments):
            continue
        columns = [fragment.local_bbox[0] for fragment in first.fragments]
        if any(right - left < 1.5 * em for left, right in zip(columns, columns[1:])):
            continue
        group = [first]
        for row in data_rows:
            if row.center_y <= first.center_y:
                continue
            if (
                row.bbox[1] - group[-1].bbox[3] > 1.5 * em
                or len(row.fragments) != len(columns)
                or any(abs(fragment.local_bbox[0] - x) > 0.3 * em for fragment, x in zip(row.fragments, columns))
            ):
                break
            group.append(row)
        if len(group) < 3:
            continue
        headers = [
            line
            for line in lines
            if line.font_signature is not None
            and line.font_coverage >= 0.75
            and "bold" in _font_styles_from_metadata(line.font_signature[0], line.font_signature[1], line.dominant_font_weight)
            and -0.15 * em <= first.bbox[1] - line.bbox[3] <= em
            and abs(line.bbox[0] - columns[0]) <= 0.3 * em
            and 0.8 * em <= line.effective_height <= 1.25 * em
            and line.bbox[2] >= group[0].bbox[2]
            and not line.paragraph_terminal
        ]
        if len(headers) != 1:
            continue
        header = headers[0]
        if not all(
            any(
                str(char.get("char", "")).isalpha() and char.get("bbox") is not None and abs(char["bbox"][0] - x) <= 0.25 * em
                for char in header.chars
            )
            for x in columns
        ):
            continue
        members = [header, *(by_source[fragment.line_index] for row in group for fragment in row.fragments)]
        bounds = (
            columns[0] - 0.25 * em,
            header.bbox[1] - 0.25 * em,
            max(line.bbox[2] for line in members) + 0.25 * em,
            group[-1].bbox[3] + 0.25 * em,
        )
        if any(
            _bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in [*excluded_bboxes, *(item.core_bbox for item in output)]
        ):
            continue
        edges = [bounds[0], *(x - 0.1 * em for x in columns[1:]), bounds[2]]
        row_edges = [
            bounds[1],
            (header.bbox[3] + first.bbox[1]) / 2,
            *((a.bbox[3] + b.bbox[1]) / 2 for a, b in zip(group, group[1:])),
            bounds[3],
        ]
        inferred = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in edges]
        inferred.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in row_edges)
        output.append(
            _TableCandidate(
                bbox=bounds,
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=12,
                line_indices={line.source_index for line in members},
                inferred_grid=inferred,
                preserve_inline_font_styles=True,
            )
        )
        consumed.update(line.source_index for line in members)
    return output


def _prefer_complete_table_replacements(base, recovered, replacements):
    """新候选须共同覆盖旧表的完整范围；只识别到前几行的局部表不得删除旧表或重复认领成员。"""
    rejected = set()
    for old in [*base, *recovered]:
        bounds = old.core_bbox or old.bbox
        overlapping = [
            index
            for index, item in enumerate(replacements)
            if _bbox_overlap_in_smaller(bounds, item.core_bbox or item.bbox) >= 0.9
        ]
        pieces = [
            (max(bounds[0], box[0]), max(bounds[1], box[1]), min(bounds[2], box[2]), min(bounds[3], box[3]))
            for index in overlapping
            for box in [replacements[index].core_bbox or replacements[index].bbox]
        ]
        if not pieces:
            continue
        # 按横坐标扫描交集的并面积，叠加候选不能靠重复面积虚增覆盖率。
        edges = sorted({coordinate for box in pieces for coordinate in (box[0], box[2])})
        area = 0.0
        for left, right in zip(edges, edges[1:]):
            intervals = sorted((box[1], box[3]) for box in pieces if box[0] <= left and right <= box[2])
            end = -float("inf")
            height = 0.0
            for top, bottom in intervals:
                height += max(0.0, bottom - max(top, end))
                end = max(end, bottom)
            area += (right - left) * height
        if area < 0.9 * (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]):
            rejected.update(overlapping)
    replacements = [item for index, item in enumerate(replacements) if index not in rejected]

    def keep_old(old):
        """仅在完整替代候选已通过范围检查时移除旧候选。"""
        return not any(
            _bbox_overlap_in_smaller(old.core_bbox or old.bbox, item.core_bbox or item.bbox) >= 0.9 for item in replacements
        )

    return [item for item in base if keep_old(item)], [item for item in recovered if keep_old(item)], replacements


_YEAR_GEOMETRY_REFERENCES = (
    _LineItem.__getattribute__,
    _AxisLine.__getattribute__,
    _LineItem.bbox,
    _AxisLine.bbox,
    _LineItem.text,
    _LineItem.paragraph_terminal,
    re.fullmatch,
    re.search,
)


def _native_year_header_groups(lines, rules, em):
    """逐输入状态读取所有普通框，完整几何批量匹配；不缓存表头、年份或最终成员。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is None
        or (
            _LineItem.__getattribute__,
            _AxisLine.__getattribute__,
            getattr(_LineItem, "bbox", None),
            getattr(_AxisLine, "bbox", None),
            getattr(_LineItem, "text", None),
            getattr(_LineItem, "paragraph_terminal", None),
            re.fullmatch,
            re.search,
        )
        != _YEAR_GEOMETRY_REFERENCES
    ):
        return None
    if any(
        type(line) is not _LineItem or type(line.text) is not str or type(line.paragraph_terminal) is not bool for line in lines
    ) or any(type(rule) is not _AxisLine for rule in rules):
        return None
    boxes, rule_boxes = [line.bbox for line in lines], [rule.bbox for rule in rules]
    if (
        type(em) not in (int, float)
        or abs(em) > 2**50
        or not math.isfinite(em)
        or not all(
            type(box) in (tuple, list)
            and len(box) == 4
            and all(
                type(v) is float and math.isfinite(v) and abs(v) <= 2**50 or type(v) is int and abs(v) <= 2**24 for v in box
            )
            for box in [*boxes, *rule_boxes]
        )
    ):
        return None
    return native.year_header_groups(boxes, rule_boxes, em)


def _detect_year_header_numeric_tables(source: _PageSource, excluded: list[BBox]) -> list[_TableCandidate]:
    """重复年份、独立首列表头及贴邻横线限定数值表，后一同栏年份表头形成永久上下表边界。"""
    lines = [
        line
        for line in source.lines
        if line.angle == 0
        and line.semantic_type in {None, "header", "footer"}
        and line.effective_height > 0
        and not line.formula_candidate_only
    ]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    rules = [
        rule for rule in source.drawing_lines if rule.orientation == "horizontal" and rule.bbox[2] - rule.bbox[0] >= 12 * em
    ]
    seeds = []
    header_groups = _native_year_header_groups(lines, rules, em)
    for rule_index, rule in enumerate(rules):
        header = sorted(
            [
                line
                for line in lines
                if rule.bbox[0] - 0.3 * em <= line.bbox[0]
                and line.bbox[2] <= rule.bbox[2] + 0.3 * em
                and 0 <= rule.bbox[1] - line.bbox[3] <= 0.8 * em
            ]
            if header_groups is None
            else [lines[index] for index in header_groups[rule_index]],
            key=lambda line: line.bbox[0],
        )
        if (
            not 4 <= len(header) <= 10
            or not all(re.fullmatch(r"\d{4}[A-Za-z]?", line.text.strip()) for line in header[1:])
            or not re.search(r"[A-Za-z\u3400-\u9fff]", header[0].text)
            or header[0].paragraph_terminal
            or len(header[0].text) > 30
            or any(a.bbox[2] + 0.3 * em >= b.bbox[0] for a, b in zip(header, header[1:]))
        ):
            continue
        years = [int(line.text.strip()[:4]) for line in header[1:]]
        if not all(1 <= after - before <= 10 for before, after in zip(years, years[1:])):
            # 四位金额也能贴邻底线；年份必须按小步长递增，不能将末行现金额当成下一表头。
            continue
        seeds.append((rule, header))
    output = []
    value = re.compile(r"[+−-]?\(?\d+(?:[.,]\d+)*\)?\s*[%‰]?")
    for rule, header in seeds:
        ceiling = min(
            (
                min(line.bbox[1] for line in other)
                for other_rule, other in seeds
                if other_rule.bbox[1] > rule.bbox[3] and _bbox_axis_overlap_ratio(rule.bbox, other_rule.bbox, axis="x") >= 0.9
            ),
            default=source.page_size[1],
        )
        inside = [
            line
            for line in lines
            if rule.bbox[3] <= line.bbox[1] < ceiling
            and rule.bbox[0] - 0.3 * em <= line.bbox[0]
            and line.bbox[2] <= rule.bbox[2] + 0.3 * em
        ]
        numeric = [
            line for line in inside if line.bbox[0] >= header[0].bbox[2] + 0.3 * em and value.fullmatch(line.text.strip())
        ]
        rows = _cluster_fragment_rows(_build_fragments(numeric, source.page_size), em)
        count = len(header) - 1
        group = []
        for row in rows:
            if len(row.fragments) != count or (group and row.bbox[1] - group[-1].bbox[3] > 2 * em):
                break
            if any(
                abs(fragment.local_bbox[2] - label.bbox[2]) > 0.4 * em for fragment, label in zip(row.fragments, header[1:])
            ):
                break
            group.append(row)
        if len(group) < 3 or group[0].bbox[1] - rule.bbox[3] > 1.5 * em:
            continue
        x_edges = [
            rule.bbox[0],
            (header[0].bbox[2] + min(fragment.local_bbox[0] for row in group for fragment in row.fragments[:1])) / 2,
            *(
                (
                    max(row.fragments[index].local_bbox[2] for row in group)
                    + min(row.fragments[index + 1].local_bbox[0] for row in group)
                )
                / 2
                for index in range(count - 1)
            ),
            rule.bbox[2],
        ]
        y_edges = [
            min(line.bbox[1] for line in header),
            _bbox_center_y(rule.bbox),
            *((a.center_y + b.center_y) / 2 for a, b in zip(group, group[1:])),
            group[-1].bbox[3] + 0.3 * em,
        ]
        bounds = (x_edges[0], y_edges[0], x_edges[-1], y_edges[-1])
        if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded):
            continue
        members = [
            line
            for line in inside
            if line.bbox[1] < bounds[3] and line.bbox[2] <= x_edges[1] and abs(line.bbox[0] - header[0].bbox[0]) <= 0.4 * em
        ]
        if len(members) < len(group) or any(not 0.7 * em <= line.effective_height <= 1.35 * em for line in members):
            continue
        claimed = {line.source_index for line in header + members}
        claimed.update(fragment.line_index for row in group for fragment in row.fragments)
        grid = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in x_edges]
        grid.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in y_edges)
        output.append(
            _TableCandidate(
                bbox=bounds,
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=14,
                line_indices=claimed,
                inferred_grid=grid,
                inferred_grid_authoritative=True,
                preserve_inline_font_styles=True,
            )
        )
    return output


def _detect_ruled_metric_pair_tables(source: _PageSource, excluded: list[BBox]) -> list[_TableCandidate]:
    """短题名右延横线及五行以上同栏指标值证明两列表，允许一项文字值，阻止空间分裂成为公式。"""
    lines = [
        line
        for line in source.lines
        if line.angle == 0
        and line.semantic_type is None
        and line.effective_height > 0
        and not line.formula_candidate_only
        and not line.caption_start
    ]
    output = []
    for heading in lines:
        if re.fullmatch(r"[\u3400-\u9fff]{4,12}", heading.text.strip()) is None:
            continue
        em = heading.effective_height / 1.5
        rules = [
            rule
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and abs(rule.bbox[0] - heading.bbox[2]) <= em
            and heading.bbox[1] <= rule.bbox[1] <= heading.bbox[3] + 0.5 * em
            and rule.bbox[2] - rule.bbox[0] >= 12 * em
        ]
        if len(rules) != 1:
            continue
        rule = rules[0]
        peers = [
            line
            for line in lines
            if line is not heading
            and heading.bbox[3] <= line.bbox[1] < heading.bbox[3] + 15 * em
            and heading.bbox[0] - 0.4 * em <= line.bbox[0]
            and line.bbox[2] <= rule.bbox[2] + 0.3 * em
            and 0.85 * em <= line.effective_height <= 1.15 * em
        ]
        rows = _cluster_fragment_rows(_build_fragments(peers, source.page_size), em)
        group = []
        for row in rows:
            if len(row.fragments) != 2 or (group and row.bbox[1] - group[-1].bbox[3] > em):
                break
            if (
                not re.search(r"[\u3400-\u9fff]", row.fragments[0].text)
                or len(re.sub(r"\s+", "", row.fragments[0].text)) > 16
                or row.fragments[1].local_bbox[0] - row.fragments[0].local_bbox[2] < 2 * em
            ):
                break
            group.append(row)
        if len(group) < 5 or not 0 <= group[0].bbox[1] - heading.bbox[3] <= 1.5 * em:
            continue
        value = re.compile(r"[+−-]?\d+(?:[.,]\d+)*(?:\s*[-/]\s*\d+(?:[.,]\d+)*)?(?:\s*[%‰A-Za-z\u3400-\u9fff]{0,6})?")
        if (
            sum(bool(value.fullmatch(row.fragments[1].text.strip())) for row in group) < 0.8 * len(group)
            or max(row.fragments[1].local_bbox[2] for row in group) - min(row.fragments[1].local_bbox[2] for row in group)
            > 0.4 * em
            or max(row.fragments[0].local_bbox[0] for row in group) - min(row.fragments[0].local_bbox[0] for row in group)
            > 0.4 * em
        ):
            continue
        left = min(row.bbox[0] for row in group) - 0.2 * em
        right = max(row.bbox[2] for row in group) + 0.2 * em
        split = (
            max(row.fragments[0].local_bbox[2] for row in group) + min(row.fragments[1].local_bbox[0] for row in group)
        ) / 2
        bounds = (left, group[0].bbox[1] - 0.2 * em, right, group[-1].bbox[3] + 0.2 * em)
        if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded):
            continue
        grid = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in (left, split, right)]
        grid.extend(
            _AxisLine((left, y - 0.05, right, y + 0.05), 0.1, "horizontal")
            for y in [bounds[1], *((a.center_y + b.center_y) / 2 for a, b in zip(group, group[1:])), bounds[3]]
        )
        annotation = _TableAnnotation("caption", heading.bbox, {heading.source_index}, {heading.source_index: heading.bbox})
        output.append(
            _TableCandidate(
                bbox=_bbox_union(bounds, heading.bbox),
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=14,
                line_indices={fragment.line_index for row in group for fragment in row.fragments},
                inferred_grid=grid,
                inferred_grid_authoritative=True,
                annotations=[annotation],
                preserve_inline_font_styles=True,
            )
        )
    return output


def _native_header_word_parts(line: _LineItem) -> list[_LineItem]:
    """只有整排都是年份标签时按原生字符切分定位，文字表头及原始来源身份保持不变。"""
    words = line.text.split()
    if len(words) < 3 or not all(re.fullmatch(r"\d{4}(?:[-–]\d{4}|[A-Za-z]\d?)?", word) for word in words) or not line.chars:
        return [line]
    parts = []
    start = 0
    raw = "".join(char.get("char", "") for char in line.chars)
    for word in words:
        at = raw.find(word, start)
        if at < 0:
            return [line]
        chars = line.chars[at : at + len(word)]
        start = at + len(word)
        bounds = _bbox_union_many([tuple(char["bbox"]) for char in chars if char.get("char", "").strip()])
        parts.append(replace(line, text=word, bbox=bounds, source_bbox=bounds, chars=chars))
    return parts


def _detect_captioned_mixed_numeric_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """独立表题、重复数值列及短文字首列限定无框线表，保留多层分组表头和紧邻星号注释。"""
    lines = [
        line
        for line in source.lines
        if line.angle == 0
        and line.semantic_type is None
        and line.effective_height > 0
        and not line.formula_candidate_only
        and not line.caption_start
    ]
    if not lines:
        return []
    numeric = re.compile(r"\*?[+−-]?\d+(?:[.,]\d+)*(?:\s*(?:[%‰]|[A-Za-z]{1,8}))?")
    if not _has_enough_numeric_fragments(lines, numeric, 6):
        return []
    # 最终明确要求源行的编号表题；全部水平源行都在当前映射内时，缺少表题也不会触发缺失编号异常。
    if all(
        type(line) is _LineItem and type(line.text) is str and type(line.angle) is int and type(line.source_index) is int
        for line in source.lines
    ):
        allowed = {line.source_index for line in lines}
        if not any(re.match(r"^\s*(?:table|tab\.?|表)\s*\d", line.text, re.I) for line in lines) and all(
            line.angle or line.source_index in allowed for line in source.lines
        ):
            return []
    em = statistics.median(line.effective_height for line in lines)
    rows = _cluster_fragment_rows(_build_fragments(lines, source.page_size), em)
    data = [
        row
        for row in rows
        if 3 <= len(row.fragments) <= 10
        and len(row.fragments[0].text.split()) <= 5
        and all(numeric.fullmatch(fragment.text.strip()) for fragment in row.fragments[1:])
    ]
    by_source = {line.source_index: line for line in lines}
    used = set()
    output = []
    continuation_headers = []
    for first in data:
        if any(fragment.line_index in used for fragment in first.fragments):
            continue
        group = [first]
        count = len(first.fragments)
        for row in data:
            if row.center_y <= first.center_y:
                continue
            if len(row.fragments) != count or not -0.15 * em <= row.bbox[1] - group[-1].bbox[3] <= 1.5 * em:
                break
            if any(
                min(abs(fragment.local_bbox[edge] - seed.local_bbox[edge]) for edge in (0, 2)) > 0.5 * em
                and abs(_bbox_center_x(fragment.local_bbox) - _bbox_center_x(seed.local_bbox)) > 0.5 * em
                for fragment, seed in zip(row.fragments, first.fragments)
            ):
                break
            group.append(row)
        if len(group) < 3:
            continue
        cols = [[row.fragments[index].local_bbox for row in group] for index in range(count)]
        if any(min(box[0] for box in right) - max(box[2] for box in left) < 0.4 * em for left, right in zip(cols, cols[1:])):
            continue
        anchors = [statistics.median(box[0] for box in column) for column in cols]
        headers = [
            line
            for line in lines
            if 0 <= first.bbox[1] - line.bbox[3] <= 4 * em
            and 0.8 <= line.effective_height / em <= 1.25
            and not line.paragraph_terminal
            and 1 <= len(line.text.split()) <= 8
            and not re.match(r"^\s*(?:table|tab\.?|表)\s*\d", line.text, re.I)
            and first.bbox[0] - em <= line.bbox[0]
            and line.bbox[2] <= first.bbox[2] + 8 * em
        ]
        if not headers:
            continue
        if (
            sum(
                bool(re.search(r"[A-Za-z\u3400-\u9fff]", line.text)) and not numeric.fullmatch(line.text.strip())
                for line in headers
            )
            < 2
        ):
            # 数据首行也可能是短文字加数字，至少两个非数值标签才能证明它是表头。
            continue
        header_parts = [part for line in headers for part in _native_header_word_parts(line)]
        nearest = [
            min(range(count), key=lambda index: abs(_bbox_center_x(part.bbox) - anchors[index])) for part in header_parts
        ]
        # 最下排单列表头或年份先确定空隙，上排居中分组字样不能人为挤窄数据列。
        bottom = max(line.bbox[3] for line in headers)
        leaf = [
            (part, index)
            for part, index in zip(header_parts, nearest)
            if bottom - part.bbox[3] <= 0.8 * em
            and part.bbox[2] - part.bbox[0]
            <= 1.8 * min((abs(x - anchors[index]) for x in anchors if x != anchors[index]), default=em)
        ]
        if len({index for _, index in leaf}) < count - 1:
            continue
        col_bounds = [list(column) for column in cols]
        for part, index in leaf:
            col_bounds[index].append(part.bbox)
        if any(max(box[2] for box in left) >= min(box[0] for box in right) for left, right in zip(col_bounds, col_bounds[1:])):
            continue
        edges = [
            min(box[0] for box in col_bounds[0]) - 0.25 * em,
            *(
                (max(box[2] for box in left) + min(box[0] for box in right)) / 2
                for left, right in zip(col_bounds, col_bounds[1:])
            ),
            max(box[2] for box in col_bounds[-1]) + 0.25 * em,
        ]
        bounds = (edges[0], min(line.bbox[1] for line in headers), edges[-1], group[-1].bbox[3] + 0.2 * em)
        if any(
            _bbox_overlap_in_smaller(bounds, box) >= 0.5
            for box in [*excluded_bboxes, *(candidate.core_bbox for candidate in output)]
        ):
            continue
        annotations = _bounded_table_caption_annotations(source, bounds, em, max_below_gap=2.5)
        if not annotations or not re.match(
            r"^\s*(?:table|tab\.?|表)\s*\d", by_source[min(annotations[0].line_indices)].text, re.I
        ):
            continue
        # 原生列线或重复底色单元格边缘优先于文字间隙的中点。
        for index in range(1, count):
            low = max(box[2] for box in col_bounds[index - 1])
            high = min(box[0] for box in col_bounds[index])
            native = [
                _bbox_center_x(rule.bbox)
                for rule in source.drawing_lines
                if rule.orientation == "vertical"
                and low < _bbox_center_x(rule.bbox) < high
                and rule.bbox[3] - rule.bbox[1] >= 0.7 * (bounds[3] - first.bbox[1])
            ]
            if len(native) == 1:
                edges[index] = native[0]
        header_bottom = (max(line.bbox[3] for line in headers) + first.bbox[1]) / 2
        y_edges = [bounds[1], header_bottom, *((a.bbox[3] + b.bbox[1]) / 2 for a, b in zip(group, group[1:])), bounds[3]]
        grid = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in (edges[0], edges[-1])]
        spans = []
        for upper in header_parts:
            below = [(part, index) for part, index in leaf if part.bbox[1] >= upper.bbox[3] - 0.1 * em and part is not upper]
            if len(below) < 2:
                continue
            matches = []
            for start in range(count - 1):
                for end in range(start + 1, count):
                    subset = [(part, index) for part, index in below if start <= index <= end]
                    indices = {index for _, index in subset}
                    if len(indices) != end - start + 1:
                        continue
                    leaf_box = _bbox_union_many([part.bbox for part, _ in subset])
                    if abs(_bbox_center_x(upper.bbox) - _bbox_center_x(leaf_box)) <= 1.1 * em and upper.bbox[2] - upper.bbox[
                        0
                    ] >= 0.2 * (leaf_box[2] - leaf_box[0]):
                        matches.append((start, end, (upper.bbox[3] + min(part.bbox[1] for part, _ in subset)) / 2))
            if matches:
                spans.append(max(matches, key=lambda item: item[1] - item[0]))
        if spans:
            start, end, split = min(spans, key=lambda item: item[2])
            for index, x in enumerate(edges[1:-1], 1):
                top = split if start < index <= end else bounds[1]
                grid.append(_AxisLine((x - 0.05, top, x + 0.05, bounds[3]), 0.1, "vertical"))
            grid.append(_AxisLine((edges[start], split - 0.05, edges[end + 1], split + 0.05), 0.1, "horizontal"))
            # 若原页已有整幅表头分隔线，保留其实际空格单元而非捏造跨行合并。
            grid.extend(
                rule
                for rule in source.drawing_lines
                if rule.orientation == "horizontal"
                and bounds[1] < rule.bbox[1] < header_bottom
                and rule.bbox[2] - rule.bbox[0] >= 0.9 * (bounds[2] - bounds[0])
                and _bbox_axis_overlap_ratio(rule.bbox, bounds, axis="x") >= 0.9
            )
        else:
            grid.extend(_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in edges[1:-1])
        grid.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in y_edges)
        body = [by_source[fragment.line_index] for row in group for fragment in row.fragments]
        notes = _near_native_table_note(lines, bounds, em)
        if notes:
            annotations.append(notes)
        members = {line.source_index for line in headers + body}
        candidate = _TableCandidate(
            bbox=_bbox_union_many([bounds, *(annotation.bbox for annotation in annotations)]),
            local_bbox=bounds,
            core_bbox=bounds,
            angle=0,
            score=12,
            line_indices=members,
            inferred_grid=grid,
            inferred_grid_authoritative=True,
            annotations=annotations,
            preserve_inline_font_styles=True,
        )
        output.append(candidate)
        used.update(members)
        if not spans and len(headers) == count:
            continuation_headers.append([line.text for line in sorted(headers, key=lambda line: line.bbox[0])])
    output.extend(_repeated_single_row_numeric_tables(source, lines, rows, continuation_headers, used, em, excluded_bboxes))
    return output


def _near_native_table_note(lines, bounds, em):
    """星号说明或明确来源标记及同式贴邻续行属于前表脚注，下方不同字体正文和独立表头分开。"""
    seeds = [
        line
        for line in lines
        if re.match(r"^(?:\*[A-Za-z]|(?:Source|Notes?|来源|注)\s*[:：])", line.text.strip(), re.I)
        and (len(line.text.split()) >= 4 or len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 4)
        and 0 <= line.bbox[1] - bounds[3] <= 2 * em
        and _bbox_axis_overlap_ratio(line.bbox, bounds, axis="x") >= 0.5
    ]
    if len(seeds) != 1:
        return None
    members = [seeds[0]]
    for line in sorted(lines, key=lambda line: line.bbox[1]):
        if line is seeds[0] or line.bbox[1] <= seeds[0].bbox[1]:
            continue
        if (
            line.font_signature != seeds[0].font_signature
            or not -0.25 * em <= line.bbox[1] - members[-1].bbox[3] <= 0.75 * em
            or abs(line.bbox[0] - seeds[0].bbox[0]) > 0.5 * em
        ):
            break
        members.append(line)
    boxes = {line.source_index: line.bbox for line in members}
    return _TableAnnotation("footnote", _bbox_union_many(list(boxes.values())), set(boxes), boxes)


def _repeated_single_row_numeric_tables(source, lines, rows, known_headers, used, em, excluded):
    """完整重复的原生表头与同列单行数字证明续表，不能从孤立一行任意推断表格。"""
    output = []
    for header in lines:
        if header.source_index in used or not header.chars:
            continue
        chars = [char for char in header.chars if char.get("char", "").strip()]
        raw = "".join(char["char"] for char in chars)
        matches = [labels for labels in known_headers if raw == re.sub(r"\s+", "", "".join(labels))]
        if len(matches) != 1:
            continue
        labels = matches[0]
        parts = []
        at = 0
        for label in labels:
            length = len(re.sub(r"\s+", "", label))
            part = chars[at : at + length]
            at += length
            parts.append(_bbox_union_many([tuple(char["bbox"]) for char in part]))
        following = [
            row
            for row in rows
            if len(row.fragments) == len(parts)
            and 0 <= row.bbox[1] - header.bbox[3] <= 0.75 * em
            and all(re.fullmatch(r"\d+(?:[.,]\d+)*(?:\s*[A-Za-z]{1,8})?", fragment.text.strip()) for fragment in row.fragments)
        ]
        if len(following) != 1:
            continue
        row = following[0]
        columns = [[part, fragment.local_bbox] for part, fragment in zip(parts, row.fragments)]
        if any(max(box[2] for box in left) >= min(box[0] for box in right) for left, right in zip(columns, columns[1:])):
            continue
        edges = [
            min(box[0] for box in columns[0]) - 0.2 * em,
            *((max(box[2] for box in left) + min(box[0] for box in right)) / 2 for left, right in zip(columns, columns[1:])),
            max(box[2] for box in columns[-1]) + 0.2 * em,
        ]
        bounds = (edges[0], header.bbox[1] - 0.2 * em, edges[-1], row.bbox[3] + 0.2 * em)
        if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded):
            continue
        grid = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in edges]
        grid.extend(
            _AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal")
            for y in [bounds[1], (header.bbox[3] + row.bbox[1]) / 2, bounds[3]]
        )
        output.append(
            _TableCandidate(
                bbox=bounds,
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=12,
                line_indices={header.source_index, *(fragment.line_index for fragment in row.fragments)},
                inferred_grid=grid,
                inferred_grid_authoritative=True,
                preserve_inline_font_styles=True,
            )
        )
    return output


def _detect_captioned_pair_text_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """整行底色表头、上下横线和独立表题确认双列文字表；重复列首字符验证被合并的原生行。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    rules = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal" and rule.bbox[2] - rule.bbox[0] >= 8 * em
        ],
        key=lambda box: box[1],
    )
    output = []
    for top, bottom in zip(rules, rules[1:]):
        bounds = (top[0], top[1], top[2], bottom[3])
        if (
            abs(top[0] - bottom[0]) > 0.25 * em
            or abs(top[2] - bottom[2]) > 0.25 * em
            or not 4 * em <= bottom[1] - top[3] <= 15 * em
            or any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded_bboxes)
        ):
            continue
        fills = [
            path.bbox
            for path in source.path_infos
            if path.fill_visible
            and path.segment_count == 5
            and path.fill_rgba
            and max(path.fill_rgba[:3]) <= 120
            and abs(path.bbox[0] - top[0]) <= 0.25 * em
            and abs(path.bbox[2] - top[2]) <= 0.25 * em
            and abs(path.bbox[1] - top[1]) <= 0.25 * em
            and 0.75 * em <= path.bbox[3] - path.bbox[1] <= 2 * em
        ]
        annotations = _bounded_table_caption_annotations(source, bounds, em)
        if len(fills) != 1 or len(annotations) != 1 or annotations[0].bbox[1] < bottom[3]:
            continue
        fill = fills[0]
        members = [line for line in lines if _bbox_overlap_in_first(line.bbox, bounds) >= 0.9]
        header = [line for line in members if _bbox_overlap_in_first(line.bbox, fill) >= 0.9]
        if len(header) != 1:
            continue
        rows = _cluster_fragment_rows(_build_fragments([line for line in members if line not in header], source.page_size), em)
        paired = [row for row in rows if len(row.fragments) == 2]
        if not 4 <= len(rows) <= 10 or len(paired) < 2:
            continue
        left = statistics.median(row.fragments[0].local_bbox[0] for row in paired)
        right = statistics.median(row.fragments[1].local_bbox[0] for row in paired)
        if right - left < 3 * em or any(
            abs(row.fragments[0].local_bbox[0] - left) > 0.25 * em or abs(row.fragments[1].local_bbox[0] - right) > 0.25 * em
            for row in paired
        ):
            continue
        data = [line for line in members if line not in header]
        glyphs = [
            char for line in data for char in line.chars if str(char.get("char", "")).strip() and char.get("bbox") is not None
        ]
        if not all(
            any(
                abs(char["bbox"][0] - right) <= 0.1 * em
                and row.bbox[1] - 0.3 * em <= _bbox_center_y(char["bbox"]) <= row.bbox[3] + 0.3 * em
                for char in glyphs
            )
            for row in rows
        ):
            continue
        left_glyphs = [char for char in glyphs if char["bbox"][0] < right - 0.2 * em]
        if not left_glyphs:
            continue
        left_end = max(char["bbox"][2] for char in left_glyphs)
        if not 0 <= right - left_end <= 3 * em:
            continue
        split = (left_end + right) / 2
        edges = [bounds[1], fill[3], *((a.bbox[3] + b.bbox[1]) / 2 for a, b in zip(rows, rows[1:])), bounds[3]]
        inferred = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in [bounds[0], bounds[2]]]
        inferred.append(_AxisLine((split - 0.05, fill[3], split + 0.05, bounds[3]), 0.1, "vertical"))
        inferred.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in edges)
        output.append(
            _TableCandidate(
                bbox=_bbox_union_many([bounds, annotations[0].bbox]),
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=12,
                line_indices={line.source_index for line in members},
                annotations=annotations,
                inferred_grid=inferred,
                preserve_inline_font_styles=True,
            )
        )
    return output


def _blank_answer_header_split(header: _LineItem) -> float | None:
    """单一原生表头行以唯一显著词间净空限定两列，正常单词空格不能生成空白答题列。"""
    chars = sorted(
        [char for char in header.chars if str(char.get("char", "")).strip() and char.get("bbox") is not None],
        key=lambda char: char["bbox"][0],
    )
    gaps = [
        (right["bbox"][0] - left["bbox"][2], (right["bbox"][0] + left["bbox"][2]) / 2)
        for left, right in zip(chars, chars[1:])
        if right["bbox"][0] - left["bbox"][2] > 0.1 * header.effective_height
    ]
    if len(gaps) < 2:
        return None
    typical = statistics.median(gap for gap, _ in gaps)
    candidates = [x for gap, x in gaps if gap >= max(0.6 * header.effective_height, 2.5 * typical)]
    return candidates[0] if len(candidates) == 1 else None


def _detect_captioned_blank_answer_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """独立表题、两条同宽横线、双列表头及五个短行名共同确认空白答题表，原生字符和空格均保留。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    horizontal = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal" and rule.bbox[2] - rule.bbox[0] >= 6 * em
        ],
        key=lambda box: box[1],
    )
    output = []
    for top, bottom in zip(horizontal, horizontal[1:]):
        bounds = (top[0], top[1], top[2], bottom[3])
        if (
            abs(top[0] - bottom[0]) > 0.25 * em
            or abs(top[2] - bottom[2]) > 0.25 * em
            or not 5 * em <= bottom[1] - top[3] <= 20 * em
            or any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded_bboxes)
            or any(
                rule.orientation == "vertical"
                and _bbox_axis_overlap_ratio(bounds, rule.bbox, axis="y") >= 0.5
                and bounds[0] <= _bbox_center_x(rule.bbox) <= bounds[2]
                for rule in source.drawing_lines
            )
        ):
            continue
        annotations = _bounded_table_caption_annotations(source, bounds, em)
        if len(annotations) != 1 or annotations[0].bbox[3] > top[1]:
            continue
        members = sorted(
            [line for line in lines if _bbox_overlap_in_first(line.bbox, bounds) >= 0.9], key=lambda line: line.bbox[1]
        )
        if not 6 <= len(members) <= 13:
            continue
        header, keys = members[0], members[1:]
        if (
            header.font_signature is None
            or "bold"
            not in _font_styles_from_metadata(header.font_signature[0], header.font_signature[1], header.dominant_font_weight)
            or header.bbox[1] - top[3] > em
            or bottom[1] - keys[-1].bbox[3] > em
        ):
            continue
        split = _blank_answer_header_split(header)
        if (
            split is None
            or split - bounds[0] < 3 * em
            or bounds[2] - split < 4 * em
            or any(
                len(line.text.split()) > 3
                or line.paragraph_terminal
                or abs(line.bbox[0] - header.bbox[0]) > 0.35 * em
                or line.bbox[2] > split - 0.35 * em
                or not 0.8 * header.effective_height <= line.effective_height <= 1.2 * header.effective_height
                for line in keys
            )
            or any(not 0.2 * em <= after.bbox[1] - before.bbox[3] <= 1.5 * em for before, after in zip(members, members[1:]))
        ):
            continue
        edges = [bounds[1], *((before.bbox[3] + after.bbox[1]) / 2 for before, after in zip(members, members[1:])), bounds[3]]
        inferred = [
            _AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in [bounds[0], split, bounds[2]]
        ]
        inferred.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in edges)
        output.append(
            _TableCandidate(
                bbox=_bbox_union_many([bounds, annotations[0].bbox]),
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=12,
                line_indices={line.source_index for line in members},
                annotations=annotations,
                inferred_grid=inferred,
            )
        )
    return output


def _open_table_header_members(
    lines: list[_LineItem],
    bounds: BBox,
    tracks: list[BBox],
    em: float,
) -> list[_LineItem]:
    """表顶横线上方同排、各占独立列的短表头可位于列线之外，但不能越过独立表题或正文。"""
    edges = [bounds[0], *sorted(_bbox_center_x(box) for box in tracks), bounds[2]]
    candidates = [
        line
        for line in lines
        if not line.caption_start
        and 0 <= bounds[1] - line.bbox[3] <= 1.5 * em
        and 0.7 * em <= line.effective_height <= 1.35 * em
        and bounds[0] <= line.bbox[0]
        and line.bbox[2] <= bounds[2]
        and 1 <= len(line.text.split()) <= 5
        and not line.paragraph_terminal
        and re.match(r"^\s*(?:table|figure|表|图)\s*[\dIVX]", line.text, re.I) is None
        and any(left <= line.bbox[0] and line.bbox[2] <= right for left, right in zip(edges, edges[1:]))
    ]
    if (
        len(candidates) < 2
        or max(_bbox_center_y(line.bbox) for line in candidates) - min(_bbox_center_y(line.bbox) for line in candidates)
        > 0.3 * em
    ):
        return []
    columns = [
        next(
            index for index, (left, right) in enumerate(zip(edges, edges[1:])) if left <= line.bbox[0] and line.bbox[2] <= right
        )
        for line in candidates
    ]
    return candidates if len(columns) == len(set(columns)) else []


def _detect_open_vertical_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """用长内列线及横向分隔线闭合开边表格，要求重复多列数据，完整保留线端内的表头和末行。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    vertical = [
        rule.bbox for rule in source.drawing_lines if rule.orientation == "vertical" and rule.bbox[3] - rule.bbox[1] >= 5 * em
    ]
    horizontal = sorted(
        [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal" and rule.bbox[2] - rule.bbox[0] >= 6 * em
        ],
        key=lambda box: box[2] - box[0],
        reverse=True,
    )
    output = []
    for rule in horizontal:
        tracks = [
            box
            for box in vertical
            if rule[0] + em < _bbox_center_x(box) < rule[2] - em
            and box[1] - 0.3 * em <= _bbox_center_y(rule) <= box[3] + 0.3 * em
        ]
        if len(tracks) < 2:
            continue
        top, bottom = min(box[1] for box in tracks), max(box[3] for box in tracks)
        end_spread = max(box[3] for box in tracks) - min(box[3] for box in tracks)
        full_rules = [
            box
            for box in horizontal
            if top <= _bbox_center_y(box) <= bottom and abs(box[0] - rule[0]) <= 0.3 * em and abs(box[2] - rule[2]) <= 0.3 * em
        ]
        if max(box[1] for box in tracks) - top > 3 * em or end_spread > 1.5 * em or end_spread > em and len(full_rules) < 3:
            continue
        bounds = (rule[0], top, rule[2], bottom)
        if all(
            any(
                abs(_bbox_center_x(box) - edge) <= 0.3 * em and box[1] <= top + 0.3 * em and box[3] >= bottom - 0.3 * em
                for box in vertical
            )
            for edge in [bounds[0], bounds[2]]
        ):
            # 已有外竖框的闭合网格由原规则负责，避免为完整表重建近邻外线。
            continue
        if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded_bboxes):
            continue
        if any(_bbox_overlap_in_first(bounds, candidate.core_bbox) >= 0.9 for candidate in output):
            continue
        members = [
            line
            for line in lines
            if top - 0.25 * em <= _bbox_center_y(line.bbox) <= bottom + 0.25 * em
            and rule[0] - 0.25 * em <= line.bbox[0]
            and line.bbox[2] <= rule[2] + 0.25 * em
        ]
        header_members = _open_table_header_members(lines, bounds, tracks, em)
        members.extend(header_members)
        rows = []
        for line in sorted(members, key=lambda item: (_bbox_center_y(item.bbox), item.bbox[0])):
            if (
                rows
                and abs(_bbox_center_y(line.bbox) - statistics.median(_bbox_center_y(item.bbox) for item in rows[-1]))
                <= 0.5 * em
            ):
                rows[-1].append(line)
            else:
                rows.append([line])
        data = [row for row in rows if min(line.bbox[1] for line in row) >= rule[3] - 0.25 * em]
        # 短表有三条内列线、完整表头及至少两条同宽行线时，重复多列数据可只有两行。
        short_grid = (
            len(tracks) >= 3
            and len(full_rules) >= 2
            and any(len(row) >= 3 and max(line.bbox[3] for line in row) < rule[1] for row in rows)
        )
        required_rows = 2 if short_grid else 4
        if len(data) < required_rows or sum(len(row) >= 2 for row in data) < required_rows:
            continue
        if any(
            line.bbox[2] - line.bbox[0] > 0.75 * (rule[2] - rule[0]) and len(line.text.split()) > 6
            for row in data
            for line in row
        ):
            continue
        # 表内的编号与字符必须在各列线之间；横跨多数列的自然句子不构成开边表格。
        long_tracks = [box for box in tracks if box[1] <= top + 0.5 * em]
        if any(
            line.bbox[0] + 0.3 * em < _bbox_center_x(box) < line.bbox[2] - 0.3 * em
            for row in data
            for line in row
            for box in long_tracks
        ):
            continue
        bounds = (
            bounds[0],
            min(top, *(line.bbox[1] for line in members)),
            bounds[2],
            max(bottom, *(line.bbox[3] for line in members)),
        )
        annotations = _bounded_table_caption_annotations(source, bounds, em)
        inferred = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in [bounds[0], bounds[2]]]
        if header_members:
            # 表头各自占据已确认列，但印刷列线从表顶横线才开始；仅向上补齐表头列，保留表内真实跨列单元格。
            inferred.extend(
                _AxisLine(
                    (_bbox_center_x(box) - 0.05, bounds[1], _bbox_center_x(box) + 0.05, max(top, box[1])), 0.1, "vertical"
                )
                for box in tracks
            )
        edges = {bounds[1], bounds[3]}
        # 表内浅色整行矩形的上下边是原生行边界，可补足只有表头横线的开边表，不依赖行中数字。
        for path in source.path_infos:
            if (
                path.fill_visible
                and path.segment_count == 5
                and path.fill_rgba
                and min(path.fill_rgba[:3]) >= 180
                and abs(path.bbox[0] - bounds[0]) <= em
                and abs(path.bbox[2] - bounds[2]) <= em
                and bounds[1] <= path.bbox[1] < path.bbox[3] <= bounds[3] + 0.2 * em
            ):
                edges.update((path.bbox[1], min(bounds[3], path.bbox[3])))
        keys = sorted(
            [
                line
                for line in members
                if _bbox_center_x(line.bbox) < min(_bbox_center_x(box) for box in tracks)
                and line.bbox[1] >= rule[3] - 0.25 * em
                and re.fullmatch(r"\d{1,3}", line.text.strip())
            ],
            key=lambda line: line.bbox[1],
        )
        if len(keys) >= 4 and all(int(b.text) == int(a.text) + 1 for a, b in zip(keys, keys[1:])):
            # 稳定编号列提供逻辑行身份，多行描述不会把相邻数据行粘成同一单元格。
            edges.update((_bbox_center_y(a.bbox) + _bbox_center_y(b.bbox)) / 2 for a, b in zip(keys, keys[1:]))
        for y in sorted(edges):
            if not any(
                abs(_bbox_center_y(box) - y) <= 0.25 * em and abs(box[0] - bounds[0]) <= em and abs(box[2] - bounds[2]) <= em
                for box in horizontal
            ):
                inferred.append(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal"))
        output.append(
            _TableCandidate(
                bbox=_bbox_union_many([bounds, *(a.bbox for a in annotations)]),
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=12,
                line_indices={line.source_index for line in members},
                annotations=annotations,
                inferred_grid=inferred,
            )
        )
    return output


def _detect_short_caption_ruled_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """独立表题与三条同宽横线确认单列多行或多列两行表；表格不要求必须有大量数字。"""
    lines = [line for line in source.lines if line.angle == 0 and line.effective_height > 0]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines)
    captions = [line for line, _ in _table_caption_candidates(lines, source.page_size, 0, em)]
    rules = [
        rule.bbox for rule in source.drawing_lines if rule.orientation == "horizontal" and rule.bbox[2] - rule.bbox[0] >= 6 * em
    ]
    output = []
    for caption in captions:
        ends = [
            box
            for box in rules
            if 0 <= caption.bbox[1] - box[3] <= 2 * em and _bbox_axis_overlap_ratio(box, caption.bbox, axis="x") >= 0.8
        ]
        if not ends:
            continue
        end = max(ends, key=lambda box: box[3])
        floor = max(
            (
                other.bbox[3]
                for other in captions
                if other.bbox[3] < end[1] and _bbox_axis_overlap_ratio(other.bbox, end, axis="x") >= 0.8
            ),
            default=0,
        )
        aligned = sorted(
            [
                box
                for box in rules
                if floor < box[1] <= end[1] and abs(box[0] - end[0]) <= 0.3 * em and abs(box[2] - end[2]) <= 0.3 * em
            ],
            key=lambda box: box[1],
        )
        if len(aligned) != 3:
            continue
        bounds = (end[0], aligned[0][1], end[2], end[3])
        if any(_bbox_overlap_in_smaller(bounds, box) >= 0.5 for box in excluded_bboxes):
            continue
        members = [line for line in lines if _bbox_overlap_in_first(line.bbox, bounds) >= 0.9]
        rows = []
        for line in sorted(members, key=lambda item: (_bbox_center_y(item.bbox), item.bbox[0])):
            if (
                rows
                and abs(_bbox_center_y(line.bbox) - statistics.median(_bbox_center_y(item.bbox) for item in rows[-1]))
                <= 0.5 * em
            ):
                rows[-1].append(line)
            else:
                rows.append([line])
        single_column = len(rows) >= 5 and all(len(row) == 1 and len(row[0].text.split()) <= 5 for row in rows)
        paired = (
            len(rows) == 2
            and len(rows[1]) >= 3
            and (len(rows[0]) == len(rows[1]) or len(rows[0]) == 1 and len(rows[0][0].text.split()) == len(rows[1]))
        )
        if not single_column and not paired:
            continue
        if paired and sum(bool(re.search(r"\d", line.text)) for line in rows[1]) < 2:
            continue
        inferred = []
        if single_column:
            edges = [
                bounds[1],
                *((max(a[0].bbox[3], a[0].bbox[1]) + b[0].bbox[1]) / 2 for a, b in zip(rows, rows[1:])),
                bounds[3],
            ]
            # 已有横线优先于文字中点，避免在真实表头线旁新增窄空行。
            edges = [min(aligned, key=lambda box: abs(_bbox_center_y(box) - y)) for y in edges]
            midpoints = [bounds[1], *((a[0].bbox[3] + b[0].bbox[1]) / 2 for a, b in zip(rows, rows[1:])), bounds[3]]
            edges = [_bbox_center_y(box) if abs(_bbox_center_y(box) - y) <= 0.5 * em else y for box, y in zip(edges, midpoints)]
            inferred = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in [bounds[0], bounds[2]]]
            inferred.extend(_AxisLine((bounds[0], y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal") for y in edges)
        annotations = _bounded_table_caption_annotations(source, bounds, em)
        output.append(
            _TableCandidate(
                bbox=_bbox_union_many([bounds, *(a.bbox for a in annotations)]),
                local_bbox=bounds,
                core_bbox=bounds,
                angle=0,
                score=9,
                line_indices={line.source_index for line in members},
                annotations=annotations,
                inferred_grid=inferred,
            )
        )
    return output


_OVERLAP_MEMBER_REFERENCES = (_bbox_overlap_in_first, _overlap_geometry._bbox_area)


def _native_overlap_member_groups(boxes, regions, threshold):
    """完整框匹配仅接受普通数值，较大整数、自定义框及替换的几何规则完整回退。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None or (_bbox_overlap_in_first, _overlap_geometry._bbox_area) != _OVERLAP_MEMBER_REFERENCES:
        return None
    if type(boxes) not in (tuple, list) or type(regions) not in (tuple, list):
        return None
    if not all(
        type(box) in (tuple, list)
        and len(box) == 4
        and all(
            type(value) is float and math.isfinite(value) and abs(value) <= 2**50 or type(value) is int and abs(value) <= 2**24
            for value in box
        )
        for box in [*boxes, *regions]
    ):
        return None
    return native.overlap_member_groups(boxes, regions, threshold)


def _detect_shaded_header_tables(source: _PageSource, excluded_bboxes: list[BBox]) -> list[_TableCandidate]:
    """浅色整行表头与稳定共享列共同恢复无外框表，允许数值空单元格和跨行文字。"""
    lines = [line for line in source.lines if line.angle == 0 and line.semantic_type in {None, "header", "footer"}]
    if not lines:
        return []
    em = statistics.median(line.effective_height for line in lines if line.effective_height > 0)
    header_cells = [
        path.bbox
        for path in source.path_infos
        if path.fill_visible
        and path.segment_count == 5
        and path.fill_rgba
        and min(path.fill_rgba[:3]) >= 180
        and 0.5 * em <= path.bbox[3] - path.bbox[1] <= 2.5 * em
        and not any(_bbox_overlap_in_smaller(path.bbox, box) >= 0.5 for box in excluded_bboxes)
    ]
    headers = []
    for cell in sorted(header_cells, key=lambda box: (box[1], box[0])):
        if (
            headers
            and abs(cell[1] - headers[-1][1]) <= 0.2 * em
            and abs(cell[3] - headers[-1][3]) <= 0.2 * em
            and 0 <= cell[0] - headers[-1][2] <= 0.2 * em
        ):
            headers[-1] = _bbox_union(headers[-1], cell)
        else:
            headers.append(cell)
    # 表格截图可仅提供底色与线条；原生文字层的多列表头和重复栏缘仍能证实其结构。
    boxes = (
        [line.bbox for line in lines]
        if (
            (
                _native_clip_stage_grid,
                _bbox_axis_overlap_ratio,
                _bbox_center_x,
                _bbox_center_y,
                _bbox_union,
                _bbox_union_many,
                _LineItem.__getattribute__,
            )
            == _SHADED_GROUP_STATE_REFERENCES
            and all(type(line) is _LineItem for line in lines)
        )
        else None
    )
    image_groups = _native_overlap_member_groups(boxes, source.image_bboxes, 0.9)
    for image_index, image in enumerate(source.image_bboxes):
        inside = (
            [lines[i] for i in image_groups[image_index]]
            if image_groups is not None
            else [line for line in lines if _bbox_overlap_in_first(line.bbox, image) >= 0.9]
        )
        if len(inside) >= 20:
            top = min(line.bbox[1] for line in inside)
            headers.append((image[0], top - 0.2 * em, image[2], top + 1.5 * em))
    output = []
    ordered_headers = sorted(headers, key=lambda box: box[1])
    groups = _native_overlap_member_groups(boxes, ordered_headers, 0.8)
    cell_groups = _native_overlap_member_groups(header_cells, ordered_headers, 0.9) if boxes is not None else None
    for header_index, header in enumerate(ordered_headers):
        labels = (
            [lines[i] for i in groups[header_index]]
            if groups is not None
            else [line for line in lines if _bbox_overlap_in_first(line.bbox, header) >= 0.8]
        )
        labels.sort(key=lambda line: line.bbox[0])
        if not 3 <= len(labels) <= 8 or any(a.bbox[2] + em > b.bbox[0] for a, b in zip(labels, labels[1:])):
            continue
        columns = [line.bbox[0] for line in labels]
        cells = sorted(
            [header_cells[i] for i in cell_groups[header_index]]
            if cell_groups is not None
            else [cell for cell in header_cells if _bbox_overlap_in_first(cell, header) >= 0.9],
            key=lambda box: box[0],
        )
        raster_support = any(
            _bbox_overlap_in_first(header, image) >= 0.9 and image[3] - header[3] >= 8 * em for image in source.image_bboxes
        )
        floor = min(
            (
                other[1]
                for other in headers
                if other[1] > header[3] + em and _bbox_axis_overlap_ratio(header, other, axis="x") >= 0.8
            ),
            default=source.page_size[1],
        )
        peers = sorted(
            [
                line
                for line in lines
                if header[3] <= _bbox_center_y(line.bbox) < floor
                and header[0] <= line.bbox[0]
                and line.bbox[2] <= header[2] + 0.25 * em
            ],
            key=lambda line: (_bbox_center_y(line.bbox), line.bbox[0]),
        )
        rows = []
        for line in peers:
            if (
                rows
                and abs(_bbox_center_y(line.bbox) - statistics.median(_bbox_center_y(item.bbox) for item in rows[-1]))
                <= 0.5 * em
            ):
                rows[-1].append(line)
            else:
                rows.append([line])
        numeric_rows = 0
        members = list(labels)
        bottom = header[3]
        multiline_rows = 0
        accepted_rows = [labels]
        for row in rows:
            if min(line.bbox[1] for line in row) - bottom > 2.5 * em:
                break
            numeric = all(re.fullmatch(r"[\d\s.,%$€£()−+/-]+", line.text) for line in row)
            if any(
                min(abs(line.bbox[0] - left) for left in columns) > 0.6 * em
                and not (numeric and any(cell[0] <= _bbox_center_x(line.bbox) <= cell[2] for cell in cells))
                and not (
                    raster_support and len(row) >= 2 and line.effective_height <= em and re.fullmatch(r"\d{1,3}", line.text)
                )
                for line in row
            ):
                break
            if numeric:
                numeric_rows += 1
            elif len(row) >= 2:
                multiline_rows += 1
            elif len(row[0].text.split()) > 5 and not raster_support:
                break
            members.extend(row)
            accepted_rows.append(row)
            bottom = max(bottom, max(line.bbox[3] for line in row))
        horizontal_rules = [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and header[1] <= rule.bbox[1] <= bottom + em
            and _bbox_axis_overlap_ratio(header, rule.bbox, axis="x") >= 0.8
        ]
        if numeric_rows < 3 and not (
            multiline_rows >= 4
            and len(horizontal_rules) >= 3
            or multiline_rows >= 8
            and raster_support
            and min(line.effective_height for line in labels) >= 1.15 * em
        ):
            continue
        bbox = (header[0], header[1], header[2], bottom + 0.2 * em)
        inferred_grid = []
        if numeric_rows >= 3 and len(cells) == len(labels) and numeric_rows == len(accepted_rows) - 1:
            # 仅数字单行表由真实表头矩形限定列，视觉行中点限定行，不推断任意正文或多行表网格。
            x_edges = [cell[0] for cell in cells] + [header[2]]
            y_edges = (
                [header[1]]
                + [
                    (max(line.bbox[3] for line in a) + min(line.bbox[1] for line in b)) / 2
                    for a, b in zip(accepted_rows, accepted_rows[1:])
                ]
                + [bbox[3]]
            )
            inferred_grid = [_AxisLine((x - 0.05, bbox[1], x + 0.05, bbox[3]), 0.1, "vertical") for x in x_edges]
            inferred_grid.extend(_AxisLine((bbox[0], y - 0.05, bbox[2], y + 0.05), 0.1, "horizontal") for y in y_edges)
        stage_grid = _native_clip_stage_grid(source, labels, bbox) if raster_support and multiline_rows >= 8 else []
        if stage_grid:
            inferred_grid = stage_grid
        output.append(
            _TableCandidate(
                bbox=bbox,
                local_bbox=bbox,
                angle=0,
                score=8,
                core_bbox=bbox,
                line_indices={line.source_index for line in members},
                inferred_grid=inferred_grid,
                inferred_grid_authoritative=bool(stage_grid),
                preserve_inline_font_styles=bool(stage_grid),
            )
        )
    return output


def _native_clip_stage_grid(source: _PageSource, labels: list[_LineItem], bounds: BBox) -> list[_AxisLine]:
    """重复原生文字框限定多列功能行，首列连续编号限定阶段；空白首列续行以缺省横线表达跨行。"""
    if len(labels) < 3:
        return []
    em = statistics.median(line.effective_height for line in source.lines if line.effective_height > 0)
    columns = []
    for label in labels:
        boxes = [
            path.bbox
            for path in source.path_infos
            if path.segment_count == 5
            and not path.fill_visible
            and not path.stroke_visible
            and abs(path.bbox[0] - label.bbox[0]) <= 0.25 * em
            and bounds[1] <= path.bbox[1] < bounds[3]
            and path.bbox[2] <= bounds[2]
            and path.bbox[2] - path.bbox[0] >= 3 * em
        ]
        if len(boxes) < 6:
            return []
        width = statistics.median(box[2] - box[0] for box in boxes)
        boxes = [box for box in boxes if abs(box[2] - box[0] - width) <= 0.3 * em]
        if len(boxes) < 6:
            return []
        columns.append(boxes)
    function_boxes = sorted(
        [
            box
            for box in columns[1]
            if box[1] > max(line.bbox[3] for line in labels)
            and any(
                abs(line.bbox[0] - labels[1].bbox[0]) <= 0.25 * em and box[1] <= _bbox_center_y(line.bbox) <= box[3]
                for line in source.lines
            )
        ],
        key=lambda box: box[1],
    )
    if len(function_boxes) < 5 or any(a[3] >= b[1] for a, b in zip(function_boxes, function_boxes[1:])):
        return []
    rows = [[] for _ in function_boxes]
    for line in source.lines:
        if line in labels or line.angle != 0 or line.bbox[1] < function_boxes[0][1] - 0.9 * em:
            continue
        column = next(
            (
                index
                for index, label in enumerate(labels)
                if abs(line.bbox[0] - label.bbox[0]) <= 0.3 * em
                and line.bbox[2] <= max(box[2] for box in columns[index]) + 0.3 * em
            ),
            None,
        )
        if column is None:
            continue
        matching = [index for index, box in enumerate(function_boxes) if box[1] - 0.9 * em <= line.bbox[1]]
        if matching and line.bbox[3] <= bounds[3]:
            rows[matching[-1]].append((column, line))
    if any(len({column for column, line in row if column > 0}) != len(labels) - 1 for row in rows):
        return []
    stages = []
    for index, row in enumerate(rows):
        first = sorted([line for column, line in row if column == 0], key=lambda line: line.bbox[1])
        if first:
            number = re.match(r"\s*(\d{1,2})[.)、]\s*[A-Za-z\u3400-\u9fff]", first[0].text)
            if not number:
                return []
            stages.append((index, int(number.group(1))))
    if (
        len(stages) < 3
        or stages[0][0] != 0
        or len(stages) == len(rows)
        or any(b[1] - a[1] != 1 for a, b in zip(stages, stages[1:]))
    ):
        return []
    x_edges = [
        bounds[0],
        *((max(box[2] for box in column) + labels[index + 1].bbox[0]) / 2 for index, column in enumerate(columns[:-1])),
        bounds[2],
    ]
    row_boxes = [_bbox_union_many([line.bbox for column, line in row]) for row in rows]
    if any(a[3] >= b[1] for a, b in zip(row_boxes, row_boxes[1:])):
        return []
    y_edges = [
        bounds[1],
        (max(line.bbox[3] for line in labels) + row_boxes[0][1]) / 2,
        *((a[3] + b[1]) / 2 for a, b in zip(row_boxes, row_boxes[1:])),
        bounds[3],
    ]
    grid = [_AxisLine((x - 0.05, bounds[1], x + 0.05, bounds[3]), 0.1, "vertical") for x in x_edges]
    starts = {index + 1 for index, number in stages}
    for index, y in enumerate(y_edges):
        left = bounds[0] if index in {0, 1, len(y_edges) - 1} or index in starts else x_edges[1]
        grid.append(_AxisLine((left, y - 0.05, bounds[2], y + 0.05), 0.1, "horizontal"))
    return grid


_SHADED_GROUP_STATE_REFERENCES = (
    _native_clip_stage_grid,
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_union,
    _bbox_union_many,
    _LineItem.__getattribute__,
)


def _join_caption_supported_table_groups(source: _PageSource, candidates: list[_TableCandidate]) -> list[_TableCandidate]:
    """先按共享栏缘连接分组，再以独立表题确认整表，避免长表及双面板被截断。"""
    heights = [line.effective_height for line in source.lines if line.angle == 0 and line.effective_height > 0]
    em = statistics.median(heights) if heights else 10.0
    captions = [
        line
        for line in source.lines
        if line.angle == 0 and re.match(r"^\s*(?:table|tab\.?|表)\s*(?:[A-Z][.]?)?\d+[.:：]", line.text, re.IGNORECASE)
    ]
    candidates = _join_sparse_ruled_caption_tables(source, candidates, captions, em)
    pending = set(range(len(candidates)))
    output = []
    while pending:
        seed = pending.pop()
        group_indices, frontier = [seed], [seed]
        while frontier:
            current = candidates[frontier.pop()]
            adjacent = []
            for index in pending:
                other = candidates[index]
                if current.angle != 0 or other.angle != 0:
                    continue
                # 注释外框只用于整体去重；分组合并必须比较真实表体，避免表题和脚注成为接桥。
                a, b = current.core_bbox or current.local_bbox, other.core_bbox or other.local_bbox
                gap = max(0.0, a[1] - b[3], b[1] - a[3])
                barrier = any(
                    min(a[3], b[3]) <= caption.bbox[1] <= max(a[1], b[1])
                    and _bbox_axis_overlap_ratio(caption.bbox, a, axis="x") >= 0.5
                    for caption in captions
                )
                if abs(a[0] - b[0]) <= em and abs(a[2] - b[2]) <= em and gap <= 1.5 * em and not barrier:
                    adjacent.append(index)
            pending.difference_update(adjacent)
            frontier.extend(adjacent)
            group_indices.extend(adjacent)
        group = [candidates[index] for index in group_indices]
        bbox = group[0].core_bbox or group[0].local_bbox
        for candidate in group[1:]:
            bbox = _bbox_union(bbox, candidate.core_bbox or candidate.local_bbox)
        support = [
            caption
            for caption in captions
            if 0 <= caption.bbox[1] - bbox[3] <= 4 * em and _bbox_axis_overlap_ratio(caption.bbox, bbox, axis="x") >= 0.8
        ]
        if not support or group[0].angle != 0:
            output.extend(group)
            continue
        # 第一条横线上方可能依次是分组行及列名；每层必须提供多列或斜体分组排版证据。
        annotation_indices = {
            index for candidate in group for annotation in candidate.annotations for index in annotation.line_indices
        }
        for _ in range(3):
            header = [
                line
                for line in source.lines
                if line.angle == 0
                and line.source_index not in annotation_indices
                and line not in captions
                and bbox[0] - em <= line.bbox[0]
                and line.bbox[2] <= bbox[2] + em
                and 0 <= bbox[1] - line.bbox[3] <= 1.5 * em
            ]
            if not header:
                break
            bottom = max(line.bbox[3] for line in header)
            row = [line for line in header if abs(line.bbox[3] - bottom) <= 0.5 * em]
            separated_columns = len(row) >= 2 and max(line.bbox[0] for line in row) - min(line.bbox[0] for line in row) > 2 * em
            italic_group = len(row) == 1 and row[0].font_signature is not None and bool(row[0].font_signature[1] & (1 << 6))
            if not separated_columns and not italic_group:
                break
            bbox = (bbox[0], min(line.bbox[1] for line in row), bbox[2], bbox[3])
        matching_rules = [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and abs(rule.bbox[0] - bbox[0]) <= em
            and abs(rule.bbox[2] - bbox[2]) <= em
            and bbox[1] - em <= rule.bbox[1] <= bbox[3] + 0.25 * em
        ]
        if matching_rules:
            # 补回列名时同步纳入表格外线，否则恢复器会把完整表当成缺失顶边的片段。
            bbox = (
                min(bbox[0], *(rule[0] for rule in matching_rules)),
                min(bbox[1], *(rule[1] for rule in matching_rules)),
                max(bbox[2], *(rule[2] for rule in matching_rules)),
                max(bbox[3], *(rule[3] for rule in matching_rules)),
            )
        members = {
            line.source_index
            for line in source.lines
            if line.angle == 0
            and bbox[0] - 0.25 * em <= line.bbox[0]
            and line.bbox[2] <= bbox[2] + 0.25 * em
            and bbox[1] <= (line.bbox[1] + line.bbox[3]) / 2 <= bbox[3]
        }
        output.append(
            replace(
                group[0],
                bbox=_bbox_union_many(
                    [bbox, *(annotation.bbox for candidate in group for annotation in candidate.annotations)]
                ),
                local_bbox=bbox,
                core_bbox=bbox,
                line_indices=members,
                annotations=[annotation for candidate in group for annotation in candidate.annotations],
            )
        )
    return output


def _join_sparse_ruled_caption_tables(source, candidates, captions, em):
    """多段共享栏缘的横线链由独立表题闭合，补回稀疏分组及中间只有单列的标签。"""
    result = list(candidates)
    rules = [rule.bbox for rule in source.drawing_lines if rule.orientation == "horizontal"]
    for caption in captions:
        for seed in list(result):
            if seed.angle != 0 or seed.bbox[3] >= caption.bbox[1]:
                continue
            left, _, right, _ = seed.bbox
            if _bbox_axis_overlap_ratio(seed.bbox, caption.bbox, axis="x") < 0.8:
                continue
            previous = [
                line.bbox[3]
                for line in captions
                if line is not caption
                and line.bbox[3] < caption.bbox[1]
                and _bbox_axis_overlap_ratio(line.bbox, seed.bbox, axis="x") >= 0.8
            ]
            floor = max(previous, default=0)
            aligned = sorted(
                [
                    box
                    for box in rules
                    if abs(box[0] - left) <= em and abs(box[2] - right) <= em and floor < box[1] < caption.bbox[1]
                ],
                key=lambda box: box[1],
            )
            if len(aligned) < 6 or caption.bbox[1] - aligned[-1][3] > 2 * em:
                continue
            chain = [aligned[-1]]
            for box in reversed(aligned[:-1]):
                if chain[-1][1] - box[3] > 5 * em:
                    break
                chain.append(box)
            if len(chain) < 6:
                continue
            bounds = _bbox_union_many(chain)
            group = [
                candidate
                for candidate in result
                if candidate.angle == 0
                and abs(candidate.bbox[0] - left) <= em
                and abs(candidate.bbox[2] - right) <= em
                and _bbox_overlap_in_first(candidate.bbox, bounds) >= 0.9
            ]
            if len(group) < 2:
                continue
            rows = [
                line
                for line in source.lines
                if line.angle == 0
                and bounds[0] <= _bbox_center_x(line.bbox) <= bounds[2]
                and bounds[1] <= _bbox_center_y(line.bbox) <= bounds[3]
            ]
            if any(
                line.semantic_type not in {None, "header", "footer"}
                or line.bbox[0] < bounds[0] - 0.25 * em
                or line.bbox[2] > bounds[2] + 0.25 * em
                for line in rows
            ):
                continue
            if any(
                len(re.findall(r"[A-Za-z]{3,}|[\u3400-\u9fff]", line.text)) >= 8
                and line.bbox[2] - line.bbox[0] >= 0.85 * (right - left)
                for line in rows
            ):
                continue
            result = [candidate for candidate in result if not any(candidate is member for member in group)]
            result.append(
                replace(
                    group[0],
                    bbox=bounds,
                    local_bbox=bounds,
                    core_bbox=bounds,
                    line_indices={line.source_index for line in rows},
                    annotations=[annotation for candidate in group for annotation in candidate.annotations],
                )
            )
    return result


def _externalize_table_continuation_captions(
    source: _PageSource,
    candidates: list[_TableCandidate],
) -> None:
    """以连通物理网格收紧表体，并把其上方续表标记外置为 caption。"""

    for candidate in candidates:
        angle_lines = [line for line in source.lines if line.angle == candidate.angle]
        if not angle_lines:
            continue
        fragments = _build_fragments(
            angle_lines,
            source.page_size,
        )
        if not fragments:
            continue
        median_height = _median_fragment_height(fragments)
        local_axis_lines = _transform_axis_lines(
            source.drawing_lines,
            source.page_size,
            candidate.angle,
        )
        grid_matches = [
            grid_bbox
            for grid_bbox in _connected_rule_grid_bboxes(
                local_axis_lines,
                median_height,
            )
            if _bbox_axis_overlap_ratio(
                grid_bbox,
                candidate.local_bbox,
                axis="x",
            )
            >= 0.9
            and _bbox_overlap_in_smaller(
                grid_bbox,
                candidate.local_bbox,
            )
            >= 0.9
        ]
        if not grid_matches:
            continue
        grid_bbox = max(
            grid_matches,
            key=lambda bbox: _bbox_area(bbox),
        )
        continuation_lines = []
        for line in angle_lines:
            if (
                _TABLE_CONTINUATION_RE.fullmatch(
                    line.text.strip(),
                )
                is None
            ):
                continue
            local_bbox = _rotate_bbox_to_upright(
                line.bbox,
                source.page_size,
                candidate.angle,
            )
            gap = grid_bbox[1] - local_bbox[3]
            if (
                local_bbox[3] <= grid_bbox[1] + 0.25 * median_height
                and -0.25 * median_height <= gap <= 3.0 * median_height
                and _bbox_axis_overlap_ratio(
                    local_bbox,
                    grid_bbox,
                    axis="x",
                )
                >= 0.8
            ):
                continuation_lines.append(
                    (line, local_bbox),
                )
        if len(continuation_lines) != 1:
            continue
        continuation_line, continuation_local_bbox = continuation_lines[0]
        continuation_bbox = continuation_line.bbox
        existing_caption = next(
            (annotation for annotation in candidate.annotations if annotation.kind == "caption"),
            None,
        )
        if existing_caption is None:
            candidate.annotations.append(
                _TableAnnotation(
                    kind="caption",
                    bbox=continuation_bbox,
                    line_indices={continuation_line.source_index},
                    line_bboxes={
                        continuation_line.source_index: continuation_bbox,
                    },
                )
            )
        else:
            existing_caption.bbox = _bbox_union(
                existing_caption.bbox,
                continuation_bbox,
            )
            existing_caption.line_indices.add(
                continuation_line.source_index,
            )
            existing_caption.line_bboxes[continuation_line.source_index] = continuation_bbox
        candidate.line_indices.discard(
            continuation_line.source_index,
        )
        core_bbox = _rotate_bbox_from_upright(
            grid_bbox,
            source.page_size,
            candidate.angle,
        )
        candidate.core_bbox = core_bbox
        candidate.local_bbox = _bbox_union(
            grid_bbox,
            continuation_local_bbox,
        )
        candidate.bbox = _bbox_union(
            core_bbox,
            continuation_bbox,
        )


def has_native_closed_grid(source, candidate, *, allow_two_by_two: bool = False):
    """真实横竖边界贯穿候选且多行列含原生文字时，表格不应被附近Figure图题认领。"""
    if candidate.angle != 0:
        return False
    box = candidate.core_bbox or candidate.bbox
    if any(
        _bbox_overlap_in_first(box, form) >= 0.95 and _bbox_area(form) > 1.5 * _bbox_area(box) for form in source.form_bboxes
    ):
        # 组合图Form内的示意小网格由完整图形认领，不能切断示意图的整体上下文。
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    members = [line for line in source.lines if line.source_index in candidate.line_indices]
    if len(members) < (4 if allow_two_by_two else 6) or width <= 0 or height <= 0:
        return False
    em = statistics.median(line.effective_height for line in members if line.effective_height > 0)
    tolerance = max(1.0, 0.2 * em)
    horizontal = [
        rule
        for rule in source.drawing_lines
        if rule.orientation == "horizontal"
        and abs(rule.bbox[0] - box[0]) <= tolerance
        and abs(rule.bbox[2] - box[2]) <= tolerance
        and box[1] - tolerance <= rule.bbox[1] <= box[3] + tolerance
    ]
    vertical = [
        rule
        for rule in source.drawing_lines
        if rule.orientation == "vertical"
        and abs(rule.bbox[1] - box[1]) <= tolerance
        and abs(rule.bbox[3] - box[3]) <= tolerance
        and box[0] - tolerance <= rule.bbox[0] <= box[2] + tolerance
    ]
    xs = sorted({round((rule.bbox[0] + rule.bbox[2]) / 2, 2) for rule in vertical})
    ys = sorted({round((rule.bbox[1] + rule.bbox[3]) / 2, 2) for rule in horizontal})
    if len(xs) < 3 or len(ys) < (3 if allow_two_by_two else 4) or min(xs[0] - box[0], ys[0] - box[1]) < -tolerance:
        return False
    if max(abs(xs[0] - box[0]), abs(xs[-1] - box[2]), abs(ys[0] - box[1]), abs(ys[-1] - box[3])) > tolerance:
        return False
    occupied = {
        (sum(x < (line.bbox[0] + line.bbox[2]) / 2 for x in xs), sum(y < (line.bbox[1] + line.bbox[3]) / 2 for y in ys))
        for line in members
    }
    return len({column for column, _ in occupied}) >= 2 and len({row for _, row in occupied}) >= (2 if allow_two_by_two else 3)
