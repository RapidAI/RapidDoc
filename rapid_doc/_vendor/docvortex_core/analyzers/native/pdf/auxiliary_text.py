"""分类页眉、页脚、页码、侧栏和页脚注。"""

from __future__ import annotations

import re
import statistics
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Literal


from ....schema import BBox

from .models import _AxisLine, _LineItem, _LocalAxisLine, _MarginalCandidate, _PageSource, _PreparedPage, _TextLane
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_intersects,
    _bbox_overlap_in_first,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _expand_bbox,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .layout_evidence import build_layout_evidence
from .text_roles import metadata_field
from .line_layout import _effective_text_row_gap, _infer_text_lanes, _line_effective_height, _line_canonical_style_scale

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?[\-\u2013\u2014\u00b7\u2022]*\s*(?:\u7b2c\s*)?"
    r"(?P<value>\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+)"
    r"(?:\s*(?:/|of|\u5171)\s*(?:\d{1,4}|[ivxlcdm]+|[\u3007\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u4e24]+))?"
    r"\s*(?:\u9875)?\s*[\-\u2013\u2014\u00b7\u2022]*\s*$",
    re.IGNORECASE,
)

# 页眉或页脚同伴块允许的最大高度占页面高度比例，近整页容器不允许仅凭与页码行重叠改判。
_MARGINAL_ROW_MAX_HEIGHT_RATIO = 0.15


def _classify_page_auxiliary_text(prepared: _PreparedPage) -> None:
    """在容器认领后仅按空间关系标注侧栏文字和页脚注。"""

    _classify_aside_text(prepared.remaining_lines, prepared.page_size)
    _classify_image_footnotes(
        prepared.remaining_lines,
        [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
        prepared.table_bboxes,
        prepared.drawing_lines,
        prepared.page_size,
        prepared=prepared,
    )
    prepared.page_footnote_groups += _classify_page_footnotes(
        prepared.remaining_lines,
        prepared.table_bboxes,
        prepared.drawing_lines,
        prepared.page_size,
        visual_bboxes=[block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
        prepared=prepared,
    )


def _classify_first_page_correspondence_footnotes(source: _PageSource) -> None:
    """首页通讯字段只认领同一实际栏或跨栏元数据带，字号、连续性及正文屏障共同限制范围。"""
    width, height = source.page_size
    lower = [line for line in source.lines if line.angle == 0 and line.bbox[1] >= 0.65 * height and line.semantic_type is None]
    contacts = [line for line in lower if metadata_field(line.text) == "contact"]
    body = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.angle == 0 and 0.2 * height < line.bbox[1] < 0.75 * height and line.bbox[2] - line.bbox[0] > 0.3 * width
    ]
    if not contacts or not body:
        return
    body_height = statistics.median(body)
    layout = build_layout_evidence(source.lines, source.page_size)
    for anchor in contacts:
        corridor = layout.corridor(anchor.bbox)
        if corridor is None:
            continue
        fields = [
            line
            for line in lower
            if metadata_field(line.text)
            and layout.corridor(line.bbox) in (corridor, (0.0, width))
            and abs(line.bbox[1] - anchor.bbox[1]) <= 5 * body_height
        ]
        if not fields:
            continue
        top = min(line.bbox[1] for line in fields)
        members = []
        for line in sorted(lower, key=lambda item: (item.bbox[1], item.bbox[0])):
            if line.bbox[1] < top or not corridor[0] <= _bbox_center_x(line.bbox) <= corridor[1]:
                continue
            if _line_effective_height(line, line.bbox) > 0.85 * body_height:
                break
            if members and (
                line.bbox[1] - members[-1].bbox[3] > 1.5 * body_height
                or _bbox_axis_overlap_ratio(anchor.bbox, line.bbox, axis="x") < 0.5
            ):
                break
            if members and re.match(r"^\s*\d+[.)]\s+", line.text):
                break
            members.append(line)
        if anchor in members and len({metadata_field(line.text) for line in members} - {None}) >= 2:
            for line in members:
                line.semantic_type = "page_footnote"


def _classify_aside_text(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> None:
    """在横排正文占绝对多数时，以边缘带和物理尺寸识别垂直侧栏。"""

    available = [line for line in lines if line.semantic_type is None]
    upright_lines = [line for line in available if line.angle == 0]
    if len(upright_lines) < 4:
        return

    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    total_support = sum(support_by_angle.values())
    if total_support <= 0 or support_by_angle.get(0, 0.0) / total_support < 0.8:
        return

    page_width, page_height = page_size
    if page_width <= 0 or page_height <= 0:
        return
    # 侧栏必须完整位于 12% 边缘带，且兼具不超过 8% 的窄宽和至少 15% 的物理高度。
    aside_source_indices = {
        line.source_index
        for line in available
        if line.angle in {90, 270}
        and line.bbox[2] - line.bbox[0] <= 0.08 * page_width
        and line.bbox[3] - line.bbox[1] >= 0.15 * page_height
        and (line.bbox[2] <= 0.12 * page_width or line.bbox[0] >= 0.88 * page_width)
    }
    for line in available:
        if line.source_index in aside_source_indices:
            line.semantic_type = "aside_text"


def _geometric_text_support_by_angle(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[int, float]:
    """按局部行宽乘有效行高累计各文字方向的纯几何支持度。"""

    support_by_angle: dict[int, float] = {}
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        local_width = max(0.1, local_bbox[2] - local_bbox[0])
        support_by_angle[line.angle] = support_by_angle.get(line.angle, 0.0) + (
            local_width * _line_effective_height(line, local_bbox)
        )
    return support_by_angle


def _classify_image_footnotes(
    lines: list[_LineItem],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    *,
    reference_body_height: float | None = None,
    prepared: _PreparedPage | None = None,
) -> None:
    """用图片、下缘长横线和紧凑小字的联合关系识别图表脚注。"""

    available = [line for line in lines if line.semantic_type is None]
    if not available or not image_bboxes or not drawing_lines:
        return
    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    if not support_by_angle:
        return
    dominant_angle = max(
        sorted(support_by_angle),
        key=lambda angle: support_by_angle[angle],
    )
    local_page_size = (page_size[1], page_size[0]) if dominant_angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return

    line_geometry = sorted(
        [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, dominant_angle))
            for line in available
            if line.angle == dominant_angle
        ],
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if not line_geometry:
        return
    if reference_body_height is not None and reference_body_height > 0:
        # 图片占主导的稀疏页可能只剩图注和脚注，延迟复核时改用全文正文尺度。
        body_height = max(0.1, reference_body_height)
    else:
        body_samples = [
            _line_effective_height(line, bbox) for line, bbox in line_geometry if bbox[2] - bbox[0] >= 0.2 * local_page_width
        ]
        if not body_samples:
            body_samples = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        body_height = max(0.1, statistics.median(body_samples))
    local_images = [_rotate_bbox_to_upright(bbox, page_size, dominant_angle) for bbox in image_bboxes]
    if prepared is None:
        local_axis_lines = _transform_axis_lines(drawing_lines, page_size, dominant_angle)
        table_lines = _confirmed_table_horizontal_lines(local_axis_lines, table_bboxes)
    else:
        local_axis_lines, table_lines = _prepared_local_axis_table_lines(prepared, dominant_angle)
    matched_source_indices: set[int] = set()
    for image_bbox in local_images:
        image_width = max(0.1, image_bbox[2] - image_bbox[0])
        # 同一视觉行的并排图可能高度略有差异；共享较低下缘可避免把留白误作远距。
        row_bottom = max(
            peer_bbox[3] for peer_bbox in local_images if _bbox_axis_overlap_ratio(image_bbox, peer_bbox, axis="y") >= 0.5
        )
        candidate_rules = [
            axis_line
            for axis_line in local_axis_lines
            if axis_line.orientation == "horizontal"
            and 0.75 * image_width <= axis_line.bbox[2] - axis_line.bbox[0] <= 1.3 * image_width
            and max(
                0.0,
                min(axis_line.bbox[2], image_bbox[2]) - max(axis_line.bbox[0], image_bbox[0]),
            )
            >= 0.85 * image_width
            # 图片外框的底边属于图形本身，不能拿来证明下方文字是图表脚注。
            and 0.0 <= axis_line.bbox[1] - row_bottom <= max(0.01 * local_page_height, 0.75 * body_height)
            and not _rule_belongs_to_confirmed_table(
                axis_line,
                table_lines,
                local_page_width,
            )
        ]
        if not candidate_rules:
            continue
        rule = min(
            candidate_rules,
            key=lambda item: (max(0.0, item.bbox[1] - row_bottom), item.bbox[1]),
        )
        matched_source_indices.update(
            _image_footnote_members(
                line_geometry,
                rule.bbox,
                body_height,
                local_page_height,
            )
        )

    for line in available:
        if line.source_index in matched_source_indices:
            line.semantic_type = "footnote"


def _classify_deferred_image_footnotes(
    prepared_pages: list[_PreparedPage],
    body_height: float,
) -> None:
    """在全文正文尺度确定后，仅重试仍未分类的图片脚注候选。"""

    if body_height <= 0:
        return
    for prepared in prepared_pages:
        _classify_image_footnotes(
            prepared.remaining_lines,
            [block["bbox"] for block in prepared.fixed_blocks if block.get("type") == "image"],
            prepared.table_bboxes,
            prepared.drawing_lines,
            prepared.page_size,
            reference_body_height=body_height,
            prepared=prepared,
        )


def _image_footnote_members(
    line_geometry: list[tuple[_LineItem, BBox]],
    rule_bbox: BBox,
    body_height: float,
    local_page_height: float,
) -> set[int]:
    """返回长横线下方、位于同一水平走廊内的连续小字号文本行。"""

    first_gap_limit = max(0.025 * local_page_height, 2.0 * body_height)
    horizontal_tolerance = 0.5 * body_height
    candidates = [
        item
        for item in line_geometry
        if -0.25 * body_height <= item[1][1] - rule_bbox[3] <= first_gap_limit
        and item[1][0] >= rule_bbox[0] - horizontal_tolerance
        and item[1][2] <= rule_bbox[2] + horizontal_tolerance
        and _line_effective_height(*item) <= 0.9 * body_height
    ]
    if not candidates:
        return set()
    first = min(candidates, key=lambda item: (item[1][1], item[1][0]))
    members = [first]
    continuation_gap_limit = max(1.25 * _line_effective_height(*first), 0.01 * local_page_height)
    for current in line_geometry:
        if current[0] is first[0] or current[1][1] < first[1][1]:
            continue
        if current[1][0] < rule_bbox[0] - horizontal_tolerance:
            continue
        if current[1][2] > rule_bbox[2] + horizontal_tolerance:
            continue
        if _line_effective_height(*current) > 0.95 * body_height:
            continue
        if _effective_text_row_gap(members[-1], current) > continuation_gap_limit:
            break
        members.append(current)
    return {line.source_index for line, _bbox in members}


def _footnote_visual_rows(
    geometry: list[tuple[_LineItem, BBox]],
) -> tuple[list[tuple[_LineItem, BBox]], dict[int, set[int]]]:
    """按紧邻端点与同基线重建脚注判定用的视觉行，保留原始成员供最终认领。"""
    groups: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(geometry, key=lambda item: (_bbox_center_y(item[1]), item[1][0])):
        line, bbox = item
        joined = None
        for group in reversed(groups):
            reference = _bbox_union_many([b for _, b in group])
            height = min(
                _line_effective_height(line, bbox),
                statistics.median(_line_effective_height(member, bounds) for member, bounds in group),
            )
            if _bbox_center_y(bbox) - _bbox_center_y(reference) > 2 * height:
                break
            if (
                abs(_bbox_center_y(bbox) - _bbox_center_y(reference)) <= 0.4 * height
                and _bbox_axis_overlap_ratio(bbox, reference, axis="y") >= 0.6
                and min(abs(bbox[0] - reference[2]), abs(reference[0] - bbox[2])) <= 0.75 * height
            ):
                joined = group
                break
        if joined is None:
            groups.append([item])
        else:
            joined.append(item)
    output = []
    members = {}
    for group in groups:
        first = min((line for line, _ in group), key=lambda line: line.source_index)
        members[first.source_index] = {line.source_index for line, _ in group}
        if len(group) == 1:
            output.append(group[0])
        else:
            height = statistics.median(_line_effective_height(line, bbox) for line, bbox in group)
            merged = replace(
                first, bbox=_bbox_union_many([line.bbox for line, _ in group]), effective_height=height, em_height=height
            )
            output.append((merged, _bbox_union_many([bbox for _, bbox in group])))
    return output, members


def _classify_page_footnotes(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    *,
    visual_bboxes: list[BBox] | None = None,
    prepared: _PreparedPage | None = None,
    reference_lines: list[_LineItem] | None = None,
) -> list[set[int]]:
    """识别主方向页脚注，并按触发分隔线返回来源编号分组。"""

    available = [line for line in lines if line.semantic_type is None]
    if not available:
        return []
    support_by_angle = _geometric_text_support_by_angle(available, page_size)
    if not support_by_angle:
        return []
    dominant_angle = max(
        sorted(support_by_angle),
        key=lambda angle: support_by_angle[angle],
    )
    line_geometry = [
        (line, _rotate_bbox_to_upright(line.bbox, page_size, dominant_angle))
        for line in available
        if line.angle == dominant_angle
    ]
    if not line_geometry:
        return []

    # 分式证据使用原始墨迹，脚注栏与续行则使用恢复后的整条物理行。
    ink_bboxes = [
        _rotate_bbox_to_upright(line.ink_bbox or line.bbox, page_size, dominant_angle) for line, _bbox in line_geometry
    ]
    line_geometry, row_members = _footnote_visual_rows(line_geometry)

    local_page_size = (page_size[1], page_size[0]) if dominant_angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return []
    effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    lanes = _infer_text_lanes(
        line_geometry,
        local_page_width,
        median_height,
        # 脚注分隔线应对齐稳定栏锚点，不能被页眉或跨栏关键词的宽行扩张污染。
        recalculate_intervals=False,
    )
    if prepared is None:
        local_axis_lines = _transform_axis_lines(drawing_lines, page_size, dominant_angle)
        table_lines = _confirmed_table_horizontal_lines(local_axis_lines, table_bboxes)
    else:
        local_axis_lines, table_lines = _prepared_local_axis_table_lines(prepared, dominant_angle)

    candidate_groups: list[set[int]] = _unruled_numbered_footnote_groups(line_geometry, local_page_size)
    visual_bboxes = visual_bboxes or []
    # 图体是此规则的必要证据；没有图体时无需读取每个字符的上标编号。
    if visual_bboxes:
        candidate_groups.extend(
            _chart_referenced_note_groups(
                line_geometry,
                visual_bboxes,
                lambda: _native_note_reference_values(reference_lines or available),
                local_page_size,
            )
        )
    for axis_line in local_axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        # 常规短分隔线仍要求进入页面下方 30%；栏宽分隔线可在下方 45% 内
        # 依靠严格的单栏对齐和字号收缩证据提前触发。
        rule_center_y = _bbox_center_y(axis_line.bbox)
        if rule_center_y < 0.55 * local_page_height:
            continue
        # 短横线上下紧贴且同宽的两行是分式几何，不能提前认领分母及后续正文为脚注。
        rule_width = axis_line.bbox[2] - axis_line.bbox[0]
        compact_neighbors = [
            bbox
            for bbox in ink_bboxes
            if 0.4 * rule_width <= bbox[2] - bbox[0] <= rule_width + median_height
            and _bbox_axis_overlap_ratio(bbox, axis_line.bbox, axis="x") >= 0.8
            and abs(_bbox_center_x(bbox) - _bbox_center_x(axis_line.bbox)) <= 0.15 * rule_width
        ]
        if (
            rule_width < 0.25 * local_page_width
            and any(0 <= rule_center_y - bbox[3] <= median_height for bbox in compact_neighbors)
            and any(0 <= bbox[1] - rule_center_y <= median_height for bbox in compact_neighbors)
        ):
            continue
        # 表格边界会产生断裂横线；除框内线段外，也排除与其同高且近邻的框外线段。
        if _rule_belongs_to_confirmed_table(
            axis_line,
            table_lines,
            local_page_width,
        ):
            continue
        if any(
            _bbox_intersects(
                _expand_bbox(axis_line.original_bbox, max(0.5, axis_line.width)),
                visual_bbox,
            )
            for visual_bbox in visual_bboxes
        ):
            # 图形坐标轴和外框不能充当页面脚注分隔线。
            continue
        rule_source_indices: set[int] = set()
        rule_lanes = list(lanes)
        rule_width = axis_line.bbox[2] - axis_line.bbox[0]
        # 通栏长线自身提供局部栏界；上方双栏布局不能拆散线下的整页脚注。
        if rule_width >= 0.7 * local_page_width:
            corridor = [
                (line, bbox)
                for line, bbox in line_geometry
                if bbox[0] >= axis_line.bbox[0] - 0.25 * median_height and bbox[2] <= axis_line.bbox[2] + 0.25 * median_height
            ]
            rule_lanes.append(_TextLane(axis_line.bbox[0], axis_line.bbox[2], corridor))
        for lane in rule_lanes:
            following_rule_tops = [
                other.bbox[1]
                for other in local_axis_lines
                if other.orientation == "horizontal"
                and other.bbox[2] - other.bbox[0] >= max(4.0 * median_height, 0.04 * local_page_width)
                and other.bbox[1] - axis_line.bbox[3] > 0.5 * median_height
                and _bbox_axis_overlap_ratio(
                    axis_line.bbox,
                    other.bbox,
                    axis="x",
                )
                >= 0.8
            ]
            rule_source_indices.update(
                _footnote_lane_members(
                    lane,
                    axis_line.bbox,
                    local_page_size,
                    page_median_height=median_height,
                    lane_width_reference=_footnote_lane_width_reference(
                        lane,
                        lanes,
                        median_height,
                    ),
                    allow_column_width_rule=(rule_center_y >= 0.55 * local_page_height),
                    lower_barrier_y=(min(following_rule_tops) if following_rule_tops else None),
                )
            )
        if rule_source_indices:
            candidate_groups.append(rule_source_indices)

    page_footnote_groups = _merge_overlapping_source_groups(candidate_groups)
    _augment_footnote_groups_with_edge_markers(
        page_footnote_groups,
        line_geometry,
        median_height,
    )
    page_footnote_groups = [set().union(*(row_members[index] for index in group)) for group in page_footnote_groups]
    footnote_source_indices = set().union(*page_footnote_groups) if page_footnote_groups else set()
    for line in available:
        if line.source_index in footnote_source_indices:
            line.semantic_type = "page_footnote"
    return page_footnote_groups


def _native_note_reference_values(lines: list[_LineItem]) -> set[str]:
    """读取行内小号且抬高的原生数字，给同字号无线脚注提供编号对应证据。"""
    values = set()
    for line in lines:
        chars = [char for char in line.chars if str(char.get("char", "")).strip() and char.get("origin")]
        if len(chars) < 4:
            continue
        heights = [char["bbox"][3] - char["bbox"][1] for char in chars]
        reference_height = statistics.median(heights)
        baseline = statistics.median(char["origin"][1] for char in chars)
        digits = ""
        for char in chars:
            if (
                str(char.get("char", "")).isdigit()
                and char["bbox"][3] - char["bbox"][1] <= 0.82 * reference_height
                and baseline - char["origin"][1] >= 0.15 * reference_height
            ):
                digits += str(char["char"])
            else:
                if 1 <= len(digits) <= 3:
                    values.add(digits)
                digits = ""
        if 1 <= len(digits) <= 3:
            values.add(digits)
    return values


def _chart_referenced_note_groups(line_geometry, visual_bboxes, references, page_size):
    """图体下方的编号说明须有原生上标对应；同栏同式连续编号延续注释，正文列表和页脚不参与。"""
    height = page_size[1]
    ordered = sorted(line_geometry, key=lambda item: (item[1][1], item[1][0]))
    pattern = re.compile(r"^\s*(\d{1,3})[.)]?\s+\S")
    groups = []
    consumed = set()
    reference_values = None
    for index, (first, bounds) in enumerate(ordered):
        marker = pattern.match(first.text)
        em = _line_effective_height(first, bounds)
        if not marker or first.source_index in consumed or bounds[1] < 0.7 * height:
            continue
        charts = [
            box
            for box in visual_bboxes
            if box[3] - box[1] >= 8 * em
            and box[2] - box[0] >= 0.2 * page_size[0]
            and 0.8 * em <= bounds[1] - box[3] <= 12 * em
            and _bbox_axis_overlap_ratio(bounds, box, axis="x") >= 0.6
        ]
        if not charts or (first.dominant_font_weight or 400) >= 600:
            continue
        # 只有可能认领的图下注释才读取整页上标，必要证据与原有候选覆盖保持一致。
        if reference_values is None:
            reference_values = references() if callable(references) else references
        if marker[1] not in reference_values:
            continue
        members = [(first, bounds)]
        number = int(marker[1])
        for line, box in ordered[index + 1 :]:
            if _bbox_axis_overlap_ratio(bounds, box, axis="x") < 0.3:
                continue
            if (
                not 0.85 * em <= _line_effective_height(line, box) <= 1.15 * em
                or line.font_signature != first.font_signature
                or abs(box[0] - bounds[0]) > 0.4 * em
                or _effective_text_row_gap(members[-1], (line, box)) > 1.5 * em
            ):
                break
            next_marker = pattern.match(line.text)
            if next_marker:
                if int(next_marker[1]) != number + 1:
                    break
                group = {item.source_index for item, _ in members}
                groups.append(group)
                consumed.update(group)
                members = []
                number += 1
            members.append((line, box))
        group = {item.source_index for item, _ in members}
        groups.append(group)
        consumed.update(group)
    return groups


def _unruled_numbered_footnote_groups(
    line_geometry: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
) -> list[set[int]]:
    """用底部编号、字号收缩与正文净空识别无线注释，同字号跨栏注释另需原生上标对应。"""
    width, height = page_size
    ordered = sorted(line_geometry, key=lambda item: (item[1][1], item[1][0]))
    references = None
    marker_pattern = re.compile(r"^\s*(\d{1,3})[.)]\s+\S")
    groups = []
    consumed = set()
    for index, (first, bounds) in enumerate(ordered):
        marker = marker_pattern.match(first.text)
        if (
            first.source_index in consumed
            or bounds[1] < 0.7 * height
            or not marker
            or (first.dominant_font_weight or 400) >= 600
        ):
            continue
        first_height = _line_effective_height(first, bounds)
        above = [
            (line, box)
            for line, box in ordered[:index]
            if _bbox_center_y(box) <= _bbox_center_y(bounds) - 0.5 * first_height
            and box[2] - box[0] >= 0.25 * width
            and _bbox_axis_overlap_ratio(box, bounds, axis="x") >= 0.3
        ]
        if len(above) < 3:
            continue
        body_height = statistics.median(_line_effective_height(line, box) for line, box in above)
        # 松字框可能跨越下一行；按最近物理行及有效字高测量净空，避免遗漏重叠的参考文献续行。
        nearest_above = max(above, key=lambda item: _bbox_center_y(item[1]))
        gap = _effective_text_row_gap(nearest_above, (first, bounds))
        shrunken = first_height <= 0.9 * body_height and gap >= 0.75 * body_height
        body_left = statistics.median(box[0] for _, box in above)
        spanning_geometry = (
            bounds[2] - bounds[0] >= 0.65 * width
            and first_height <= 1.25 * body_height
            and gap >= 2 * body_height
            and body_left - bounds[0] >= 0.5 * body_height
        )
        # 缩小字号已足以认领，或者跨栏几何不成立时，不重复扫描整页字符。
        spanning_reference = False
        if not shrunken and spanning_geometry:
            if references is None:
                references = _native_note_reference_values([line for line, _ in ordered])
            spanning_reference = marker[1] in references
        if not shrunken and not spanning_reference:
            continue
        members = [(first, bounds)]
        for current, box in ordered[index + 1 :]:
            if box[1] < bounds[1] or _bbox_axis_overlap_ratio(bounds, box, axis="x") < 0.3:
                continue
            current_height = _line_effective_height(current, box)
            row_gap = _effective_text_row_gap(members[-1], (current, box))
            if (
                marker_pattern.match(current.text)
                or current_height > 1.15 * first_height
                or row_gap > 1.5 * first_height
                or not bounds[0] - 0.4 * first_height <= box[0] <= bounds[0] + 2 * first_height
            ):
                break
            # 独立页脚可能只比脚注略小；底边窄行及额外净空共同阻止页脚混入。
            if (
                box[1] >= 0.92 * height
                and box[2] - box[0] <= 0.55 * (bounds[2] - bounds[0])
                and current_height <= 0.95 * first_height
                and row_gap >= 0.25 * first_height
            ):
                break
            members.append((current, box))
        group = {line.source_index for line, _ in members}
        groups.append(group)
        consumed.update(group)
    return groups


def _augment_footnote_groups_with_edge_markers(
    groups: list[set[int]],
    line_geometry: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> None:
    """把脚注正文左侧同高的窄编号标记补入对应分隔线分组。"""

    geometry_by_source = {line.source_index: (line, bbox) for line, bbox in line_geometry}
    for group in groups:
        members = [geometry_by_source[source_index] for source_index in group if source_index in geometry_by_source]
        if not members:
            continue
        group_top = min(bbox[1] for _line, bbox in members)
        group_bottom = max(bbox[3] for _line, bbox in members)
        content_left = min(bbox[0] for _line, bbox in members)
        for line, bbox in line_geometry:
            if line.source_index in group:
                continue
            line_width = bbox[2] - bbox[0]
            center_y = _bbox_center_y(bbox)
            if (
                line_width <= 1.5 * median_height
                and content_left - 2.0 * median_height <= bbox[0] <= content_left
                and bbox[2] <= content_left + 0.5 * median_height
                and group_top - median_height <= center_y <= group_bottom + median_height
            ):
                group.add(line.source_index)


def _prepared_local_axis_table_lines(
    prepared: _PreparedPage,
    dominant_angle: int,
) -> tuple[list[_LocalAxisLine], list[_LocalAxisLine]]:
    """按页和主方向复用轴线变换及确认表格横线，输入身份变化时重新计算。"""

    key = (dominant_angle, id(prepared.drawing_lines), id(prepared.table_bboxes))
    cached = prepared.local_axis_table_cache.get(key)
    if cached is not None:
        return cached
    local_axis_lines = _transform_axis_lines(prepared.drawing_lines, prepared.page_size, dominant_angle)
    table_lines = _confirmed_table_horizontal_lines(local_axis_lines, prepared.table_bboxes)
    result = (local_axis_lines, table_lines)
    prepared.local_axis_table_cache[key] = result
    return result


def _confirmed_table_horizontal_lines(
    local_axis_lines: list[_LocalAxisLine],
    table_bboxes: list[BBox],
) -> list[_LocalAxisLine]:
    """按页预计算与确认表格框相交的横线，供所有候选排除判断复用。"""

    return [
        table_line
        for table_line in local_axis_lines
        if table_line.orientation == "horizontal"
        and any(
            _bbox_intersects(
                _expand_bbox(table_line.original_bbox, max(0.5, table_line.width)),
                table_bbox,
            )
            for table_bbox in table_bboxes
        )
    ]


def _rule_belongs_to_confirmed_table(
    candidate: _LocalAxisLine,
    table_lines: list[_LocalAxisLine],
    local_page_width: float,
) -> bool:
    """把表格框内横线及其同高近邻断裂段一并排除，避免框外残段触发脚注。"""

    maximum_segment_gap = 0.04 * local_page_width
    for table_line in table_lines:
        center_tolerance = max(1.0, candidate.width, table_line.width)
        if abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(table_line.bbox)) > center_tolerance:
            continue
        if _horizontal_bbox_gap(candidate.bbox, table_line.bbox) <= maximum_segment_gap:
            return True
    return False


def _merge_overlapping_source_groups(groups: list[set[int]]) -> list[set[int]]:
    """合并共享来源行的分隔线候选组，消除重复绘图线造成的重复分组。"""

    merged: list[set[int]] = []
    for group in groups:
        combined = set(group)
        index = 0
        while index < len(merged):
            if combined & merged[index]:
                combined.update(merged.pop(index))
                index = 0
                continue
            index += 1
        merged.append(combined)
    return sorted(merged, key=lambda group: min(group))


def _native_lowercase_ink_height(line: _LineItem) -> float | None:
    """取原生小写字形的稳健高度，用可见字形补充异常字体度量，不引入OCR。"""
    heights = []
    for char in line.chars:
        text = str(char.get("char", ""))
        box = _coerce_bbox(char.get("tight_bbox"))
        if len(text) == 1 and "a" <= text <= "z" and box is not None and box[3] > box[1]:
            heights.append(box[3] - box[1])
    return statistics.median(heights) if len(heights) >= 6 else None


def _single_line_ink_note_evidence(
    first: _LineItem,
    above: list[tuple[_LineItem, BBox]],
    rule_bbox: BBox,
    page_size: tuple[float, float],
    body_height: float,
) -> bool:
    """页底通栏线、编号净空及相对字形缩小共同确认单行注释，普通编号正文不享受例外。"""
    em = body_height
    if (
        rule_bbox[2] - rule_bbox[0] < 0.7 * page_size[0]
        or _bbox_center_y(rule_bbox) < 0.8 * page_size[1]
        or (first.dominant_font_weight or 400) >= 600
        or abs(first.bbox[0] - rule_bbox[0]) > em
        or first.bbox[2] - first.bbox[0] > 0.7 * (rule_bbox[2] - rule_bbox[0])
        or not (match := re.match(r"^\s*\d{1,3}[.)]\s+", first.text))
        or len(above) < 3
        or any(first.bbox[1] - box[3] < 0.75 * em for _, box in above)
    ):
        return False
    letters = [char for char in first.chars if str(char.get("char", "")).strip()]
    marker_length = len(match.group().strip())
    if len(letters) <= marker_length:
        return False
    marker = [_coerce_bbox(char.get("tight_bbox")) for char in letters[:marker_length]]
    following = _coerce_bbox(letters[marker_length].get("tight_bbox"))
    if following is None or any(box is None for box in marker):
        return False
    marker_box = _bbox_union_many(marker)
    if marker_box[2] - marker_box[0] > 1.5 * em or following[0] - marker_box[2] < 0.5 * em:
        return False
    note_ink = _native_lowercase_ink_height(first)
    body_ink = [height for line, _ in above if (height := _native_lowercase_ink_height(line)) is not None]
    return note_ink is not None and len(body_ink) >= 3 and note_ink <= 0.85 * statistics.median(body_ink)


def _footnote_lane_members(
    lane: _TextLane,
    rule_bbox: BBox,
    local_page_size: tuple[float, float],
    *,
    page_median_height: float | None = None,
    lane_width_reference: float | None = None,
    allow_column_width_rule: bool = False,
    lower_barrier_y: float | None = None,
) -> set[int]:
    """验证横线与单个栏带的对齐关系，并返回其下连续脚注行的来源编号。"""

    lane_lines = [item for item in lane.lines if item[0].semantic_type is None]
    if not lane_lines:
        return set()
    lane_lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    # 显式图题已经在原生行入口标注，图题装饰横线不能重新把该行认领为页面脚注。
    if any(
        line.caption_start and bbox[1] >= rule_bbox[1] and bbox[1] <= rule_bbox[3] + 3 * _line_effective_height(line, bbox)
        for line, bbox in lane_lines
    ):
        return set()
    local_page_width, local_page_height = local_page_size
    lane_width = max(
        0.1,
        lane.right - lane.left,
        lane_width_reference or 0.0,
    )
    lane_heights = [_line_effective_height(line, bbox) for line, bbox in lane_lines]
    median_height = statistics.median(lane_heights) if lane_heights else 1.0
    rule_width = max(0.0, rule_bbox[2] - rule_bbox[0])
    # 同时限制绝对短线、相对长线和左缘偏移，排除图标、公式线及跨栏正文分隔线。
    requires_short_rule_evidence = rule_width < max(4.0 * median_height, 0.04 * local_page_width)
    if requires_short_rule_evidence and rule_width < max(2.0 * median_height, 0.08 * lane_width):
        return set()
    strict_left_tolerance = max(
        2.0 * median_height,
        0.04 * lane_width,
    )
    strict_left_alignment = abs(rule_bbox[0] - lane.left) <= strict_left_tolerance
    wide_body_heights = [
        _line_effective_height(line, bounds)
        for line, bounds in lane_lines
        if bounds[3] <= rule_bbox[1] and bounds[2] - bounds[0] >= 0.35 * lane_width
    ]
    marker_body_reference = max(
        median_height, page_median_height or 0, statistics.median(wide_body_heights) if len(wide_body_heights) >= 3 else 0
    )
    # 正文缩进不能替代脚注栏左缘；线下真实编号与小字正文同行时，由编号补足左缘证据。
    numbered_left_alignment = any(
        marker.note_marker_value is not None
        and abs(bounds[0] - rule_bbox[0]) <= 0.5 * median_height
        and 0 <= bounds[1] - rule_bbox[3] <= 3 * median_height
        and bounds[2] - bounds[0] <= 1.25 * median_height
        and any(
            body_bounds[0] >= bounds[2]
            and body_bounds[0] - bounds[2] <= 2 * median_height
            and body_bounds[2] - body_bounds[0] >= 4 * median_height
            and _bbox_axis_overlap_ratio(bounds, body_bounds, axis="y") >= 0.6
            and _line_effective_height(body, body_bounds) <= 0.9 * marker_body_reference
            for body, body_bounds in lane_lines
            if body is not marker
        )
        for marker, bounds in lane_lines
    )
    strict_left_alignment = strict_left_alignment or numbered_left_alignment
    relaxed_left_alignment = rule_bbox[0] < lane.left and lane.left - rule_bbox[0] <= 2.25 * median_height
    rule_center_y = _bbox_center_y(rule_bbox)
    centered_short_alignment = (
        not lane.is_span
        and rule_center_y >= 0.7 * local_page_height
        and 0.35 * lane_width <= rule_width <= 0.7 * lane_width
        and abs(_bbox_center_x(rule_bbox) - 0.5 * (lane.left + lane.right)) <= 0.08 * lane_width
    )
    if not strict_left_alignment and not relaxed_left_alignment and not centered_short_alignment:
        return set()

    is_regular_short_rule = (
        rule_center_y >= (0.55 if requires_short_rule_evidence else 0.7) * local_page_height and rule_width <= 0.65 * lane_width
    )
    endpoint_tolerance = max(2.0 * median_height, 0.05 * lane_width)
    is_column_width_rule = (
        allow_column_width_rule
        and not lane.is_span
        and 0.65 * lane_width <= rule_width <= 1.05 * lane_width
        and abs(rule_bbox[2] - lane.right) <= endpoint_tolerance
    )
    if not is_regular_short_rule and not is_column_width_rule and not centered_short_alignment:
        return set()

    # 首行采用较宽的 3.5% 页高窗口；命中后仅按紧凑的连续净空向下扩展。
    first_gap_limit = max(3.0 * median_height, 0.035 * local_page_height)
    first_index: int | None = None
    for index, (_line, bbox) in enumerate(lane_lines):
        if min(rule_bbox[2], bbox[2]) <= max(rule_bbox[0], bbox[0]):
            continue
        if bbox[2] - bbox[0] < 2 * median_height:
            continue
        rule_gap = bbox[1] - rule_bbox[3]
        if rule_gap < -0.5 * median_height:
            continue
        if lower_barrier_y is not None and bbox[1] >= lower_barrier_y:
            break
        if rule_gap <= first_gap_limit:
            first_index = index
        break
    if first_index is None:
        return set()

    if is_regular_short_rule:
        # 与上方正文框交叠的细线属于行内装饰，不是留白中的脚注分隔线；下行字号不能改变线的归属。
        first_line, first_bbox = lane_lines[first_index]
        if any(
            bounds[1] < rule_center_y < bounds[3] and _bbox_axis_overlap_ratio(bounds, rule_bbox, axis="x") >= 0.5
            for line, bounds in lane_lines[:first_index]
        ):
            return set()

    if requires_short_rule_evidence:
        # 不直接放宽短线门槛：小字编号、栏缘及上下净空必须同时成立，排除分式和章节线。
        first_line, first_bbox = lane_lines[first_index]
        above = [(line, bbox) for line, bbox in lane_lines[:first_index] if bbox[1] < rule_bbox[1]]
        reference_height = max(
            page_median_height or 0.0, statistics.median(_line_effective_height(*item) for item in above) if above else 0.0
        )
        numbered = first_line.footnote_marker_start or any(
            bounds[2] - bounds[0] <= 1.5 * median_height
            and abs(bounds[0] - rule_bbox[0]) <= 0.5 * median_height
            and 0 <= first_bbox[0] - bounds[2] <= 1.5 * median_height
            and _bbox_axis_overlap_ratio(bounds, first_bbox, axis="y") >= 0.6
            for _line, bounds in lane_lines
        )
        # 跨页续注可能没有当前页首编号；分隔线左端与悬挂正文之间保留一字以上的编号槽。
        marker_slot = median_height <= first_bbox[0] - rule_bbox[0] <= 2.25 * median_height
        if (
            not strict_left_alignment
            or not (numbered or marker_slot)
            or reference_height <= 0
            or _line_effective_height(first_line, first_bbox) > 0.9 * reference_height
            or any(
                bbox[3] > rule_bbox[1] - 0.25 * reference_height or first_bbox[1] - bbox[3] < 0.75 * reference_height
                for _line, bbox in above
            )
        ):
            return set()

    if is_column_width_rule:
        # 页面中段的栏宽横线只有在下方首行相对上方正文明显收缩时才可触发脚注，
        # 避免把章节分隔线或普通栏内横线误当成脚注边界。
        body_heights = [
            _line_effective_height(line, bbox) for line, bbox in lane_lines if bbox[3] <= rule_bbox[1] + 0.5 * median_height
        ]
        first_height = _line_effective_height(*lane_lines[first_index])
        body_reference_height = statistics.median(body_heights) if body_heights else 0.0
        if page_median_height is not None:
            body_reference_height = max(
                body_reference_height,
                page_median_height,
            )
        first_line, first_bbox = lane_lines[first_index]
        continuation_boxes = [
            bounds
            for _, bounds in lane_lines[first_index + 1 :]
            if first_bbox[1] + 0.5 * median_height <= bounds[1] <= first_bbox[3] + 3 * median_height
        ]
        continuation_left = (
            statistics.median(bounds[0] for bounds in continuation_boxes) if continuation_boxes else first_bbox[0]
        )
        prefix_bbox = first_line.source_bbox or first_bbox
        # 同字号通栏脚注须同时有独立编号槽、悬挂续行和正文净空，普通章节分隔线不享受此例外。
        hanging_note = (
            rule_width >= 0.7 * local_page_width
            and rule_center_y >= 0.75 * local_page_height
            and (first_line.dominant_font_weight or 400) < 600
            and first_height <= 1.25 * body_reference_height
            and (
                prefix_bbox[2] - prefix_bbox[0] <= 1.5 * median_height
                and abs(prefix_bbox[0] - rule_bbox[0]) <= median_height
                and 0.5 * median_height <= continuation_left - prefix_bbox[2] <= 2 * median_height
                or any(
                    bounds[2] - bounds[0] <= 1.5 * median_height
                    and abs(bounds[0] - rule_bbox[0]) <= median_height
                    and 0 <= first_bbox[0] - bounds[2] <= 2 * median_height
                    and _bbox_axis_overlap_ratio(bounds, first_bbox, axis="y") >= 0.8
                    for _, bounds in lane_lines
                )
            )
            and sum(abs(bounds[0] - continuation_left) <= 0.3 * median_height for bounds in continuation_boxes) >= 2
            and all(
                first_bbox[1] - bounds[3] >= 0.75 * body_reference_height
                for _, bounds in lane_lines[:first_index]
                if bounds[3] <= rule_bbox[1]
            )
        )
        single_line_ink_note = _single_line_ink_note_evidence(
            first_line,
            [(line, bounds) for line, bounds in lane_lines[:first_index] if bounds[3] <= rule_bbox[1]],
            rule_bbox,
            local_page_size,
            body_reference_height,
        )
        if len(body_heights) < 3 or first_height > 0.95 * body_reference_height and not (hanging_note or single_line_ink_note):
            return set()

    continuation_gap_limit = _page_footnote_continuation_gap_limit(
        median_height,
        local_page_height,
    )
    members = [lane_lines[first_index]]
    for current in lane_lines[first_index + 1 :]:
        if _bbox_axis_overlap_ratio(members[0][1], current[1], axis="x") < 0.5:
            continue
        if lower_barrier_y is not None and current[1][1] >= lower_barrier_y:
            break
        if _effective_text_row_gap(members[-1], current) > continuation_gap_limit:
            break
        members.append(current)
    if relaxed_left_alignment and not strict_left_alignment:
        first_bbox = members[0][1]
        first_height = _line_effective_height(*members[0])
        reference_height = max(
            median_height,
            page_median_height or 0.0,
        )
        horizontal_overlap = max(
            0.0,
            min(rule_bbox[2], first_bbox[2]) - max(rule_bbox[0], first_bbox[0]),
        )
        if len(members) < 2 or first_height > 0.9 * reference_height or horizontal_overlap / max(0.1, rule_width) < 0.8:
            return set()
    if centered_short_alignment:
        reference_height = max(
            median_height,
            page_median_height or 0.0,
        )
        projecting_rows_above = []
        for _line, bbox in lane_lines[:first_index]:
            overlap = max(
                0.0,
                min(rule_bbox[2], bbox[2]) - max(rule_bbox[0], bbox[0]),
            )
            row_width = max(0.1, bbox[2] - bbox[0])
            if bbox[1] < rule_bbox[1] and overlap >= 0.2 * min(rule_width, row_width):
                projecting_rows_above.append(bbox)
        if any(bbox[3] > rule_bbox[1] - 0.75 * reference_height for bbox in projecting_rows_above):
            # 分式横线位于公式成员之间；真正的脚注分隔线上方应保留正文净空。
            return set()
        member_height = statistics.median(_line_effective_height(*member) for member in members)
        if len(members) < 2 or member_height > 0.9 * reference_height:
            return set()
    return {line.source_index for line, _bbox in members}


def _page_footnote_continuation_gap_limit(
    reference_height: float,
    local_page_height: float,
) -> float:
    """统一返回页脚注连续扩展允许的最大有效净空。"""

    return max(1.25 * reference_height, 0.01 * local_page_height)


def _footnote_lane_width_reference(
    lane: _TextLane,
    lanes: list[_TextLane],
    median_height: float,
) -> float:
    """用下一稳定栏的左缘补偿当前栏因正文右缘参差造成的宽度低估。"""

    lane_width = max(0.1, lane.right - lane.left)
    stable_lanes = sorted(
        [candidate for candidate in lanes if not candidate.is_span and len(candidate.lines) >= 3],
        key=lambda candidate: candidate.left,
    )
    if lane not in stable_lanes:
        return lane_width
    lane_index = stable_lanes.index(lane)
    if lane_index + 1 >= len(stable_lanes):
        return lane_width
    minimum_gutter = max(6.0, 0.75 * median_height)
    next_lane = stable_lanes[lane_index + 1]
    return max(
        lane_width,
        next_lane.left - lane.left - minimum_gutter,
    )


def _classify_rule_delimited_headers(pages: list[_PreparedPage]) -> None:
    """在页码完成跨页判定后，用页首长横线补标其上方未分类文本。"""

    for page in pages:
        available = [line for line in page.remaining_lines if line.semantic_type is None]
        if not available or not page.drawing_lines:
            continue
        support_by_angle = _geometric_text_support_by_angle(
            page.remaining_lines,
            page.page_size,
        )
        if not support_by_angle:
            continue
        dominant_angle = max(
            sorted(support_by_angle),
            key=lambda angle: support_by_angle[angle],
        )
        local_page_size = (page.page_size[1], page.page_size[0]) if dominant_angle in {90, 270} else page.page_size
        local_page_width, local_page_height = local_page_size
        if local_page_width <= 0 or local_page_height <= 0:
            continue
        local_lines = [
            (
                line,
                _rotate_bbox_to_upright(
                    line.ink_bbox or line.bbox,
                    page.page_size,
                    dominant_angle,
                ),
            )
            for line in available
            if line.angle == dominant_angle
        ]
        header_evidence_bboxes = [
            _rotate_bbox_to_upright(
                line.ink_bbox or line.bbox,
                page.page_size,
                dominant_angle,
            )
            for line in page.remaining_lines
            if line.angle == dominant_angle and line.semantic_type in {None, "header", "page_number"}
        ]
        heights = [_line_effective_height(line, bbox) for line, bbox in local_lines]
        median_height = statistics.median(heights) if heights else 1.0
        canonical_scales = [_line_canonical_style_scale(line, bbox) for line, bbox in local_lines]
        median_scale = statistics.median(canonical_scales) if canonical_scales else median_height
        local_axis_lines, table_lines = _prepared_local_axis_table_lines(page, dominant_angle)
        candidates = [
            axis_line
            for axis_line in local_axis_lines
            if axis_line.orientation == "horizontal"
            and _bbox_center_y(axis_line.bbox) <= 0.15 * local_page_height
            and axis_line.bbox[2] - axis_line.bbox[0] >= 0.6 * local_page_width
            and any(bbox[3] <= _bbox_center_y(axis_line.bbox) for bbox in header_evidence_bboxes)
            and not _rule_belongs_to_confirmed_table(
                axis_line,
                table_lines,
                local_page_width,
            )
            and not _rule_overlaps_fixed_container(
                axis_line,
                page.fixed_blocks,
                page.page_size,
            )
        ]
        if not candidates:
            continue
        separator = min(candidates, key=lambda item: _bbox_center_y(item.bbox))
        separator_y = _bbox_center_y(separator.bbox)
        if not any(_bbox_center_y(bbox) >= separator_y + median_height for _line, bbox in local_lines):
            continue
        for line, bbox in local_lines:
            if bbox[3] <= separator_y and not line.caption_start and not line.marginal_body_context:
                if _line_canonical_style_scale(line, bbox) >= 1.5 * median_scale:
                    # 装饰横线也会包围真正章节标题；显著大于正文的文字不能仅凭页首横线成为页眉。
                    # 使用字形校准尺度，避免上下两行包络重叠把小号日期误当作大标题。
                    if line.numbered_heading_start:
                        line.semantic_type = "paragraph_title"
                        line.structural_title = True
                        line.explicit_section_title = True
                    continue
                line.semantic_type = "header"


def _classify_rule_delimited_footers(pages: list[_PreparedPage]) -> None:
    """用页面底部横线确认双线间页脚或单线下方的小字号栏内页脚。"""

    for page in pages:
        available = [line for line in page.remaining_lines if line.semantic_type is None]
        if not available or not page.drawing_lines:
            continue
        support_by_angle = _geometric_text_support_by_angle(
            page.remaining_lines,
            page.page_size,
        )
        if not support_by_angle:
            continue
        dominant_angle = max(
            sorted(support_by_angle),
            key=lambda angle: support_by_angle[angle],
        )
        local_page_size = (page.page_size[1], page.page_size[0]) if dominant_angle in {90, 270} else page.page_size
        local_page_width, local_page_height = local_page_size
        if local_page_width <= 0 or local_page_height <= 0:
            continue
        local_axis_lines, table_lines = _prepared_local_axis_table_lines(page, dominant_angle)
        rules = [
            rule
            for rule in local_axis_lines
            if rule.orientation == "horizontal"
            and _bbox_center_y(rule.bbox) >= 0.85 * local_page_height
            and rule.bbox[2] - rule.bbox[0] >= 0.2 * local_page_width
            and not _rule_belongs_to_confirmed_table(
                rule,
                table_lines,
                local_page_width,
            )
            and not _rule_overlaps_fixed_container(
                rule,
                page.fixed_blocks,
                page.page_size,
            )
        ]
        local_lines = [
            (
                line,
                _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    dominant_angle,
                ),
            )
            for line in available
            if line.angle == dominant_angle
        ]
        if not local_lines:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in local_lines)
        lanes = [
            lane
            for lane in _infer_text_lanes(
                local_lines,
                local_page_width,
                median_height,
                recalculate_intervals=False,
            )
            if not lane.is_span
        ]
        for upper_index, upper in enumerate(rules[:-1]):
            for lower in rules[upper_index + 1 :]:
                vertical_gap = lower.bbox[1] - upper.bbox[3]
                if not 2.0 * median_height <= vertical_gap <= 6.0 * median_height:
                    continue
                if _bbox_axis_overlap_ratio(upper.bbox, lower.bbox, axis="x") < 0.9:
                    continue
                corridor_left = max(upper.bbox[0], lower.bbox[0])
                corridor_right = min(upper.bbox[2], lower.bbox[2])
                members = [
                    (line, bbox)
                    for line, bbox in local_lines
                    if bbox[1] >= upper.bbox[3]
                    and bbox[3] <= lower.bbox[1]
                    and bbox[0] >= corridor_left - 0.5 * median_height
                    and bbox[2] <= corridor_right + 0.5 * median_height
                ]
                if not 1 <= len(members) <= 3:
                    continue
                if any(
                    abs(_bbox_center_x(bbox) - 0.5 * (corridor_left + corridor_right))
                    > 0.15 * max(0.1, corridor_right - corridor_left)
                    for _line, bbox in members
                ):
                    continue
                for line, _bbox in members:
                    line.semantic_type = "footer"
                break
        for rule in rules:
            for line in _single_rule_footer_members(
                rule,
                local_lines,
                lanes,
                median_height,
            ):
                line.semantic_type = "footer"


def _single_rule_footer_members(
    rule: _LocalAxisLine,
    local_lines: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    body_height: float,
) -> list[_LineItem]:
    """返回底部单横线下方、唯一栏内连续的小字号页脚行。"""

    rule_width = max(0.1, rule.bbox[2] - rule.bbox[0])
    rule_center_x = _bbox_center_x(rule.bbox)
    matching_lanes = []
    for lane in lanes:
        overlap = max(
            0.0,
            min(rule.bbox[2], lane.right) - max(rule.bbox[0], lane.left),
        )
        if overlap / rule_width >= 0.8 and lane.left <= rule_center_x <= lane.right:
            matching_lanes.append(lane)
    if len(matching_lanes) != 1:
        return []

    lane = matching_lanes[0]
    if len(lanes) > 1 and lane.left < max(candidate_lane.left for candidate_lane in lanes) - body_height:
        return []
    tolerance = 0.5 * body_height
    rows_below = sorted(
        (
            (line, bbox)
            for line, bbox in local_lines
            if line.semantic_type is None
            and bbox[1] >= rule.bbox[3]
            and bbox[0] >= lane.left - tolerance
            and bbox[2] <= lane.right + tolerance
        ),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if not rows_below:
        return []
    first_line, first_bbox = rows_below[0]
    if first_bbox[1] - rule.bbox[3] > body_height or _line_effective_height(first_line, first_bbox) > 0.9 * body_height:
        return []

    members = [(first_line, first_bbox)]
    for line, bbox in rows_below[1:]:
        previous_bbox = members[-1][1]
        if (
            bbox[1] - previous_bbox[3] > body_height
            or bbox[3] - first_bbox[1] > 5.0 * body_height
            or _line_effective_height(line, bbox) > 0.9 * body_height
        ):
            break
        members.append((line, bbox))
    if not 2 <= len(members) <= 8:
        return []
    member_left_edges = [bbox[0] for _line, bbox in members]
    if max(member_left_edges) - min(member_left_edges) > 0.75 * body_height:
        return []
    return [line for line, _bbox in members]


def _rule_overlaps_fixed_container(
    rule: _LocalAxisLine,
    fixed_blocks: list[dict[str, object]],
    page_size: tuple[float, float],
) -> bool:
    """排除落在表格、图片、公式或代码容器内的页首横线。"""

    expanded_rule = _expand_bbox(
        rule.original_bbox,
        max(1.0, rule.width),
    )
    for block in fixed_blocks:
        if block.get("type") not in {"table", "image", "equation", "code"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page_size)
        if bbox is not None and _bbox_intersects(expanded_rule, bbox):
            return True
    return False


def _classify_page_number_outer_companions(
    pages: list[_PreparedPage],
) -> None:
    """把上下页码外侧的未分类文本和图片标为对应页眉或页脚。"""

    for page in pages:
        page_numbers = [line for line in page.remaining_lines if line.semantic_type == "page_number"]
        for page_number in page_numbers:
            angle = page_number.angle
            local_page_size = (page.page_size[1], page.page_size[0]) if angle in {90, 270} else page.page_size
            local_page_height = local_page_size[1]
            if local_page_height <= 0:
                continue
            page_number_bbox = _rotate_bbox_to_upright(
                page_number.bbox,
                page.page_size,
                angle,
            )
            normalized_center_y = _bbox_center_y(page_number_bbox) / local_page_height
            if normalized_center_y <= 0.3:
                target_type: Literal["header", "footer"] = "header"
                outward_limit = page_number_bbox[1]
            elif normalized_center_y >= 0.7:
                target_type = "footer"
                outward_limit = page_number_bbox[3]
            else:
                continue
            for line in page.remaining_lines:
                if line.semantic_type is not None or line.angle != angle:
                    continue
                local_bbox = _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    angle,
                )
                is_outward = local_bbox[3] <= outward_limit if target_type == "header" else local_bbox[1] >= outward_limit
                # 同一边缘行还要求候选自身是窄带对象；近整页的大容器只蹭到页码行边时不算同行。
                same_marginal_row = (
                    _bbox_axis_overlap_ratio(
                        local_bbox,
                        page_number_bbox,
                        axis="y",
                    )
                    >= 0.5
                    and local_bbox[3] - local_bbox[1] <= _MARGINAL_ROW_MAX_HEIGHT_RATIO * local_page_height
                )
                if is_outward or same_marginal_row:
                    line.semantic_type = target_type
            for block in page.fixed_blocks:
                if block.get("type") != "image":
                    continue
                block_angle = int(block.get("angle", 0) or 0) % 360
                if block_angle != angle:
                    continue
                bbox = _clip_bbox(
                    _coerce_bbox(block.get("bbox")),
                    page.page_size,
                )
                if bbox is None:
                    continue
                local_bbox = _rotate_bbox_to_upright(
                    bbox,
                    page.page_size,
                    angle,
                )
                is_outward = local_bbox[3] <= outward_limit if target_type == "header" else local_bbox[1] >= outward_limit
                # 同一边缘行还要求图片自身是窄带对象；近整页的大图只蹭到页码行边时不算同行。
                same_marginal_row = (
                    _bbox_axis_overlap_ratio(
                        local_bbox,
                        page_number_bbox,
                        axis="y",
                    )
                    >= 0.5
                    and local_bbox[3] - local_bbox[1] <= _MARGINAL_ROW_MAX_HEIGHT_RATIO * local_page_height
                )
                if is_outward or same_marginal_row:
                    block["type"] = target_type


def _classify_split_marginal_row_companions(
    pages: list[_PreparedPage],
) -> None:
    """把页边缘同一拆分视觉行中的未分类碎片继承为页眉或页脚。"""

    for page in pages:
        row_groups: dict[tuple[int, int], list[_LineItem]] = {}
        for line in page.remaining_lines:
            if line.visual_row_id is None or not line.split_from_row:
                continue
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)
        for (angle, _row_id), members in row_groups.items():
            local_page_height = page.page_size[0] if angle in {90, 270} else page.page_size[1]
            local_bboxes = [_rotate_bbox_to_upright(line.bbox, page.page_size, angle) for line in members]
            row_center = statistics.fmean(_bbox_center_y(bbox) for bbox in local_bboxes)
            if row_center <= 0.1 * local_page_height:
                target_type: Literal["header", "footer"] = "header"
            elif row_center >= 0.9 * local_page_height:
                target_type = "footer"
            else:
                continue
            anchor_types = {line.semantic_type for line in members if line.semantic_type in {target_type, "page_number"}}
            if not anchor_types:
                continue
            for line in members:
                if line.semantic_type is None and not line.marginal_body_context:
                    line.semantic_type = target_type


def _classify_raw_page_marginals(sources: list[_PageSource]) -> None:
    """在视觉容器认领前保护强跨页页码、页眉和页脚文本。"""

    for source in sources:
        _protect_local_body_marginals(source)
        _protect_leading_title_groups(source)
    if len(sources) < 2:
        return
    candidates = [
        candidate
        for page_index, source in enumerate(sources)
        for line in source.lines
        if (
            candidate := _build_marginal_candidate(
                page_index,
                line,
                source.page_size,
            )
        )
        is not None
        and (
            _bbox_center_y(candidate.local_bbox) / candidate.local_page_size[1] <= 0.08
            or _bbox_center_y(candidate.local_bbox) / candidate.local_page_size[1] >= 0.92
            or (
                _parse_page_number_value(line.text) is not None
                and (
                    _bbox_center_y(candidate.local_bbox) / candidate.local_page_size[1] <= 0.15
                    or _bbox_center_y(candidate.local_bbox) / candidate.local_page_size[1] >= 0.85
                )
            )
        )
    ]
    _classify_marginal_candidates(candidates)


def _protect_leading_title_groups(source: _PageSource) -> None:
    """Protect a centered document identifier adjoining a larger local heading.

    Repetition is secondary to a page-local title hierarchy. Independent header
    rules, small running text and large whitespace gaps remain header evidence.
    """
    for angle in {line.angle for line in source.lines}:
        width, height = source.page_size[::-1] if angle in {90, 270} else source.page_size
        if width <= 0 or height <= 0:
            continue
        rows = sorted(
            [(line, _rotate_bbox_to_upright(line.bbox, source.page_size, angle))
             for line in source.lines if line.angle == angle and line.text.strip()],
            key=lambda item: (item[1][1], item[1][0]),
        )
        if len(rows) < 4:
            continue
        identifier, box = rows[0]
        heading, heading_box = rows[1]
        em = _line_effective_height(identifier, box)
        heading_em = _line_effective_height(heading, heading_box)
        if (
            identifier.semantic_type is not None
            or heading.semantic_type is not None
            or _parse_page_number_value(identifier.text) is not None
            or identifier.caption_start
            or heading.caption_start
            or em <= 0
            or box[3] > .15 * height
            or not .2 * width <= box[2] - box[0] <= .85 * width
            or abs(_bbox_center_x(box) - .5 * width) > .04 * width
            or abs(_bbox_center_x(heading_box) - _bbox_center_x(box)) > 2 * em
            or heading_box[2] - heading_box[0] > .8 * (box[2] - box[0])
            or heading_box[2] - heading_box[0] < 3 * heading_em
            or heading_em < 1.1 * em
            or not .5 * em <= heading_box[1] - box[3] <= 3 * em
        ):
            continue
        body = [(line, bounds) for line, bounds in rows[2:]
                if bounds[1] >= heading_box[3]
                and bounds[1] < .85 * height
                and bounds[2] - bounds[0] >= .35 * width
                and line.semantic_type is None]
        if len(body) < 2:
            continue
        body_em = statistics.median(_line_effective_height(line, bounds) for line, bounds in body)
        if (body_em <= 0 or em < .95 * body_em
                or not 0 <= body[0][1][1] - heading_box[3] <= 4 * heading_em):
            continue
        rules = _transform_axis_lines(source.drawing_lines, source.page_size, angle)
        if any(rule.orientation == "horizontal"
               and box[3] <= _bbox_center_y(rule.bbox) <= heading_box[1]
               and rule.bbox[2] - rule.bbox[0] >= .6 * width for rule in rules):
            continue
        identifier.marginal_body_context = True
        heading.marginal_body_context = True


def _protect_local_body_marginals(source: _PageSource) -> None:
    """按短行和紧邻长行的排版连续性保护正文，不枚举标签或要求底色。"""
    width, height = source.page_size
    panels = []
    for path in source.path_infos:
        box = path.bbox
        color = path.fill_rgba
        if (not path.fill_visible or not color or color[3] < 128
                or max(color[:3]) - min(color[:3]) < 12
                or box[1] < .025 * height
                or (box[2] - box[0]) * (box[3] - box[1]) > .5 * width * height):
            continue
        panels.append(box)
    for angle in {line.angle for line in source.lines}:
        members = [(line, _rotate_bbox_to_upright(line.bbox, source.page_size, angle))
                   for line in source.lines if line.angle == angle]
        local_height = width if angle in {90, 270} else height
        for label, box in members:
            em = _line_effective_height(label, box)
            if (em <= 0 or box[2] - box[0] > 8 * em
                    or not (box[3] <= .18 * local_height or box[1] >= .82 * local_height)):
                continue
            for peer, peer_box in members:
                peer_em = _line_effective_height(peer, peer_box)
                if (peer is label or not .7 * em <= peer_em <= 1.4 * em
                        or not 0 <= peer_box[1] - box[3] <= 1.5 * em
                        or abs(peer_box[0] - box[0]) > 2 * em
                        or peer_box[2] - peer_box[0] < max(4 * em, 1.8 * (box[2] - box[0]))):
                    continue
                # 一个独立页眉/页脚与内容区之间有留白；局部标签紧接长行。
                # 额外要求更外侧存在独立文字带、同段续行或共同容器，避免仅凭一对行误保刊头。
                outer_band = any(other is not label and other is not peer
                                 and other_box[3] < box[1] - 1.5 * em
                                 for other, other_box in members) if box[3] <= .18 * local_height else False
                continuation = any(other is not label and other is not peer
                                   and 0 <= other_box[1] - peer_box[3] <= 1.5 * em
                                   and abs(other_box[0] - peer_box[0]) <= 2 * em
                                   and other_box[2] - other_box[0] >= 4 * em
                                   for other, other_box in members)
                panel = any(_bbox_overlap_in_first(label.bbox, bounds) >= .99
                            and _bbox_overlap_in_first(peer.bbox, bounds) >= .99
                            and bounds[3] - bounds[1] >= 4 * em for bounds in panels)
                if outer_band or continuation or panel:
                    label.marginal_body_context = True
                    break


def _classify_repeated_page_marginals(pages: list[_PreparedPage]) -> None:
    """仅用相邻或同奇偶页的重复证据标注页码、页眉和页脚。"""

    if len(pages) < 2:
        return
    candidates = [
        candidate
        for page_index, page in enumerate(pages)
        for line in page.remaining_lines
        if (candidate := _build_marginal_candidate(page_index, line, page.page_size)) is not None
    ]

    _classify_marginal_candidates(candidates)


def _classify_marginal_candidates(
    candidates: list[_MarginalCandidate],
) -> None:
    """复用跨页递增页码和稳定边缘文本的强证据匹配。"""

    for left_index, left in enumerate(candidates):
        left_value = _parse_page_number_value(left.line.text)
        if left_value is None:
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            right_value = _parse_page_number_value(right.line.text)
            if (
                page_delta > 0
                and right_value is not None
                and right_value - left_value == page_delta
                and _page_number_candidates_match(left, right)
            ):
                left.line.semantic_type = "page_number"
                right.line.semantic_type = "page_number"

    for left_index, left in enumerate(candidates):
        if left.line.semantic_type == "page_number" or _parse_page_number_value(left.line.text) is not None:
            continue
        for right in candidates[left_index + 1 :]:
            page_delta = right.page_index - left.page_index
            if page_delta > 2:
                break
            if (
                page_delta > 0
                and left.region != "side"
                and right.region != "side"
                and right.line.semantic_type != "page_number"
                and _parse_page_number_value(right.line.text) is None
                and _marginal_geometry_matches(left, right)
                and _marginal_text_matches(left.line.text, right.line.text)
            ):
                left.line.semantic_type = left.region
                right.line.semantic_type = right.region


def _classify_single_page_compound_headers(pages: list[_PreparedPage]) -> None:
    """以拆分同行、字号收缩和正文栏右缘共同确认单页复合页眉。"""

    if len(pages) != 1:
        return
    page = pages[0]
    page_width, page_height = page.page_size
    if page_width <= 0 or page_height <= 0:
        return

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in page.remaining_lines:
        if line.semantic_type is None and line.visual_row_id is not None and line.split_from_row:
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    for (angle, _row_id), members in row_groups.items():
        if len(members) < 2:
            continue
        local_page_width = page_height if angle in {90, 270} else page_width
        local_page_height = page_width if angle in {90, 270} else page_height
        local_members = [(line, _rotate_bbox_to_upright(line.bbox, page.page_size, angle)) for line in members]
        row_top = min(bbox[1] for _line, bbox in local_members)
        row_bottom = max(bbox[3] for _line, bbox in local_members)
        if row_top < 0 or row_bottom > 0.05 * local_page_height:
            continue

        row_left = min(bbox[0] for _line, bbox in local_members)
        row_right = max(bbox[2] for _line, bbox in local_members)
        related_body = [
            (line, local_bbox)
            for line in page.remaining_lines
            if line.semantic_type is None
            and line.angle == angle
            and line not in members
            and (
                local_bbox := _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    angle,
                )
            )[1]
            >= 0.05 * local_page_height
            and local_bbox[2] - local_bbox[0] >= 0.3 * local_page_width
            and _bbox_axis_overlap_ratio(
                (row_left, row_top, row_right, row_bottom),
                local_bbox,
                axis="x",
            )
            >= 0.2
        ]
        if len(related_body) < 3:
            continue
        body_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in related_body)
        row_height = max(_line_effective_height(line, bbox) for line, bbox in local_members)
        if row_height > 0.85 * body_height:
            continue
        body_right = statistics.median(bbox[2] for _line, bbox in related_body)
        has_right_sidecar = any(
            bbox[2] - bbox[0] <= max(4.0 * row_height, 0.12 * local_page_width)
            and abs(bbox[2] - body_right) <= max(3.0, 0.02 * local_page_width)
            for _line, bbox in local_members
        )
        if has_right_sidecar:
            for line, _bbox in local_members:
                line.semantic_type = "header"


def _classify_page_footnote_trailing_footers(
    pages: list[_PreparedPage],
) -> None:
    """把任意页脚注投影下方、已越过续行边界的紧凑尾段标为页脚。"""

    for page in pages:
        line_by_source = {line.source_index: line for line in page.remaining_lines}
        ranked_groups: list[tuple[float, set[int]]] = []
        for source_indices in page.page_footnote_groups:
            anchor_lines = [
                line_by_source[source_index]
                for source_index in source_indices
                if source_index in line_by_source and line_by_source[source_index].semantic_type == "page_footnote"
            ]
            angles = {line.angle for line in anchor_lines}
            if len(angles) != 1:
                continue
            angle = next(iter(angles))
            local_bottom = max(
                _rotate_bbox_to_upright(
                    line.bbox,
                    page.page_size,
                    angle,
                )[3]
                for line in anchor_lines
            )
            ranked_groups.append((local_bottom, source_indices))

        # 优先处理页面最下方的脚注组，避免上方脚注跨过下方脚注寻找页脚。
        for _local_bottom, source_indices in sorted(
            ranked_groups,
            key=lambda item: item[0],
            reverse=True,
        ):
            for line in _page_footnote_trailing_footer_members(
                page,
                source_indices,
            ):
                line.semantic_type = "footer"


def _page_footnote_trailing_footer_members(
    page: _PreparedPage,
    source_indices: set[int],
) -> list[_LineItem]:
    """返回脚注水平投影下方唯一、紧凑且小于正文尺度的页脚行。"""

    line_by_source = {line.source_index: line for line in page.remaining_lines}
    anchor_lines = [
        line_by_source[source_index]
        for source_index in source_indices
        if source_index in line_by_source and line_by_source[source_index].semantic_type == "page_footnote"
    ]
    angles = {line.angle for line in anchor_lines}
    if len(angles) != 1:
        return []
    angle = next(iter(angles))
    local_page_size = (page.page_size[1], page.page_size[0]) if angle in {90, 270} else page.page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return []

    anchor_geometry = [
        (
            line,
            _rotate_bbox_to_upright(
                line.bbox,
                page.page_size,
                angle,
            ),
        )
        for line in anchor_lines
    ]
    anchor_bbox = _bbox_union_many([bbox for _line, bbox in anchor_geometry])
    if anchor_bbox[3] < 0.75 * local_page_height:
        return []

    unresolved_geometry = [
        (
            line,
            _rotate_bbox_to_upright(
                line.bbox,
                page.page_size,
                angle,
            ),
        )
        for line in page.remaining_lines
        if line.semantic_type is None and line.angle == angle
    ]
    body_geometry = [
        (line, bbox)
        for line, bbox in unresolved_geometry
        if bbox[3] <= anchor_bbox[1]
        and bbox[2] - bbox[0] >= 0.3 * local_page_width
        and 0.1 * local_page_height <= _bbox_center_y(bbox) <= 0.94 * local_page_height
        and _bbox_axis_overlap_ratio(
            anchor_bbox,
            bbox,
            axis="x",
        )
        >= 0.2
    ]
    if len(body_geometry) < 3:
        return []
    body_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in body_geometry)
    if body_height <= 0:
        return []

    if any(bbox[1] <= anchor_bbox[3] < bbox[3] for _line, bbox in unresolved_geometry):
        # 另一栏正文仍跨过脚注底边时，不能把其下方局部文本猜成全页页脚。
        return []
    trailing_geometry = sorted(
        ((line, bbox) for line, bbox in unresolved_geometry if bbox[1] > anchor_bbox[3]),
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    if not 1 <= len(trailing_geometry) <= 3:
        return []

    first = trailing_geometry[0]
    candidate_bbox = _bbox_union_many([bbox for _line, bbox in trailing_geometry])
    if first[1][1] < 0.82 * local_page_height:
        return []
    projection_tolerance = 0.5 * body_height
    if (
        candidate_bbox[0] < anchor_bbox[0] - projection_tolerance
        or candidate_bbox[2] > anchor_bbox[2] + projection_tolerance
        or abs(candidate_bbox[0] - anchor_bbox[0]) > body_height
    ):
        return []
    left_edges = [bbox[0] for _line, bbox in trailing_geometry]
    if max(left_edges) - min(left_edges) > 0.75 * body_height:
        return []
    if candidate_bbox[2] - candidate_bbox[0] > 0.6 * local_page_width:
        return []
    if candidate_bbox[3] - candidate_bbox[1] > 4.0 * body_height:
        return []
    if any(_line_effective_height(line, bbox) > 0.95 * body_height for line, bbox in trailing_geometry):
        return []

    projecting_anchor_rows = [
        item
        for item in anchor_geometry
        if _bbox_axis_overlap_ratio(
            item[1],
            candidate_bbox,
            axis="x",
        )
        >= 0.2
    ]
    if not projecting_anchor_rows:
        return []
    anchor_last = max(
        projecting_anchor_rows,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    continuation_gap_limit = _page_footnote_continuation_gap_limit(
        body_height,
        local_page_height,
    )
    first_gap = _effective_text_row_gap(anchor_last, first)
    if not continuation_gap_limit < first_gap <= 3.0 * body_height:
        return []
    if any(
        _effective_text_row_gap(previous, current) > continuation_gap_limit
        for previous, current in zip(
            trailing_geometry,
            trailing_geometry[1:],
        )
    ):
        return []

    original_candidate_bbox = _bbox_union_many([line.bbox for line, _bbox in trailing_geometry])
    if any(
        container_bbox is not None and _bbox_intersects(original_candidate_bbox, container_bbox)
        for block in page.fixed_blocks
        if (
            container_bbox := _clip_bbox(
                _coerce_bbox(block.get("bbox")),
                page.page_size,
            )
        )
        is not None
    ):
        return []
    return [line for line, _bbox in trailing_geometry]


def _classify_isolated_first_page_footer(pages: list[_PreparedPage]) -> None:
    """用多页首页的极底位置、正文尺度和孤立净空补标唯一页脚。"""

    if len(pages) < 2:
        return
    page = pages[0]
    page_width, page_height = page.page_size
    if page_width <= 0 or page_height <= 0:
        return

    body_lines = [
        line
        for line in page.remaining_lines
        if line.semantic_type is None
        and line.angle == 0
        and line.bbox[2] - line.bbox[0] >= 0.3 * page_width
        and 0.1 * page_height <= _bbox_center_y(line.bbox) <= 0.94 * page_height
    ]
    if len(body_lines) < 4:
        return
    body_height = statistics.median(_line_effective_height(line, line.bbox) for line in body_lines)
    # 首页刊物编号和版权声明有明确语义，允许略高于普通孤立页脚，并恢复同行碎片。
    copyright_lines = [
        line
        for line in page.remaining_lines
        if line.semantic_type is None
        and line.angle == 0
        and line.bbox[1] >= 0.92 * page_height
        and re.search(r"(?:©|copyright).*\bpublish", line.text, re.IGNORECASE)
    ]
    if len(copyright_lines) == 1:
        anchor = copyright_lines[0]
        members = [
            line
            for line in page.remaining_lines
            if line.semantic_type is None
            and line.angle == 0
            and abs(_bbox_center_y(line.bbox) - _bbox_center_y(anchor.bbox)) <= 0.3 * body_height
        ]
        bbox = _bbox_union_many([line.bbox for line in members])
        body_above = [line.bbox[3] for line in body_lines if line.bbox[3] < bbox[1]]
        if (
            bbox[2] - bbox[0] <= 0.6 * page_width
            and body_above
            and bbox[1] - max(body_above) >= 0.5 * body_height
            and all(_line_effective_height(line, line.bbox) <= body_height for line in members)
        ):
            for line in members:
                line.semantic_type = "footer"
    body_bottom = max(line.bbox[3] for line in body_lines)
    if body_bottom < 0.7 * page_height:
        return
    container_bboxes = [bbox for block in page.fixed_blocks if (bbox := _coerce_bbox(block.get("bbox"))) is not None]
    candidates = [
        line
        for line in page.remaining_lines
        if line.semantic_type is None
        and line.angle == 0
        and line.bbox[1] >= 0.94 * page_height
        and line.bbox[2] - line.bbox[0] <= 0.6 * page_width
        and _line_effective_height(line, line.bbox) <= 0.95 * body_height
        and line.bbox[1] - body_bottom >= 1.5 * body_height
        and not any(_bbox_intersects(line.bbox, container_bbox) for container_bbox in container_bboxes)
    ]
    # 出版编号和版权文字可能被字体拆成多个 run，先按同一视觉行验证整体净空。
    if candidates:
        bbox = _bbox_union_many([line.bbox for line in candidates])
        if (
            bbox[2] - bbox[0] <= 0.6 * page_width
            and abs(_bbox_center_x(bbox) - 0.5 * page_width) <= 0.08 * page_width
            and max(_bbox_center_y(line.bbox) for line in candidates) - min(_bbox_center_y(line.bbox) for line in candidates)
            <= 0.4 * body_height
        ):
            for line in candidates:
                line.semantic_type = "footer"


def _classify_repeated_visual_headers(pages: list[_PreparedPage]) -> None:
    """仅按页首位置与跨页重复几何，把整体图片重标为视觉页眉。"""

    candidates: list[tuple[int, dict[str, object], BBox, int]] = []
    for page_index, page in enumerate(pages):
        # 首页常使用独立封面版式，不参与正文页视觉页眉聚类。
        if page_index == 0:
            continue
        page_width, page_height = page.page_size
        if page_width <= 0 or page_height <= 0:
            continue
        for block in page.fixed_blocks:
            if block.get("type") != "image":
                continue
            bbox = _clip_bbox(_coerce_bbox(block.get("bbox")), page.page_size)
            if bbox is None or bbox[3] > 0.12 * page_height:
                continue
            normalized_bbox = (
                bbox[0] / page_width,
                bbox[1] / page_height,
                bbox[2] / page_width,
                bbox[3] / page_height,
            )
            angle = int(block.get("angle", 0) or 0) % 360
            candidates.append((page_index, block, normalized_bbox, angle))

    if len(candidates) < 3:
        return

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        """查找视觉页眉候选所属几何簇的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并跨页距离和归一化几何均匹配的两个候选。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, (
        first_page,
        _first_block,
        first_bbox,
        first_angle,
    ) in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second_page, _second_block, second_bbox, second_angle = candidates[second_index]
            page_delta = second_page - first_page
            if page_delta > 2:
                break
            if page_delta > 0 and first_angle == second_angle and _visual_header_geometry_matches(first_bbox, second_bbox):
                union(first_index, second_index)

    clusters: dict[int, list[int]] = {}
    for candidate_index in range(len(candidates)):
        clusters.setdefault(find(candidate_index), []).append(candidate_index)
    for member_indices in clusters.values():
        page_indices = {candidates[index][0] for index in member_indices}
        if len(page_indices) < 3:
            continue
        for index in member_indices:
            candidates[index][1]["type"] = "header"


def _visual_header_geometry_matches(first: BBox, second: BBox) -> bool:
    """比较两个归一化页首图片的 IoU 与宽高尺度。"""

    first_width = first[2] - first[0]
    first_height = first[3] - first[1]
    second_width = second[2] - second[0]
    second_height = second[3] - second[1]
    if min(first_width, first_height, second_width, second_height) <= 0:
        return False
    if max(first_width, second_width) / min(first_width, second_width) > 1.1:
        return False
    if max(first_height, second_height) / min(first_height, second_height) > 1.1:
        return False

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    union_area = first_width * first_height + second_width * second_height - intersection
    return union_area > 0 and intersection / union_area >= 0.9


def _build_marginal_candidate(
    page_index: int,
    line: _LineItem,
    page_size: tuple[float, float],
) -> _MarginalCandidate | None:
    """把页面上下百分之十五内的常规小行转换成跨页比较候选。"""

    if line.semantic_type not in {None, "page_footnote"} or line.caption_start or line.marginal_body_context:
        return None
    # 点线和目标页码属于目录正文；大量相同的点及数字屏蔽会制造虚假的刊头相似度。
    leader_text = unicodedata.normalize("NFKC", line.text)
    if re.search(r"\w.*(?:[.·]\s*){3,}\s*(?:\d+|[ivxlcdmIVXLCDM]+)\s*$", leader_text):
        return None
    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    local_page_size = (page_size[1], page_size[0]) if line.angle in {90, 270} else page_size
    local_page_width, local_page_height = local_page_size
    if local_page_width <= 0 or local_page_height <= 0:
        return None
    normalized_center_y = _bbox_center_y(local_bbox) / local_page_height
    normalized_center_x = _bbox_center_x(local_bbox) / local_page_width
    if line.semantic_type == "page_footnote" and normalized_center_y < 0.94:
        # 只允许极底部脚注重新参加跨页强证据匹配，正文脚注继续保留原类型。
        return None
    if normalized_center_y <= 0.15:
        region: Literal["header", "footer", "side"] = "header"
    elif normalized_center_y >= 0.9:
        region = "footer"
    elif (
        normalized_center_y <= 0.18
        or normalized_center_y >= 0.82
        or (
            (normalized_center_x <= 0.15 or normalized_center_x >= 0.85)
            and (normalized_center_y <= 0.3 or normalized_center_y >= 0.7)
        )
    ):
        # 仅页码递增逻辑会消费 side；稳定文本不会被侧栏位置猜成页眉页脚。
        region = "side"
    else:
        return None
    if _line_effective_height(line, local_bbox) > 0.06 * local_page_height:
        return None
    return _MarginalCandidate(
        page_index=page_index,
        line=line,
        local_bbox=local_bbox,
        local_page_size=local_page_size,
        region=region,
    )


def _page_number_candidates_match(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """校验连续页码的同边缘几何，横竖版切换时允许边缘位置随版面改变。"""

    if _marginal_geometry_matches(first, second):
        return True
    first_landscape = first.local_page_size[0] > first.local_page_size[1]
    second_landscape = second.local_page_size[0] > second.local_page_size[1]
    if first_landscape == second_landscape or first.line.angle != second.line.angle:
        return False
    first_height = _line_effective_height(first.line, first.local_bbox) / first.local_page_size[1]
    second_height = _line_effective_height(second.line, second.local_bbox) / second.local_page_size[1]
    return (
        min(first_height, second_height) > 0
        and max(first_height, second_height)
        / min(
            first_height,
            second_height,
        )
        <= 1.5
    )


def _marginal_geometry_matches(
    first: _MarginalCandidate,
    second: _MarginalCandidate,
) -> bool:
    """比较边缘候选的方向、纵向带、字号以及同侧或镜像横向位置。"""

    if first.region != second.region or first.line.angle != second.line.angle:
        return False
    first_width, first_height = first.local_page_size
    second_width, second_height = second.local_page_size
    first_y = _bbox_center_y(first.local_bbox) / first_height
    second_y = _bbox_center_y(second.local_bbox) / second_height
    if abs(first_y - second_y) > 0.025:
        return False
    first_line_height = _line_effective_height(first.line, first.local_bbox) / first_height
    second_line_height = _line_effective_height(second.line, second.local_bbox) / second_height
    if (
        min(first_line_height, second_line_height) <= 0
        or max(first_line_height, second_line_height)
        / min(
            first_line_height,
            second_line_height,
        )
        > 1.35
    ):
        return False
    if (
        first.line.font_signature is not None
        and second.line.font_signature is not None
        and first.line.font_coverage >= 0.75
        and second.line.font_coverage >= 0.75
        and first.line.font_signature != second.line.font_signature
    ):
        return False

    first_normalized_bbox = (
        first.local_bbox[0] / first_width,
        first.local_bbox[1] / first_height,
        first.local_bbox[2] / first_width,
        first.local_bbox[3] / first_height,
    )
    second_normalized_bbox = (
        second.local_bbox[0] / second_width,
        second.local_bbox[1] / second_height,
        second.local_bbox[2] / second_width,
        second.local_bbox[3] / second_height,
    )
    same_side = (
        _bbox_axis_overlap_ratio(first_normalized_bbox, second_normalized_bbox, axis="x") >= 0.4
        or abs(_bbox_center_x(first_normalized_bbox) - _bbox_center_x(second_normalized_bbox)) <= 0.08
    )
    mirrored = abs(_bbox_center_x(first_normalized_bbox) + _bbox_center_x(second_normalized_bbox) - 1.0) <= 0.12
    return same_side or mirrored


def _parse_page_number_value(text: str) -> int | None:
    """解析整行阿拉伯、罗马或中文页码；混有稳定正文的行不作为纯页码。"""

    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = _PAGE_NUMBER_RE.fullmatch(normalized)
    if match is None:
        return None
    value = match.group("value")
    if value.isdecimal():
        return int(value)
    if re.fullmatch(r"[ivxlcdm]+", value, re.IGNORECASE):
        return _roman_number_to_int(value)
    return _chinese_page_number_to_int(value)


def _roman_number_to_int(value: str) -> int | None:
    """把页码中的规范罗马数字转换成整数，非法组合返回空。"""

    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    normalized = value.upper()
    total = 0
    previous = 0
    for char in reversed(normalized):
        current = roman_values.get(char)
        if current is None:
            return None
        total += -current if current < previous else current
        previous = max(previous, current)
    if total <= 0 or total > 4999:
        return None
    return total


def _chinese_page_number_to_int(value: str) -> int | None:
    """把常见百位以内中文页码转换成整数，供跨页递增校验使用。"""

    digits = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if all(char in digits for char in value):
        try:
            return int("".join(str(digits[char]) for char in value))
        except ValueError:
            return None
    total = 0
    current_digit = 0
    for char in value:
        if char in digits:
            current_digit = digits[char]
        elif char == "十":
            total += (current_digit or 1) * 10
            current_digit = 0
        elif char == "百":
            total += (current_digit or 1) * 100
            current_digit = 0
        else:
            return None
    return total + current_digit if total + current_digit > 0 else None


def _marginal_text_matches(first_text: str, second_text: str) -> bool:
    """在屏蔽变化数字后比较边缘稳定文本，短文本只接受完全一致。"""

    # 年份、卷页和预印本编号是参考条目尾行，不是可复用的刊头文字。
    citation_tail = r"(?:\b(?:18|19|20)\d{2}\s*[;,.:]|\barxiv\s*:\s*\d|\d+\s*:\s*\d+\s*[–—-]\s*\d+)"
    if any(re.match(citation_tail, text.strip(), re.IGNORECASE) for text in (first_text, second_text)):
        return False

    # 公式编号在数字屏蔽后都会成为同一标记，不能作为重复页脚的文本证据。
    if any(re.fullmatch(r"[（(﹙]\s*[A-Za-z]?\d+(?:[.\-]\d+)*\s*[)）﹚]", text.strip()) for text in (first_text, second_text)):
        return False
    # 层级条目号携带正文身份，不能把不同条目经数字屏蔽后当作重复刊头。
    if any(re.fullmatch(r"\d+(?:\.\d+)+", text.strip()) for text in (first_text, second_text)):
        return False
    first = _normalize_marginal_text(first_text)
    second = _normalize_marginal_text(second_text)
    if not first or not second:
        return False
    if first == second:
        return True
    if min(len(first), len(second)) < 8:
        return False
    return SequenceMatcher(a=first, b=second, autojunk=False).ratio() >= 0.92


def _normalize_marginal_text(text: str) -> str:
    """统一边缘重复文本的宽窄字符、大小写、空白和可变数字。"""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"\d+", "#", normalized)
    return re.sub(r"\s+", "", normalized).strip()
