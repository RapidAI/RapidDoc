"""编排页面文档标题、跨栏标题及误判回退。"""

from __future__ import annotations

import re
import statistics
import unicodedata

from .....schema import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many, _rotate_bbox_to_upright
from ..inline.types import PDF_FONT_ITALIC_FLAG
from ..line_layout import (
    _effective_text_row_gap,
    _font_signatures_share_family,
    _infer_text_lanes,
    _line_effective_height,
    _title_fonts_compatible,
)
from ..models import _DocumentBodyProfile, _DocumentTitleProfile, _LineItem, _TextLane
from .body_profile import _infer_lane_body_profile, _line_uses_document_regular_font, _stage_profile_context
from .common import _build_physical_title_gap_map, _line_inside_visual_container, _line_near_visual_container
from .lane_titles import _classify_paragraph_titles_in_lane


def _classify_secondary_front_matter_title(lines: list[_LineItem], page_size: tuple[float, float]) -> None:
    """独立摘要、关键词与作者行共同确认后页题名，居中多行题名保持同一标题带。"""
    width, height = page_size
    if not any(re.match(r"^\s*(?:abstract|摘要)\s*[:：]", line.text, re.I) for line in lines):
        return
    if not any(re.match(r"^\s*(?:key\s*words?|关键词)\s*[:：]", line.text, re.I) for line in lines):
        return
    candidates = sorted(
        [
            line
            for line in lines
            if line.angle == 0
            and line.semantic_type is None
            and not line.title_suppressed
            and 0.075 * height <= line.bbox[1] <= 0.18 * height
            and abs(_bbox_center_x(line.bbox) - width / 2) <= 0.04 * width
            and re.search(r"[A-Za-z]{3,}", line.text)
        ],
        key=lambda line: line.bbox[1],
    )
    if not 2 <= len(candidates) <= 4 or max(line.bbox[2] - line.bbox[0] for line in candidates) < 0.65 * width:
        return
    bottom = candidates[-1].bbox[3]
    authors = [
        line
        for line in lines
        if bottom < line.bbox[1] < bottom + 6 * _line_effective_height(candidates[-1], candidates[-1].bbox)
        and re.search(r"\b[A-Z]{2,}\s+[A-Z][a-z]+", line.text)
    ]
    if not authors or min(line.bbox[1] for line in authors) - bottom < _line_effective_height(
        candidates[-1], candidates[-1].bbox
    ):
        return
    band = min(line.source_index for line in candidates)
    for line in candidates:
        line.semantic_type = "doc_title"
        line.title_band_id = band
        line.structural_title = True


def _protect_secondary_front_matter_metadata(lines: list[_LineItem], page_size: tuple[float, float]) -> None:
    """译文题名和摘要间的带编号机构行保留作者元数据身份。"""
    abstracts = [
        line.bbox[1]
        for line in lines
        if re.match(r"^\s*(?:abstract|摘要)\s*[:：]", unicodedata.normalize("NFKC", line.text), re.I)
    ]
    for line in lines:
        # 带机构编号且位于译文题名与摘要间的英文行是作者元数据，不能用参考栏字号基线晋升。
        normalized = unicodedata.normalize("NFKC", line.text)
        if (
            line.semantic_type is None
            and re.match(r"^\s*\d{1,2}[A-Z][a-z]", normalized)
            and any(line.bbox[3] < top <= line.bbox[3] + 12 * _line_effective_height(line, line.bbox) for top in abstracts)
            and any(
                peer.bbox[3] < line.bbox[1]
                and line.bbox[1] - peer.bbox[3] <= 8 * _line_effective_height(peer, peer.bbox)
                and peer.bbox[2] - peer.bbox[0] >= 0.6 * page_size[0]
                and len(re.findall(r"[A-Za-z]", unicodedata.normalize("NFKC", peer.text))) >= 20
                for peer in lines
            )
        ):
            line.semantic_type = "text"
            line.title_suppressed = True


def _classify_page_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
    caption_container_bboxes: list[BBox] | None = None,
    document_body_profile: _DocumentBodyProfile | None = None,
    document_title_profile: _DocumentTitleProfile | None = None,
) -> None:
    """只用页面几何与字体排版标注首页文档标题和各页段落标题。"""

    if page_index > 0:
        _classify_secondary_front_matter_title(lines, page_size)
        _protect_secondary_front_matter_metadata(lines, page_size)
    for angle in sorted({line.angle for line in lines if line.semantic_type is None and not line.title_suppressed}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and (line.semantic_type is None or line.explicit_section_title) and not line.title_suppressed
        ]
        if not line_geometry:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        physical_gaps = _build_physical_title_gap_map(line_geometry)
        grid_title_suppressions = _find_repeated_grid_title_suppressions(
            lanes,
            median_height,
        )
        local_container_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        grid_title_suppressions.update(
            _find_container_visual_row_title_suppressions(
                line_geometry,
                local_container_bboxes,
                median_height,
            )
        )

        document_title_bottom: float | None = None
        if page_index == 0:
            document_title_bottom = _classify_document_title(
                lanes,
                local_page_height,
                local_page_width,
                document_body_profile=document_body_profile,
            )
            document_title_bottom = _expand_document_title_across_lanes(
                line_geometry,
                local_page_width,
                local_page_height,
                document_title_bottom,
            )
            document_title_bottom = _classify_additional_document_title_bands(
                line_geometry,
                local_page_width,
                local_page_height,
                document_title_bottom,
                document_body_profile=document_body_profile,
            )
        preserve_front_matter_boundaries = _document_title_uses_page_fallback(
            lanes,
            document_body_profile=document_body_profile,
        )

        _classify_cross_lane_centered_section_titles(
            line_geometry,
            lanes,
            local_page_width,
            local_page_height,
            local_container_bboxes,
            page_index=page_index,
            document_title_bottom=document_title_bottom,
        )
        _classify_cross_lane_emphasized_section_titles(
            line_geometry,
            lanes,
            local_page_width,
            local_page_height,
            local_container_bboxes,
            document_title_bottom=document_title_bottom,
            document_body_profile=document_body_profile,
        )

        for lane in lanes:
            profile = _infer_lane_body_profile(lane)
            _classify_paragraph_titles_in_lane(
                lane,
                profile,
                local_page_width,
                local_page_height,
                local_container_bboxes,
                document_title_bottom=document_title_bottom,
                preserve_front_matter_boundaries=preserve_front_matter_boundaries,
                physical_gaps=physical_gaps,
                grid_title_suppressions=grid_title_suppressions,
                document_body_profile=document_body_profile,
                document_title_profile=document_title_profile,
                page_index=page_index,
            )
        _expand_cross_lane_paragraph_title_neighbors(
            line_geometry,
        )
        _demote_hanging_multiline_text_titles(
            lanes,
            document_body_profile,
            page_index=page_index,
        )
        _demote_cross_lane_body_continuation_titles(
            line_geometry,
            lanes,
        )
        _demote_visual_container_caption_titles(
            line_geometry,
            [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in (caption_container_bboxes or [])],
        )
        _demote_sentence_tail_titles(
            [
                (
                    line,
                    _rotate_bbox_to_upright(
                        line.bbox,
                        page_size,
                        angle,
                    ),
                )
                for line in lines
                if line.angle == angle and line.semantic_type in {None, "paragraph_title"} and not line.title_suppressed
            ],
        )
        if document_body_profile is not None and document_body_profile.has_style_scale_repairs:
            _demote_non_structural_anomaly_titles(line_geometry)
        peer_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type in {None, "paragraph_title"} and not line.title_suppressed
        ]
        _restore_parallel_small_column_labels(peer_geometry, local_container_bboxes)


def _restore_parallel_small_column_labels(
    geometry: list[tuple[_LineItem, BBox]],
    containers: list[BBox],
) -> None:
    """三栏同排小标签与各栏较大的简介构成重复标题结构，用已确认同级标题补齐其余栏。"""
    candidates = []
    for line, bounds in geometry:
        height = _line_effective_height(line, bounds)
        if (
            line.paragraph_terminal
            or line.caption_start
            or line.font_signature is None
            or not 1 <= len(line.text.split()) <= 6
            or re.search(r"\d|[.!?：:;,]$", line.text.strip())
            or bounds[2] - bounds[0] > 9 * height
            or _line_inside_visual_container(bounds, containers)
        ):
            continue
        followers = [
            (other, bbox)
            for other, bbox in geometry
            if other is not line
            and other.semantic_type is None
            and abs(bbox[0] - bounds[0]) <= 0.25 * height
            and 0.5 * height <= bbox[1] - bounds[3] <= 2 * height
            and 1.2 * height <= _line_effective_height(other, bbox) <= 1.6 * height
            and bbox[2] - bbox[0] >= 1.3 * (bounds[2] - bounds[0])
            and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", other.text)) >= 2
        ]
        if len(followers) == 1:
            candidates.append((line, bounds))
    for line, bounds in candidates:
        height = _line_effective_height(line, bounds)
        band = [
            (other, bbox)
            for other, bbox in candidates
            if other.font_signature == line.font_signature
            and 0.9 * height <= _line_effective_height(other, bbox) <= 1.1 * height
            and abs(_bbox_center_y(bbox) - _bbox_center_y(bounds)) <= 0.15 * height
        ]
        band.sort(key=lambda item: item[1][0])
        if (
            len(band) < 3
            or not any(other.semantic_type == "paragraph_title" for other, _ in band)
            or any(after[1][0] - before[1][2] < 4 * height for before, after in zip(band, band[1:]))
        ):
            continue
        for other, _ in band:
            other.semantic_type = "paragraph_title"
            other.structural_title = other.explicit_section_title = True
            other.title_band_id = other.source_index


def _demote_non_structural_anomaly_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
) -> None:
    """在 loose 高度异常文档中只保留预先通过结构转折校验的段落标题。"""

    for line, _bbox in line_geometry:
        if line.semantic_type == "paragraph_title" and not line.structural_title:
            line.semantic_type = None


def _find_repeated_grid_title_suppressions(
    lanes: list[_TextLane],
    median_height: float,
) -> set[int]:
    """识别重复双栏信息网格中的短首行，避免把城市等记录头标成标题。"""

    candidates: list[tuple[int, int, float]] = []
    for lane_index, lane in enumerate(lanes):
        if lane.is_span:
            continue
        lane_width = max(0.1, lane.right - lane.left)
        for row_index, (line, bbox) in enumerate(lane.lines[:-1]):
            next_line, next_bbox = lane.lines[row_index + 1]
            width = bbox[2] - bbox[0]
            next_width = next_bbox[2] - next_bbox[0]
            gap = _effective_text_row_gap(
                (line, bbox),
                (next_line, next_bbox),
            )
            if (
                line.semantic_type is not None
                or next_line.semantic_type is not None
                or width > 0.35 * lane_width
                or next_width < max(0.6 * lane_width, 1.8 * width)
                or abs(next_bbox[0] - bbox[0]) > 0.75 * median_height
                or not -0.25 * median_height <= gap <= 0.75 * median_height
            ):
                continue
            candidates.append((lane_index, line.source_index, _bbox_center_y(bbox)))

    bands: list[list[tuple[int, int, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[2]):
        target = next(
            (band for band in bands if abs(candidate[2] - statistics.median(item[2] for item in band)) <= 0.5 * median_height),
            None,
        )
        if target is None:
            bands.append([candidate])
        else:
            target.append(candidate)
    paired_bands = [
        {source_index for _lane_index, source_index, _center_y in band}
        for band in bands
        if len({lane_index for lane_index, _source_index, _center_y in band}) >= 2
    ]
    if len(paired_bands) < 2:
        return set()
    return set().union(*paired_bands)


def _find_container_visual_row_title_suppressions(
    line_geometry: list[tuple[_LineItem, BBox]],
    container_bboxes: list[BBox],
    median_height: float,
) -> set[int]:
    """用完整视觉行与图表容器的邻接关系抑制拆分 caption 标题误报。"""

    visual_rows: dict[int, list[tuple[_LineItem, BBox]]] = {}
    for item in line_geometry:
        row_id = item[0].visual_row_id
        if row_id is not None:
            visual_rows.setdefault(row_id, []).append(item)
    suppressed: set[int] = set()
    for members in visual_rows.values():
        if len(members) < 2:
            continue
        row_bbox = _bbox_union_many([bbox for _line, bbox in members])
        if any(
            _bbox_axis_overlap_ratio(row_bbox, container_bbox, axis="x") >= 0.35
            and max(
                row_bbox[1] - container_bbox[3],
                container_bbox[1] - row_bbox[3],
                0.0,
            )
            <= 1.5 * median_height
            for container_bbox in container_bboxes
        ):
            suppressed.update(line.source_index for line, _bbox in members)
    return suppressed


def _classify_document_title(
    lanes: list[_TextLane],
    local_page_height: float,
    local_page_width: float,
    *,
    document_body_profile: _DocumentBodyProfile | None = None,
) -> float | None:
    """从首页上部选取显著大字号锚点，并用同版式邻行扩展多行文档标题。"""

    candidates: list[tuple[float, _TextLane, int, tuple[_LineItem, BBox]]] = []
    lane_profiles = [(lane, _infer_lane_body_profile(lane)) for lane in lanes]
    column_body_heights = [profile.body_height for lane, profile in lane_profiles if not lane.is_span]
    if document_body_profile is not None:
        document_body_height = document_body_profile.body_height
    else:
        document_body_height = (
            statistics.median(column_body_heights)
            if column_body_heights
            else min(
                (profile.body_height for _lane, profile in lane_profiles),
                default=1.0,
            )
        )
    for lane, profile in lane_profiles:
        lane_width = max(0.1, lane.right - lane.left)
        available = [item for item in lane.lines if item[0].semantic_type is None]
        for row_index, item in enumerate(available):
            line, bbox = item
            reference_height = min(profile.body_height, 1.25 * document_body_height)
            height_ratio = _line_effective_height(line, bbox) / max(0.1, reference_height)
            document_height_ratio = _line_effective_height(line, bbox) / max(
                0.1,
                document_body_height,
            )
            width_ratio = (bbox[2] - bbox[0]) / lane_width
            centered = abs(_bbox_center_x(bbox) - (lane.left + lane.right) / 2.0) <= 0.15 * lane_width
            page_centered = abs(_bbox_center_x(bbox) - 0.5 * local_page_width) <= 0.15 * local_page_width
            page_width_ratio = (bbox[2] - bbox[0]) / max(0.1, local_page_width)
            spans_columns_fallback = lane.is_span and centered and width_ratio >= 0.65 and document_height_ratio >= 1.3
            isolated_front_title = (
                page_centered
                and page_width_ratio >= 0.5
                and document_height_ratio >= 1.12
                and _bbox_center_y(bbox) < 0.25 * local_page_height
                and not any(
                    other.font_signature == line.font_signature and other_bbox[1] > bbox[1] for other, other_bbox in available
                )
                and all(
                    abs(other_bbox[1] - bbox[1]) > 1.5 * _line_effective_height(line, bbox)
                    for other, other_bbox in available
                    if other is not line
                )
            )
            if (
                _bbox_center_y(bbox) > 0.45 * local_page_height
                or (height_ratio < 1.4 and not spans_columns_fallback and not isolated_front_title)
                or width_ratio < 0.2
                or (not centered and height_ratio < 1.7)
                or (not page_centered and page_width_ratio < 0.45 and height_ratio < 1.8)
            ):
                continue
            top_preference = max(0.0, 0.45 - _bbox_center_y(bbox) / local_page_height)
            top_preference_weight = (
                4.0 if document_body_profile is not None and document_body_profile.has_style_scale_repairs else 1.0
            )
            score = (
                height_ratio
                + (0.75 if centered else 0.0)
                + (1.25 if page_centered else 0.0)
                + (0.75 if page_width_ratio >= 0.55 else 0.0)
                + top_preference_weight * top_preference
                - 0.02 * row_index
            )
            candidates.append((score, lane, row_index, item))
    if not candidates:
        return None

    _score, lane, _row_index, anchor = max(candidates, key=lambda item: item[0])
    anchor_line, anchor_bbox = anchor
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    anchor_line.semantic_type = "doc_title"
    selected = [anchor]
    ordered = lane.lines
    anchor_index = ordered.index(anchor)
    for direction in (-1, 1):
        index = anchor_index + direction
        previous_bbox = anchor_bbox
        while 0 <= index < len(ordered):
            candidate_line, candidate_bbox = ordered[index]
            if candidate_line.semantic_type is not None:
                break
            candidate_height = _line_effective_height(candidate_line, candidate_bbox)
            if (
                candidate_bbox[1] > anchor_bbox[3]
                and candidate_line.font_signature == anchor_line.font_signature
                and candidate_height < 0.9 * anchor_height
            ):
                # 题名下方的显著字号收缩建立排版边界，不能只因同字体和居中继续扩张。
                break
            if not 0.8 <= candidate_height / anchor_height <= 1.25:
                break
            if not _document_title_fonts_compatible(anchor_line, candidate_line):
                break
            vertical_gap = max(candidate_bbox[1] - previous_bbox[3], previous_bbox[1] - candidate_bbox[3], 0.0)
            lane_width = max(0.1, lane.right - lane.left)
            centered = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(anchor_bbox)) <= 0.18 * lane_width
            aligned = abs(candidate_bbox[0] - anchor_bbox[0]) <= 0.75 * anchor_height
            if vertical_gap > 1.1 * anchor_height or not (centered or aligned):
                break
            candidate_line.semantic_type = "doc_title"
            selected.append((candidate_line, candidate_bbox))
            previous_bbox = candidate_bbox
            index += direction
    return max(bbox[3] for _line, bbox in selected)


def _expand_document_title_across_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    local_page_height: float,
    document_title_bottom: float | None,
) -> float | None:
    """跨错误推断栏扩展紧邻、同字号且对齐的多行文档标题。"""

    title_items = [item for item in line_geometry if item[0].semantic_type == "doc_title"]
    if not title_items:
        return document_title_bottom
    selected_ids = {id(line) for line, _bbox in title_items}
    changed = True
    while changed:
        changed = False
        title_bbox = _bbox_union_many([bbox for _line, bbox in title_items])
        title_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in title_items)
        anchors = [line for line, _bbox in title_items]
        candidates = [
            item
            for item in line_geometry
            if id(item[0]) not in selected_ids
            and item[0].semantic_type is None
            and _bbox_center_y(item[1]) <= 0.45 * local_page_height
        ]
        candidates.sort(
            key=lambda item: min(
                abs(item[1][1] - title_bbox[3]),
                abs(title_bbox[1] - item[1][3]),
            )
        )
        for candidate_line, candidate_bbox in candidates:
            candidate_height = _line_effective_height(
                candidate_line,
                candidate_bbox,
            )
            if (
                candidate_bbox[1] > title_bbox[3]
                and any(candidate_line.font_signature == anchor.font_signature for anchor in anchors)
                and candidate_height < 0.9 * title_height
            ):
                continue
            if not 0.8 <= candidate_height / max(0.1, title_height) <= 1.25:
                continue
            if not any(_document_title_fonts_compatible(anchor, candidate_line) for anchor in anchors):
                continue
            vertical_gap = max(
                candidate_bbox[1] - title_bbox[3],
                title_bbox[1] - candidate_bbox[3],
                0.0,
            )
            centered = abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(title_bbox)) <= 0.18 * local_page_width
            aligned = abs(candidate_bbox[0] - title_bbox[0]) <= 0.75 * title_height
            title_width = max(0.1, title_bbox[2] - title_bbox[0])
            candidate_width = candidate_bbox[2] - candidate_bbox[0]
            aligned_continuation = aligned and candidate_width >= 0.25 * title_width
            centered_continuation = centered and candidate_width >= 0.45 * title_width
            if vertical_gap > 1.1 * title_height or not (aligned_continuation or centered_continuation):
                continue
            candidate_line.semantic_type = "doc_title"
            title_items.append((candidate_line, candidate_bbox))
            selected_ids.add(id(candidate_line))
            changed = True
            break
    return max(bbox[3] for _line, bbox in title_items)


def _classify_additional_document_title_bands(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    local_page_height: float,
    document_title_bottom: float | None,
    *,
    document_body_profile: _DocumentBodyProfile | None,
) -> float | None:
    """用居中译题与后续作者行结构补充首页第二文档标题带。"""

    if document_title_bottom is None or document_body_profile is None or not document_body_profile.has_style_scale_repairs:
        return document_title_bottom
    body_height = max(0.1, document_body_profile.body_height)
    ordered = sorted(
        line_geometry,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    selected_bottoms = [bbox[3] for line, bbox in ordered if line.semantic_type == "doc_title"]
    for index, (line, bbox) in enumerate(ordered):
        if line.semantic_type is not None:
            continue
        center_y = _bbox_center_y(bbox)
        width_ratio = (bbox[2] - bbox[0]) / max(0.1, local_page_width)
        if not (document_title_bottom + 4.0 * body_height <= center_y <= 0.58 * local_page_height):
            continue
        if not 0.4 <= width_ratio <= 0.8:
            continue
        if abs(_bbox_center_x(bbox) - 0.5 * local_page_width) > 0.08 * local_page_width:
            continue
        if _line_uses_document_regular_font(line, document_body_profile):
            continue
        following = [
            item for item in ordered[index + 1 :] if item[0].semantic_type is None and _bbox_center_y(item[1]) > center_y
        ]
        if not following:
            continue
        _next_line, next_bbox = following[0]
        next_width_ratio = (next_bbox[2] - next_bbox[0]) / max(0.1, local_page_width)
        next_centered = abs(_bbox_center_x(next_bbox) - 0.5 * local_page_width) <= 0.12 * local_page_width
        vertical_gap = max(0.0, next_bbox[1] - bbox[3])
        if not next_centered or next_width_ratio > 0.45 or vertical_gap > 2.0 * body_height:
            continue
        line.semantic_type = "doc_title"
        selected_bottoms.append(bbox[3])
    return max(selected_bottoms, default=document_title_bottom)


def _classify_cross_lane_centered_section_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    page_index: int,
    document_title_bottom: float | None,
) -> None:
    """用正文栏中心、上下留白和正文邻行补标被单独推成窄栏的标题。"""

    stable_lanes = [
        lane for lane in lanes if not lane.is_span and len(lane.lines) >= 5 and lane.right - lane.left >= 0.2 * local_page_width
    ]
    context = _stage_profile_context(
        stable_lanes, line_geometry, container_bboxes, scalars=(local_page_width, local_page_height, document_title_bottom)
    )
    for line, bbox in line_geometry:
        if line.semantic_type is not None:
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        candidate_lanes = [lane for lane in stable_lanes if lane.left <= _bbox_center_x(bbox) <= lane.right]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(_bbox_center_x(bbox) - 0.5 * (candidate.left + candidate.right)),
        )
        profile = context.profile(lane)
        lane_width = max(0.1, lane.right - lane.left)
        line_width = bbox[2] - bbox[0]
        line_height = context.height(line, bbox)
        if not 0.08 * lane_width <= line_width <= 0.7 * lane_width:
            continue
        if abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) > 0.05 * lane_width:
            continue
        if not 0.75 <= line_height / max(0.1, profile.body_height) <= 1.3:
            continue
        uses_regular_font = (
            profile.body_font is not None
            and line.font_signature is not None
            and line.font_coverage >= 0.75
            and _font_signatures_share_family(
                line.font_signature,
                profile.body_font,
            )
        )
        weight_emphasized = (
            profile.body_weight is not None
            and line.dominant_font_weight is not None
            and line.dominant_font_weight
            >= max(
                profile.body_weight + 100.0,
                1.15 * profile.body_weight,
            )
        )
        if page_index == 0 and line_height < 0.9 * profile.body_height and uses_regular_font and not weight_emphasized:
            continue
        if document_title_bottom is not None and bbox[1] <= document_title_bottom + profile.body_height:
            continue
        if _line_inside_visual_container(bbox, container_bboxes) or _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        ):
            continue
        body_rows = [
            item
            for item in lane.lines
            if item[0] is not line
            and item[0].semantic_type is None
            and item[1][2] - item[1][0] >= 0.5 * lane_width
            and 0.75 <= context.height(*item) / max(0.1, profile.body_height) <= 1.3
        ]
        rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < _bbox_center_y(bbox)]
        rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > _bbox_center_y(bbox)]
        if not rows_above or not rows_below:
            continue
        previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
        following_rows = sorted(
            rows_below,
            key=lambda item: _bbox_center_y(item[1]),
        )
        if len(following_rows) < 3 or any(
            _effective_text_row_gap(previous_row, current_row) > profile.regular_gap + 0.75 * profile.body_height
            for previous_row, current_row in zip(
                following_rows[:3],
                following_rows[1:3],
            )
        ):
            continue
        following = following_rows[0]
        gap_above = bbox[1] - previous[1][3]
        gap_below = following[1][1] - bbox[3]
        if (
            gap_above < 0.3 * profile.body_height
            or gap_below < -0.1 * profile.body_height
            or max(gap_above, gap_below) < profile.regular_gap + 0.2 * profile.body_height
        ):
            continue
        line.semantic_type = "paragraph_title"
        context.invalidate(line)


def _demote_cross_lane_body_continuation_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
) -> None:
    """把紧接上一正文行、同字号同字体的短续行从标题降回正文。"""

    stable_lanes = [lane for lane in lanes if not lane.is_span and len(lane.lines) >= 4]
    context = _stage_profile_context(stable_lanes, line_geometry)
    for line, bbox in line_geometry:
        if line.semantic_type != "paragraph_title":
            continue
        line_height = context.height(line, bbox)
        global_preceding = [
            item
            for item in line_geometry
            if item[0] is not line
            and item[0].semantic_type is None
            and item[0].angle == line.angle
            and _bbox_center_y(item[1]) < _bbox_center_y(bbox)
            and abs(item[1][0] - bbox[0]) <= line_height
        ]
        if global_preceding:
            previous_line, previous_bbox = max(
                global_preceding,
                key=lambda item: _bbox_center_y(item[1]),
            )
            previous_height = _line_effective_height(
                previous_line,
                previous_bbox,
            )
            same_family = (
                previous_line.font_signature is None
                or line.font_signature is None
                or _font_signatures_share_family(
                    previous_line.font_signature,
                    line.font_signature,
                )
            )
            if (
                same_family
                and 0.9 <= line_height / max(0.1, previous_height) <= 1.1
                and bbox[1] - previous_bbox[3] <= 0.2 * max(line_height, previous_height)
                and abs(bbox[0] - previous_bbox[0]) <= 0.75 * max(line_height, previous_height)
                and previous_bbox[2] - previous_bbox[0] >= 2.0 * (bbox[2] - bbox[0])
            ):
                line.semantic_type = None
                context.invalidate(line)
                continue
        candidate_lanes = [lane for lane in stable_lanes if lane.left <= _bbox_center_x(bbox) <= lane.right]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(_bbox_center_x(bbox) - 0.5 * (candidate.left + candidate.right)),
        )
        lane_width = max(0.1, lane.right - lane.left)
        lane_profile = context.profile(lane)
        if bbox[2] - bbox[0] > 0.55 * lane_width:
            continue
        preceding = [
            item
            for item in line_geometry
            if item[0] is not line
            and item[0].semantic_type is None
            and _bbox_center_y(item[1]) < _bbox_center_y(bbox)
            and lane.left <= _bbox_center_x(item[1]) <= lane.right
        ]
        if not preceding:
            continue
        previous_line, previous_bbox = max(
            preceding,
            key=lambda item: _bbox_center_y(item[1]),
        )
        previous_height = _line_effective_height(
            previous_line,
            previous_bbox,
        )
        line_height = context.height(line, bbox)
        if not 0.9 <= line_height / max(0.1, previous_height) <= 1.1:
            continue
        if (
            previous_line.font_signature is not None
            and line.font_signature is not None
            and not _font_signatures_share_family(
                previous_line.font_signature,
                line.font_signature,
            )
        ):
            continue
        if previous_bbox[2] - previous_bbox[0] < 0.5 * lane_width:
            continue
        maximum_gap = 0.15 * max(line_height, previous_height)
        if (
            re.search(
                r"[.!?。！？][\]\)}）】》”’'\"]*$",
                line.text.rstrip(),
            )
            is not None
        ):
            maximum_gap = max(
                maximum_gap,
                lane_profile.regular_gap + 0.35 * lane_profile.body_height,
            )
        if bbox[1] - previous_bbox[3] > maximum_gap:
            continue
        if abs(bbox[0] - lane.left) > max(1.0, 0.75 * line_height):
            continue
        line.semantic_type = None
        context.invalidate(line)


def _classify_cross_lane_emphasized_section_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    lanes: list[_TextLane],
    local_page_width: float,
    local_page_height: float,
    container_bboxes: list[BBox],
    *,
    document_title_bottom: float | None,
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """用正文栏左缘、强调字体和段间留白补标跨栏推断失败的小节标题。"""

    stable_lanes = [
        lane for lane in lanes if not lane.is_span and len(lane.lines) >= 5 and lane.right - lane.left >= 0.2 * local_page_width
    ]
    context = _stage_profile_context(
        stable_lanes,
        line_geometry,
        container_bboxes,
        document_body_profile,
        (local_page_width, local_page_height, document_title_bottom),
    )
    for line, bbox in line_geometry:
        if line.semantic_type is not None or line.font_signature is None:
            continue
        if not line.font_signature[1] & PDF_FONT_ITALIC_FLAG:
            continue
        if _line_uses_document_regular_font(line, document_body_profile):
            continue
        if not 0.07 * local_page_height <= _bbox_center_y(bbox) <= 0.93 * local_page_height:
            continue
        candidate_lanes = [lane for lane in stable_lanes if lane.left <= _bbox_center_x(bbox) <= lane.right]
        if not candidate_lanes:
            continue
        lane = min(
            candidate_lanes,
            key=lambda candidate: abs(bbox[0] - candidate.left),
        )
        profile = context.profile(lane)
        if profile.body_font is None or line.font_signature == profile.body_font or line.font_coverage < 0.75:
            continue
        lane_width = max(0.1, lane.right - lane.left)
        line_height = context.height(line, bbox)
        if bbox[2] - bbox[0] > 0.9 * lane_width:
            continue
        if abs(bbox[0] - lane.left) > 2.0 * profile.body_height:
            continue
        if not 0.8 <= line_height / max(0.1, profile.body_height) <= 1.35:
            continue
        if document_title_bottom is not None and bbox[1] <= document_title_bottom + 2.0 * profile.body_height:
            continue
        if _line_inside_visual_container(bbox, container_bboxes) or _line_near_visual_container(
            bbox,
            container_bboxes,
            profile.body_height,
        ):
            continue
        body_rows = [
            item
            for item in lane.lines
            if item[0] is not line
            and item[0].semantic_type is None
            and item[0].font_signature == profile.body_font
            and item[1][2] - item[1][0] >= 0.45 * lane_width
        ]
        rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < _bbox_center_y(bbox)]
        rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > _bbox_center_y(bbox)]
        if not rows_above or not rows_below:
            continue
        previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
        following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
        gap_above = bbox[1] - previous[1][3]
        gap_below = following[1][1] - bbox[3]
        if (
            gap_above < 0.35 * profile.body_height
            or gap_below > 3.0 * profile.body_height
            or gap_below < -0.1 * profile.body_height
        ):
            continue
        line.semantic_type = "paragraph_title"
        context.invalidate(line)


def _is_wide_leading_title_continuation(
    title_bbox: BBox,
    candidate_bbox: BBox,
    title_height: float,
    candidate_height: float,
) -> bool:
    """识别紧贴在窄标题锚点上方、同中心的较宽首行。"""

    pair_height = max(0.1, title_height, candidate_height)
    title_width = max(0.1, title_bbox[2] - title_bbox[0])
    width_ratio = (candidate_bbox[2] - candidate_bbox[0]) / title_width
    vertical_gap = title_bbox[1] - candidate_bbox[3]
    return (
        2.0 < width_ratio <= 2.5
        and _bbox_center_y(candidate_bbox) < _bbox_center_y(title_bbox)
        and -0.1 * pair_height <= vertical_gap <= 0.15 * pair_height
        and _bbox_axis_overlap_ratio(title_bbox, candidate_bbox, axis="x") >= 0.8
        and abs(_bbox_center_x(candidate_bbox) - _bbox_center_x(title_bbox)) <= 0.5 * pair_height
    )


def _expand_cross_lane_paragraph_title_neighbors(
    line_geometry: list[tuple[_LineItem, BBox]],
) -> None:
    """把紧贴标题锚点的同字体相邻行跨栏补标为同一标题。"""

    changed = True
    while changed:
        changed = False
        title_items = [
            item for item in line_geometry if item[0].semantic_type == "paragraph_title" and not item[0].explicit_section_title
        ]
        for title_line, title_bbox in title_items:
            title_height = _line_effective_height(title_line, title_bbox)
            for candidate_line, candidate_bbox in line_geometry:
                if (
                    candidate_line.semantic_type is not None
                    or candidate_line.title_suppressed
                    or candidate_line.paragraph_group is not None
                ):
                    continue
                candidate_height = _line_effective_height(
                    candidate_line,
                    candidate_bbox,
                )
                if not 0.85 <= candidate_height / max(0.1, title_height) <= 1.2:
                    continue
                title_width = max(0.1, title_bbox[2] - title_bbox[0])
                candidate_width = candidate_bbox[2] - candidate_bbox[0]
                if not (
                    0.25 <= candidate_width / title_width <= 2.0
                    or _is_wide_leading_title_continuation(
                        title_bbox,
                        candidate_bbox,
                        title_height,
                        candidate_height,
                    )
                ):
                    continue
                if (
                    title_line.font_signature is not None
                    and candidate_line.font_signature is not None
                    and title_line.font_signature != candidate_line.font_signature
                ):
                    continue
                vertical_gap = max(
                    candidate_bbox[1] - title_bbox[3],
                    title_bbox[1] - candidate_bbox[3],
                    0.0,
                )
                if vertical_gap > 0.35 * max(title_height, candidate_height):
                    continue
                # 相同正文字体的紧邻段落不能无限递归扩展成标题；段尾上标同样构成停止证据。
                if title_line.paragraph_terminal or candidate_line.paragraph_terminal:
                    continue
                if (
                    _bbox_axis_overlap_ratio(title_bbox, candidate_bbox, axis="x") < 0.2
                    and abs(candidate_bbox[0] - title_bbox[0]) > 2.0 * title_height
                ):
                    continue
                candidate_line.semantic_type = "paragraph_title"
                changed = True
                break
            if changed:
                break


def _demote_hanging_multiline_text_titles(
    lanes: list[_TextLane],
    document_body_profile: _DocumentBodyProfile | None,
    *,
    page_index: int,
) -> None:
    """把缩进满行后回到栏左缘的紧邻标题行组降回正文。"""

    if page_index != 0 or document_body_profile is None:
        return
    body_height = max(0.1, document_body_profile.body_height)
    for lane in lanes:
        rows = sorted(
            lane.lines,
            key=lambda item: (
                item[1][1],
                item[1][0],
                item[0].source_index,
            ),
        )
        lane_width = max(0.1, lane.right - lane.left)
        for first, second in zip(rows, rows[1:]):
            first_line, first_bbox = first
            second_line, second_bbox = second
            if (
                first_line.semantic_type != "paragraph_title"
                or second_line.semantic_type != "paragraph_title"
                or first_line.font_signature is None
                or second_line.font_signature is None
                or first_line.font_coverage < 0.75
                or second_line.font_coverage < 0.75
                or not _font_signatures_share_family(
                    first_line.font_signature,
                    second_line.font_signature,
                )
            ):
                continue
            first_height = _line_effective_height(*first)
            second_height = _line_effective_height(*second)
            pair_height = max(first_height, second_height)
            if not (
                max(first_height, second_height) <= 1.1 * body_height
                and min(first_height, second_height) >= 0.85 * pair_height
                and first_bbox[2] - first_bbox[0] >= 0.9 * lane_width
                and first_bbox[0] - lane.left >= 0.75 * pair_height
                and abs(second_bbox[0] - lane.left) <= 0.5 * pair_height
                and -0.25 * pair_height <= _effective_text_row_gap(first, second) <= 0.5 * pair_height
            ):
                continue
            first_line.semantic_type = None
            second_line.semantic_type = None
            first_line.title_suppressed = True
            second_line.title_suppressed = True


def _demote_visual_container_caption_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
    container_bboxes: list[BBox],
) -> None:
    """把紧贴视觉容器下缘且水平居中的标题候选降回普通图注文本。"""

    for line, bbox in line_geometry:
        if line.semantic_type != "paragraph_title":
            continue
        line_height = _line_effective_height(line, bbox)
        for container_bbox in container_bboxes:
            vertical_gap = bbox[1] - container_bbox[3]
            if not -0.25 * line_height <= vertical_gap <= 1.5 * line_height:
                continue
            if _bbox_axis_overlap_ratio(bbox, container_bbox, axis="x") < 0.35:
                continue
            if abs(_bbox_center_x(bbox) - _bbox_center_x(container_bbox)) > 0.35 * max(
                container_bbox[2] - container_bbox[0],
                bbox[2] - bbox[0],
            ):
                continue
            line.semantic_type = None
            break


def _demote_sentence_tail_titles(
    line_geometry: list[tuple[_LineItem, BBox]],
) -> None:
    """把紧接未完正文、以句末标点结束的短行标题降回正文。"""

    for line, bbox in line_geometry:
        if (
            line.semantic_type != "paragraph_title"
            or re.search(
                r"[.!?。！？][\]\)}）】》”’'\"]*$",
                line.text.rstrip(),
            )
            is None
        ):
            continue
        line_height = _line_effective_height(line, bbox)
        preceding = [
            item
            for item in line_geometry
            if item[0] is not line
            and item[0].semantic_type is None
            and item[1][3] <= bbox[1]
            and abs(item[1][0] - bbox[0]) <= 0.75 * line_height
        ]
        if not preceding:
            continue
        previous_line, previous_bbox = max(
            preceding,
            key=lambda item: item[1][3],
        )
        previous_height = _line_effective_height(
            previous_line,
            previous_bbox,
        )
        same_family = (
            previous_line.font_signature is None
            or line.font_signature is None
            or _font_signatures_share_family(
                previous_line.font_signature,
                line.font_signature,
            )
        )
        bbox_height = max(0.1, bbox[3] - bbox[1])
        previous_bbox_height = max(
            0.1,
            previous_bbox[3] - previous_bbox[1],
        )
        compatible_height = (same_family and 0.75 <= line_height / max(0.1, previous_height) <= 1.25) or (
            0.8 <= bbox_height / previous_bbox_height <= 1.2
        )
        if (
            compatible_height
            and bbox[1] - previous_bbox[3]
            <= 0.9
            * max(
                line_height,
                previous_height,
                bbox_height,
                previous_bbox_height,
            )
            and previous_bbox[2] - previous_bbox[0] >= 1.5 * (bbox[2] - bbox[0])
            and re.search(
                r"[.!?。！？][\]\)}）】》”’'\"]*$",
                previous_line.text.rstrip(),
            )
            is None
        ):
            line.semantic_type = None


def _document_title_fonts_compatible(
    first: _LineItem,
    second: _LineItem,
) -> bool:
    """允许混排标题因主字体覆盖不足而切换字体，同时保留可靠字重屏障。"""

    if _title_fonts_compatible(first, second):
        return True
    uncertain_dominant_font = min(first.font_coverage, second.font_coverage) < 0.85
    weights_compatible = (
        first.dominant_font_weight is None
        or second.dominant_font_weight is None
        or abs(first.dominant_font_weight - second.dominant_font_weight) < 100.0
        or max(first.dominant_font_weight, second.dominant_font_weight)
        < 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )
    return uncertain_dominant_font and weights_compatible


def _document_title_uses_page_fallback(
    lanes: list[_TextLane],
    *,
    document_body_profile: _DocumentBodyProfile | None = None,
) -> bool:
    """判断首页标题是否依赖跨栏 1.30 倍全文正文行高兜底。"""

    title_heights = [
        _line_effective_height(line, bbox) for lane in lanes for line, bbox in lane.lines if line.semantic_type == "doc_title"
    ]
    column_body_heights = [_infer_lane_body_profile(lane).body_height for lane in lanes if not lane.is_span]
    if not title_heights or not column_body_heights:
        return False
    document_body_height = (
        document_body_profile.body_height if document_body_profile is not None else statistics.median(column_body_heights)
    )
    return any(1.3 <= title_height / max(0.1, document_body_height) < 1.4 for title_height in title_heights)


__all__ = [
    "_classify_page_titles",
    "_demote_non_structural_anomaly_titles",
    "_find_repeated_grid_title_suppressions",
    "_find_container_visual_row_title_suppressions",
    "_classify_document_title",
    "_expand_document_title_across_lanes",
    "_classify_additional_document_title_bands",
    "_classify_cross_lane_centered_section_titles",
    "_demote_cross_lane_body_continuation_titles",
    "_classify_cross_lane_emphasized_section_titles",
    "_is_wide_leading_title_continuation",
    "_expand_cross_lane_paragraph_title_neighbors",
    "_demote_hanging_multiline_text_titles",
    "_demote_visual_container_caption_titles",
    "_demote_sentence_tail_titles",
    "_document_title_fonts_compatible",
    "_document_title_uses_page_fallback",
]
