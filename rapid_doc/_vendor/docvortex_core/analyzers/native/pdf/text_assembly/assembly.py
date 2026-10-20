"""保持既有顺序编排正文行分组与块级组装。"""

from __future__ import annotations

import statistics
import re
from typing import Any, Sequence

from .....foundation._text import is_hyphen_at_line_end
from .....schema import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_union_many, _rotate_bbox_to_upright, _transform_axis_lines
from ..line_layout import (
    _title_fonts_compatible,
    _estimate_lane_gap,
    _infer_text_lanes,
    _horizontal_rule_separates_rows,
    _line_effective_height,
    _should_connect_semantic_rows,
    _should_connect_text_rows,
)
from ..models import _AxisLine, _LineItem
from ..native_text import _normalize_native_run_text
from .common import _FIGURE_CAPTION_MARKER_RE, _PARAGRAPH_FORMULA_CONTEXT_MARKER, _merge_text_line_content
from .footnotes import _build_grouped_page_footnote_blocks
from .merging import (
    _merge_inline_math_fragment_text_blocks,
    _merge_list_intro_text_components,
    _merge_overlapping_same_line_text_blocks,
    _merge_paragraph_formula_context_blocks,
    _merge_short_same_baseline_prefix_blocks,
    _merge_spatial_text_components,
    _merge_unterminated_text_components,
)
from .rows import (
    _repeated_bullet_break_sources,
    _italic_quote_to_body_break_sources,
    _top_marginal_text_break_sources,
    _caption_to_body_break_sources,
    _build_hanging_indent_group_map,
    _centered_visual_reset_break_sources,
    _component_starts_with_emphasized_row,
    _explicit_text_break_sources,
    _formula_style_text_row_break_sources,
    _front_matter_keyword_break_sources,
    _infer_local_text_lane_map,
    _isolated_indented_paragraph_break_sources,
    _prose_paragraph_break_sources,
    _cjk_prose_break_sources,
    _cjk_entry_break_sources,
    _leading_typography_reset_break_sources,
    _local_tight_output_line_bboxes,
    _starts_structural_reference_entry,
    _structured_text_break_sources,
)


def _build_text_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    drawing_lines: list[_AxisLine] | None = None,
    *,
    page_footnote_groups: Sequence[set[int]] | None = None,
    page_index: int | None = None,
    visual_bboxes: Sequence[BBox] | None = None,
) -> list[dict[str, Any]]:
    """先构建分组脚注，再按类型屏障、栏带和自然段边界聚合其余文本。"""

    blocks, grouped_footnote_indices = _build_grouped_page_footnote_blocks(
        lines,
        page_footnote_groups or [],
        page_size,
    )
    lines = [line for line in lines if line.source_index not in grouped_footnote_indices]
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        line_geometry.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_visual_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in (visual_bboxes or [])]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        bullet_break_sources = _repeated_bullet_break_sources(line_geometry)
        local_axis_lines = _transform_axis_lines(drawing_lines or [], page_size, angle)
        split_row_counts: dict[int, int] = {}
        for line, _bbox in line_geometry:
            if line.visual_row_id is not None and line.split_from_row:
                split_row_counts[line.visual_row_id] = split_row_counts.get(line.visual_row_id, 0) + 1

        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            if not lane.lines:
                continue
            regular_gap, gap_mad = _estimate_lane_gap(lane)
            local_lane_by_source = _infer_local_text_lane_map(lane)
            structured_break_sources = _structured_text_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            isolated_break_sources = _isolated_indented_paragraph_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            structured_break_sources.update(
                isolated_break_sources,
            )
            visual_reset_sources = _centered_visual_reset_break_sources(
                lane,
                local_visual_bboxes,
                local_page_height,
            )
            typography_reset_sources = _leading_typography_reset_break_sources(
                lane,
                regular_gap,
                gap_mad,
            )
            formula_text_break_sources = _formula_style_text_row_break_sources(
                lane,
            )
            structured_break_sources.update(visual_reset_sources)
            structured_break_sources.update(typography_reset_sources)
            structured_break_sources.update(formula_text_break_sources)
            protected_break_sources: set[int] = set()
            protected_break_sources.update(bullet_break_sources)
            protected_break_sources.update(visual_reset_sources)
            protected_break_sources.update(typography_reset_sources)
            protected_break_sources.update(formula_text_break_sources)
            protected_break_sources.update(_caption_to_body_break_sources(lane))
            protected_break_sources.update(_italic_quote_to_body_break_sources(lane))
            protected_break_sources.update(_top_marginal_text_break_sources(lane, local_page_height))
            protected_break_sources.update(
                _front_matter_keyword_break_sources(
                    lane,
                    local_page_height,
                    page_index,
                )
            )
            explicit_break_sources = _explicit_text_break_sources(lane)
            explicit_break_sources.update(_prose_paragraph_break_sources(lane, regular_gap, gap_mad))
            explicit_break_sources.update(_cjk_prose_break_sources(lane, regular_gap, gap_mad))
            explicit_break_sources.update(_cjk_entry_break_sources(lane))
            protected_break_sources.update(explicit_break_sources)
            structured_break_sources.update(
                protected_break_sources,
            )
            # 实际分隔线形成永久段落边界，块级续行补合并也必须遵守。
            rule_break_sources = {
                current[0].source_index
                for previous, current in zip(lane.lines, lane.lines[1:])
                if previous[0].semantic_type == current[0].semantic_type
                and _horizontal_rule_separates_rows(
                    _rotate_bbox_to_upright(previous[0].ink_bbox, page_size, angle) if previous[0].ink_bbox else previous[1],
                    _rotate_bbox_to_upright(current[0].ink_bbox, page_size, angle) if current[0].ink_bbox else current[1],
                    local_lane_by_source.get(current[0].source_index) or lane,
                    local_axis_lines,
                )
            }
            structured_break_sources.update(rule_break_sources)
            protected_break_sources.update(rule_break_sources)
            hanging_indent_groups = _build_hanging_indent_group_map(
                lane,
                table_bboxes,
                local_axis_lines,
            )
            component: list[tuple[_LineItem, BBox]] = [lane.lines[0]]
            components: list[list[tuple[_LineItem, BBox]]] = []
            for previous, current in zip(lane.lines, lane.lines[1:]):
                previous_type = previous[0].semantic_type
                current_type = current[0].semantic_type
                if previous[0].paragraph_group is not None or current[0].paragraph_group is not None:
                    should_connect = previous[0].paragraph_group == current[0].paragraph_group
                elif current[0].source_index in rule_break_sources or previous_type != current_type:
                    should_connect = False
                elif previous_type is not None:
                    should_connect = _should_connect_semantic_rows(
                        previous,
                        current,
                        lane,
                        regular_gap,
                        table_bboxes,
                        local_axis_lines,
                    )
                else:
                    previous_group = hanging_indent_groups.get(previous[0].source_index)
                    current_group = hanging_indent_groups.get(current[0].source_index)
                    previous_local_lane = local_lane_by_source.get(previous[0].source_index)
                    current_local_lane = local_lane_by_source.get(current[0].source_index)
                    connection_lane = (
                        current_local_lane
                        if current_local_lane is not None and previous_local_lane is current_local_lane
                        else lane
                    )
                    if (
                        current[0].style_scale_repaired
                        and current[0].split_from_row
                        and current[0].visual_row_id is not None
                        and split_row_counts.get(
                            current[0].visual_row_id,
                            0,
                        )
                        >= 2
                    ):
                        should_connect = False
                    elif current[0].source_index in structured_break_sources:
                        should_connect = False
                    elif _starts_structural_reference_entry(previous, current):
                        # 编号只确认已经由悬挂缩进几何形成的新条目，不能单独扩张范围。
                        should_connect = False
                    elif is_hyphen_at_line_end(previous[0].text):
                        # 断词续行优先于悬挂缩进分组，但仍复用正文连接中的距离和障碍限制。
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            connection_lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                    elif previous_group is not None or current_group is not None:
                        should_connect = previous_group is not None and previous_group == current_group
                    else:
                        should_connect = _should_connect_text_rows(
                            previous,
                            current,
                            connection_lane,
                            regular_gap,
                            gap_mad,
                            table_bboxes,
                            local_axis_lines,
                        )
                if should_connect:
                    component.append(current)
                else:
                    components.append(component)
                    component = [current]
            components.append(component)

            for component_geometry in components:
                component_lines = [item[0] for item in component_geometry]
                component_local_lane = local_lane_by_source.get(component_lines[0].source_index)
                if component_local_lane is None or not all(
                    local_lane_by_source.get(line.source_index) is component_local_lane for line in component_lines
                ):
                    component_local_lane = lane
                if component_lines[0].semantic_type == "doc_title":
                    # 文档标题保留自然换行，避免混排标题因语言检测在中文折行处插入空格。
                    content = "\n".join(
                        normalized for line in component_lines if (normalized := _normalize_native_run_text(line.text))
                    )
                else:
                    content = _merge_text_line_content([line.text for line in component_lines])
                if not content:
                    continue
                visual_row_ids = {line.visual_row_id for line in component_lines if line.visual_row_id is not None}
                single_run_row_id = (
                    component_lines[0].visual_row_id
                    if len(component_lines) == 1
                    and component_lines[0].split_from_row
                    and component_lines[0].visual_row_id is not None
                    else None
                )
                local_output_line_bboxes, output_bbox_repaired = _local_tight_output_line_bboxes(
                    component_lines,
                    page_size,
                    angle,
                )
                blocks.append(
                    {
                        "type": component_lines[0].semantic_type or "text",
                        "bbox": _bbox_union_many([line.bbox for line in component_lines]),
                        "angle": angle,
                        "content": content,
                        "_text_lines": component_lines,
                        "_reference_group": component_lines[0].paragraph_group,
                        "_visual_row_ids": visual_row_ids,
                        "_single_run_row_id": single_run_row_id,
                        "_local_line_bboxes": [bbox for _line, bbox in component_geometry],
                        "_local_output_line_bboxes": local_output_line_bboxes,
                        "_output_bbox_repaired": output_bbox_repaired,
                        "_line_heights": [_line_effective_height(line, bbox) for line, bbox in component_geometry],
                        "_font_signatures": {
                            line.font_signature
                            for line in component_lines
                            if line.font_signature is not None and line.font_coverage >= 0.5
                        },
                        "_inline_math_regions": [region for line in component_lines for region in line.inline_math_regions],
                        _PARAGRAPH_FORMULA_CONTEXT_MARKER: any(line.paragraph_formula_context for line in component_lines),
                        "_lane_interval": (
                            component_local_lane.left,
                            component_local_lane.right,
                        ),
                        "_lane_is_span": component_local_lane.is_span,
                        "_hard_break_before": (component_lines[0].source_index in structured_break_sources),
                        "_protected_hard_break_before": (component_lines[0].source_index in protected_break_sources),
                        "_explicit_break_before": component_lines[0].source_index in explicit_break_sources,
                        "_rule_break_before": component_lines[0].source_index in rule_break_sources,
                        "_hanging_indent_group": component_lines[0].paragraph_group
                        or hanging_indent_groups.get(
                            component_lines[0].source_index,
                        ),
                        "_leading_emphasis_start": _component_starts_with_emphasized_row(
                            component_lines,
                        ),
                    }
                )
    # 编号机构和参考条目已有明确成员关系，交由成员归组统一合并，避免通用续行跨组拼接。
    grouped_blocks = [block for block in blocks if block.get("_reference_group") is not None]
    blocks = [block for block in blocks if block.get("_reference_group") is None]
    blocks = _merge_short_same_baseline_prefix_blocks(
        blocks,
        page_size,
    )
    blocks = _merge_spatial_text_components(blocks, page_size)
    blocks = _merge_list_intro_text_components(blocks)
    blocks = _merge_unterminated_text_components(blocks)
    blocks = _merge_overlapping_same_line_text_blocks(blocks, page_size)
    blocks = _merge_inline_math_fragment_text_blocks(
        blocks,
        page_size,
    )
    return grouped_blocks + _merge_paragraph_formula_context_blocks(
        blocks,
        page_size,
    )


def _restore_caption_wrap_text(
    blocks: list[dict[str, Any]], image_bboxes: list[BBox], page_size: tuple[float, float]
) -> list[dict[str, Any]]:
    """按图注屏障分割连续窄宽行区间，镜像布局共用同一几何判定，几何断点不代表自然段。"""
    output = list(blocks)
    captions = [block for block in blocks if _FIGURE_CAPTION_MARKER_RE.match(str(block.get("content", "")))]
    for caption in captions:
        cb = caption["bbox"]
        for block in list(output):
            lines = sorted(block.get("_text_lines", []), key=lambda line: (line.bbox[1], line.bbox[0]))
            if block.get("type") != "text" or block is caption or len(lines) < 3:
                continue
            if any(line.angle != 0 or line.paragraph_group is not None or line.semantic_type is not None for line in lines):
                continue
            em = statistics.median(_line_effective_height(line, line.bbox) for line in lines)
            if not any(
                0 <= cb[1] - image[3] <= 3 * em and _bbox_axis_overlap_ratio(cb, image, axis="x") >= 0.75
                for image in image_bboxes
            ):
                continue
            right_side = lines[0].bbox[0] >= cb[2] - 0.25 * em

            def local(bbox):
                """将右侧正文镜像到左侧，确保左右布局使用完全相同的阈值。"""
                return (page_size[0] - bbox[2], bbox[1], page_size[0] - bbox[0], bbox[3]) if right_side else bbox

            obstacle = local(cb)
            split = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if line.bbox[1] >= cb[3] - 0.25 * em and local(line.bbox)[2] > obstacle[0] + 3 * em
                ),
                None,
            )
            if split is None or split < 2:
                continue
            narrow, tails = lines[:split], lines[split:]
            nb = _bbox_union_many([local(line.bbox) for line in narrow])
            tb = local(tails[0].bbox)
            if not (
                nb[2] <= obstacle[0] + 0.25 * em
                and nb[3] > cb[1]
                and cb[3] - 0.25 * em <= tb[1] <= cb[3] + em
                and abs(tb[0] - nb[0]) <= 0.5 * em
                and tb[2] - tb[0] >= 1.5 * (nb[2] - nb[0])
            ):
                continue
            consumed, first = [block], block
            while not any(first.get(key) for key in ("_explicit_break_before", "_rule_break_before", "_geometry_break_before")):
                preceding = [
                    other
                    for other in output
                    if all(other is not item for item in consumed)
                    and other.get("type") == "text"
                    and other.get("_text_lines")
                    and abs(local(other["bbox"])[0] - nb[0]) <= 0.5 * em
                    and local(other["bbox"])[2] <= obstacle[0] + 0.25 * em
                    and 0 <= nb[1] - other["bbox"][3] <= 0.6 * em
                ]
                if not preceding:
                    break
                previous = max(preceding, key=lambda item: item["bbox"][3])
                if previous["_text_lines"][-1].paragraph_terminal or re.search(
                    r"[.!?。！？][\])’\"']*$", str(previous.get("content", "")).rstrip()
                ):
                    break
                if not _title_fonts_compatible(previous["_text_lines"][-1], narrow[0]):
                    break
                consumed.append(previous)
                narrow = previous["_text_lines"] + narrow
                nb = _bbox_union_many([local(line.bbox) for line in narrow])
                first = previous
            rebuilt = _build_text_blocks(narrow, [], page_size)
            tail_blocks = _build_text_blocks(tails, [], page_size)
            # 独立几何约束阻止后续跨屏障扩框，不制造语义上的新自然段。
            for item in rebuilt + tail_blocks:
                item["_geometry_barriers"] = [cb]
            if tail_blocks:
                tail_blocks[0]["_geometry_break_before"] = True
            output = [item for item in output if all(item is not member for member in consumed)]
            output.extend(rebuilt + tail_blocks)
    return output


__all__ = ["_build_text_blocks"]
