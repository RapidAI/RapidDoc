"""提供同基线文本和拆分视觉行的几何合并。"""

from __future__ import annotations

from .layout_evidence import build_layout_evidence

import re
from bisect import bisect_right
import statistics
import unicodedata
import heapq
import math

from ....schema import BBox
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_y,
    _bbox_intersects,
    _bbox_union_many,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
)
from .line_layout import _connection_crosses_table, _font_signatures_share_family, _infer_text_lanes, _line_effective_height
from .models import _LineItem, _TextLane
from .native_text import _fill_native_typography, _median_native_glyph_width
from ._script_geometry import classify_char_script_roles, paired_script_roles
from ._interval_candidates import IntervalCandidates


def _inherit_unanimous_structural_membership(merged: _LineItem, members: list[_LineItem]) -> None:
    """同行片段共享明确段组或标题带时传递归属；不同归属不能由合并操作猜测。"""
    if not members:
        return
    ordered = sorted(members, key=lambda line: line.bbox[0])
    # 悬挂序号与标签合成同行时，传递标签释放字符前冻结的真实左缘。
    if (
        len(ordered) == 2
        and re.fullmatch(r"(?:[A-Za-z]|\d{1,2})[.)]", ordered[0].text.strip())
        and ordered[0].font_signature == ordered[1].font_signature
    ):
        merged.native_title_label_left = ordered[1].native_title_label_left
    paragraph_group = members[0].paragraph_group
    if paragraph_group is not None and all(member.paragraph_group == paragraph_group for member in members):
        merged.paragraph_group = paragraph_group
    title_band_id = members[0].title_band_id
    if title_band_id is not None and all(member.title_band_id == title_band_id for member in members):
        merged.title_band_id = title_band_id
        merged.structural_title = all(member.structural_title for member in members)
        merged.explicit_section_title = all(member.explicit_section_title for member in members)
    # 确认的正文也有标题反证；它不需要标题带身份，同行重建不能丢掉该边界。
    merged.title_suppressed = all(member.title_suppressed for member in members)


def _caption_crosses_left_text(members: list[_LineItem]) -> bool:
    """图题首行不得向左跨栏吞并正文，同栏后续行或右侧字体碎片仍可正常恢复。"""
    return any(
        seed.caption_start
        and other is not seed
        and other.bbox[2] <= seed.bbox[0] + 0.5 * _line_effective_height(seed, seed.bbox)
        and other.bbox[0] < seed.bbox[0] - _line_effective_height(seed, seed.bbox)
        for seed in members
        for other in members
    )


def _safe_candidate_box(box) -> bool:
    """仅让内置浮点坐标进入预筛，整数算术和其他类型保留参考遍历。"""
    return (
        type(box) in (tuple, list)
        and len(box) == 4
        and all(type(v) is float and abs(v) <= 1e100 and math.isfinite(v) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def _overlapping_candidate_pairs(members):
    """纵向区间是重叠连接的必要条件，原行序及最终判断保持不变。"""
    if len(members) <= 16 or any(not _safe_candidate_box(box) for _, box in members):
        return None
    groups = {}
    for i, (line, _) in enumerate(members):
        groups.setdefault((line.angle, line.formula_candidate_only, line.semantic_type), []).append(i)
    return IntervalCandidates([(box[1], box[3]) for _, box in members], groups)


def _same_baseline_candidate_pairs(
    lines: list[_LineItem],
    local_bboxes: list[BBox],
    compatible_indices: dict[tuple[int, bool, str | None], list[int]],
) -> list[list[int]] | IntervalCandidates | None:
    """筛除纵向远距行对，密集页面流式查询，异常几何保留原遍历。"""
    bounds: list[tuple[float, float]] = []
    for line, box in zip(lines, local_bboxes, strict=True):
        try:
            height = _line_effective_height(line, box)
            if not all(math.isfinite(value) for value in (*box, height)) or box[2] <= box[0] or box[3] <= box[1]:
                return None
            # 同行规则既接受框交叠，也接受底边差不超过 0.25 * max(height) 的行。
            # 两行各自扩展底边容差形成超集，最终仍由原判定函数决定是否合并。
            low, high = min(box[1], box[3] - 0.25 * height), box[3] + 0.25 * height
            if line.angle == 0 and line.source_bbox is not None:
                source = line.source_bbox
                if not all(math.isfinite(value) for value in source) or source[2] <= source[0] or source[3] <= source[1]:
                    return None
                source_height = source[3] - source[1]
                low = min(low, source[1], source[3] - 0.25 * source_height)
                high = max(high, source[3] + 0.25 * source_height)
            if not math.isfinite(low) or not math.isfinite(high):
                return None
        except (TypeError, ValueError, IndexError):
            return None
        bounds.append((low, high))

    from ...._compute_backend import get_native

    if len(lines) >= 32 and get_native() is not None:
        heights = [_line_effective_height(line, box) for line, box in zip(lines, local_bboxes)]
        sources = [line.source_bbox if line.angle == 0 and line.baseline is not None else None for line in lines]
        if (
            all(_safe_candidate_box(box) for box in local_bboxes)
            and all(box is None or _safe_candidate_box(box) for box in sources)
            and all(type(h) is float and math.isfinite(h) and abs(h) <= 1e100 for h in heights)
        ):
            return IntervalCandidates(bounds, compatible_indices, geometry=(local_bboxes, heights, sources))

    candidates: list[list[int]] = [[] for _ in lines]
    pair_count = 0
    pair_limit = max(4096, 32 * len(lines))
    for indices in compatible_indices.values():
        active: set[int] = set()
        ends: list[tuple[float, int]] = []
        for index in sorted(indices, key=lambda item: (bounds[item][0], item)):
            low, high = bounds[index]
            # 保留相等端点，避免漏掉原规则恰好落在容差边界上的行对。
            while ends and ends[0][0] < low:
                _, expired = heapq.heappop(ends)
                active.remove(expired)
            pair_count += len(active)
            if pair_count > pair_limit:
                return IntervalCandidates(bounds, compatible_indices)
            for previous in active:
                left, right = (previous, index) if previous < index else (index, previous)
                candidates[left].append(right)
            active.add(index)
            heapq.heappush(ends, (high, index))
    for partners in candidates:
        partners.sort()
    return candidates


def _merge_same_baseline_text_lines(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在表格认领后合并同基线、同字体且水平邻近的正文 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找同行合并并查集的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left_index: int, right_index: int) -> None:
        """合并两个满足同行条件的文本 run。"""

        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    # 只枚举满足必要分类条件的行对，组内仍按原索引递增，保持并查集认领顺序。
    compatible_indices: dict[tuple[int, bool, str | None], list[int]] = {}
    for index, line in enumerate(lines):
        compatible_indices.setdefault((line.angle, line.formula_candidate_only, line.semantic_type), []).append(index)
    candidates = _same_baseline_candidate_pairs(lines, local_bboxes, compatible_indices)
    for left_index, left_line in enumerate(lines):
        right_indices = (
            candidates[left_index]
            if candidates is not None
            else compatible_indices[(left_line.angle, left_line.formula_candidate_only, left_line.semantic_type)]
        )
        for right_index in right_indices:
            if right_index <= left_index:
                continue
            right_line = lines[right_index]
            if _can_merge_same_baseline_pair(
                left_line,
                local_bboxes[left_index],
                right_line,
                local_bboxes[right_index],
                table_bboxes,
            ):
                union(left_index, right_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)

    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(key=lambda index: (local_bboxes[index][0], local_bboxes[index][1], lines[index].source_index))
        output.append(_merge_same_baseline_group(indices, lines, local_bboxes, page_size))
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _merge_overlapping_inline_text_clusters(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在容器认领后恢复由分子、分母和上下标拆成的二维物理文本行。"""

    if len(lines) < 2:
        return list(lines)

    consumed_source_indices: set[int] = set()
    merged_lines: list[_LineItem] = []
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if len(angle_geometry) < 2:
            continue
        angle_median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in angle_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(
            angle_geometry,
            local_page_width,
            angle_median_height,
        )
        for lane in lanes:
            if lane.is_span or len(lane.lines) < 2:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            lane_median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in lane.lines)
            parents = list(range(len(lane.lines)))

            def find(index: int) -> int:
                """查找当前栏二维文本簇并查集的根节点。"""

                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(first_index: int, second_index: int) -> None:
                """合并两个满足二维物理行邻接条件的成员。"""

                first_root = find(first_index)
                second_root = find(second_index)
                if first_root != second_root:
                    parents[second_root] = first_root

            candidate_pairs = _overlapping_candidate_pairs(lane.lines)
            for first_index, first in enumerate(lane.lines):
                partners = (
                    candidate_pairs[first_index] if candidate_pairs is not None else range(first_index + 1, len(lane.lines))
                )
                for second_index in partners:
                    second = lane.lines[second_index]
                    if _overlapping_inline_cluster_pair_is_connected(
                        first,
                        second,
                        lane_median_height,
                        table_bboxes,
                        local_page_width=local_page_width,
                    ):
                        union(first_index, second_index)

            groups: dict[int, list[tuple[_LineItem, BBox]]] = {}
            for index, item in enumerate(lane.lines):
                groups.setdefault(find(index), []).append(item)
            for members in groups.values():
                cluster_kind = _classify_overlapping_inline_cluster(
                    members,
                    lane,
                    lane_median_height,
                    table_bboxes,
                )
                if cluster_kind is None:
                    continue
                merged_lines.append(
                    _merge_overlapping_inline_cluster(
                        members,
                        page_size,
                        lane_median_height,
                        compact_formula_cluster=cluster_kind == "formula",
                    )
                )
                consumed_source_indices.update(line.source_index for line, _bbox in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(merged_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[1],
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[0],
            line.source_index,
        )
    )
    return output


def _overlapping_inline_cluster_pair_is_connected(
    first: tuple[_LineItem, BBox],
    second: tuple[_LineItem, BBox],
    median_height: float,
    table_bboxes: list[BBox],
    *,
    local_page_width: float | None = None,
) -> bool:
    """判断两个同栏成员是否为同一二维物理行中的重叠片段。"""

    first_line, first_bbox = first
    second_line, second_bbox = second
    if first_line.angle != second_line.angle:
        return False
    if first_line.formula_candidate_only != second_line.formula_candidate_only:
        return False
    if first_line.semantic_type != second_line.semantic_type:
        return False
    if _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y") < 0.55:
        return False
    if _connection_crosses_table(
        first_line.bbox,
        second_line.bbox,
        table_bboxes,
    ):
        return False

    first_guard_bbox = first_line.source_bbox or first_bbox
    second_guard_bbox = second_line.source_bbox or second_bbox
    left_bbox, right_bbox = sorted(
        (first_guard_bbox, second_guard_bbox),
        key=lambda bbox: bbox[0],
    )
    if (
        local_page_width is not None
        and (first_line.style_scale_repaired or second_line.style_scale_repaired)
        and left_bbox[2] <= 0.5 * local_page_width
        and right_bbox[0] >= 0.5 * local_page_width
        and right_bbox[0] - left_bbox[2] >= 0.02 * local_page_width
    ):
        return False

    horizontal_gap = _horizontal_bbox_gap(first_bbox, second_bbox)
    if first_line.visual_row_id == second_line.visual_row_id and (first_line.split_from_row or second_line.split_from_row):
        return horizontal_gap <= 3.0 * median_height
    pair_height = max(
        _line_effective_height(first_line, first_bbox),
        _line_effective_height(second_line, second_bbox),
    )
    return _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x") > 0.0 or horizontal_gap <= 1.5 * pair_height


def _classify_overlapping_inline_cluster(
    members: list[tuple[_LineItem, BBox]],
    lane: _TextLane,
    median_height: float,
    table_bboxes: list[BBox],
) -> str | None:
    """按正文宿主和紧凑程度区分行内文本簇与独立公式簇。"""

    if len(members) < 2 or _caption_crosses_left_text([line for line, _ in members]):
        return None
    visual_row_ids = {line.visual_row_id for line, _bbox in members if line.visual_row_id is not None}
    if len(visual_row_ids) < 2:
        return None
    if any(_bbox_intersects(line.bbox, table_bbox) for line, _bbox in members for table_bbox in table_bboxes):
        return None

    union_bbox = _bbox_union_many([bbox for _line, bbox in members])
    if union_bbox[3] - union_bbox[1] > 3.0 * median_height:
        return None
    lane_width = max(0.1, lane.right - lane.left)
    has_fragment = any(
        _line_effective_height(line, bbox) <= 0.88 * median_height or line.font_coverage < 0.75 for line, bbox in members
    )
    if not has_fragment:
        return None

    has_body_host = any(
        bbox[2] - bbox[0] >= max(4.0 * median_height, 0.35 * lane_width)
        and _line_effective_height(line, bbox) >= 0.8 * median_height
        and line.font_coverage >= 0.75
        for line, bbox in members
    )
    if has_body_host or _has_paired_script_body_host(members, median_height):
        return "inline"
    if len(members) >= 3 and len(visual_row_ids) >= 3 and union_bbox[2] - union_bbox[0] <= 0.6 * lane_width:
        return "formula"
    return None


def _has_paired_script_body_host(members: list[tuple[_LineItem, BBox]], median_height: float) -> bool:
    """用正文宿主和跨 run 的成对上下标补证据，不放宽普通同行重叠阈值。"""

    if len(members) != 2 or any(line.angle != 0 or line.preserve_split_boundary or line.semantic_type for line, _ in members):
        return False
    host, bbox = max(members, key=lambda item: item[1][2] - item[1][0])
    has_prose = len(re.findall(r"[\u3400-\u9fff]", host.text)) >= 2 or len(re.findall(r"\b[A-Za-z]{2,}\b", host.text)) >= 2
    if not has_prose or bbox[2] - bbox[0] < 4.0 * median_height:
        return False
    owned_chars = [(char, line.source_index) for line, _ in members for char in line.chars]
    if not owned_chars or any(not isinstance(char.get("char_idx"), int) for char, _ in owned_chars):
        return False
    owned_chars.sort(key=lambda item: item[0]["char_idx"])
    chars = [char for char, _ in owned_chars]
    tight = {char["char_idx"]: char["tight_bbox"] for char in chars if char.get("tight_bbox") is not None}
    origins = {char["char_idx"]: char["origin"] for char in chars if char.get("origin") is not None}
    roles = classify_char_script_roles(chars, tight_bboxes=tight, origins=origins)
    paired = paired_script_roles(chars, roles, tight, origins)
    # 上下标必须分属两个待合并 run，避免把一侧已有角标当作吞并邻块的依据。
    return len(paired) == 2 and len({owned_chars[index][1] for index in paired}) == 2


def _select_overlapping_inline_cluster_host(
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> _LineItem:
    """选择与其他成员纵向重叠最多且最接近正文尺度的宿主行。"""

    union_bbox = _bbox_union_many([bbox for _line, bbox in members])
    union_center_y = _bbox_center_y(union_bbox)

    def host_score(item: tuple[_LineItem, BBox]) -> tuple[float, ...]:
        """生成宿主候选的正文尺度、同行支持和中心距离评分。"""

        line, bbox = item
        vertical_support = sum(
            max(0.0, min(bbox[3], other_bbox[3]) - max(bbox[1], other_bbox[1]))
            for other_line, other_bbox in members
            if other_line is not line
        )
        same_row_support = sum(
            line.visual_row_id is not None and line.visual_row_id == other_line.visual_row_id
            for other_line, _other_bbox in members
            if other_line is not line
        )
        body_like = float(_line_effective_height(line, bbox) >= 0.8 * median_height and line.font_coverage >= 0.75)
        return (
            body_like,
            float(same_row_support),
            vertical_support,
            bbox[2] - bbox[0],
            -abs(_bbox_center_y(bbox) - union_center_y),
            -float(line.source_index),
        )

    return max(members, key=host_score)[0]


def _merge_overlapping_inline_cluster(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    median_height: float,
    *,
    compact_formula_cluster: bool,
) -> _LineItem:
    """按来源顺序合并二维文本簇，并保留字符与宿主排版信息。"""

    ordered_members = sorted(
        (line for line, _bbox in members),
        key=lambda line: line.source_index,
    )
    host = _select_overlapping_inline_cluster_host(members, median_height)
    detected_regions = (
        [line.bbox for line in ordered_members]
        if compact_formula_cluster
        else [line.bbox for line in ordered_members if line is not host]
    )
    # 成对角标跨 run 只是 PDF 物理拆行，不应在上、下标之间补正文空格。
    separator = "" if _has_paired_script_body_host(members, median_height) else " "
    merged = _LineItem(
        text=separator.join(text for line in ordered_members if (text := line.text.strip())),
        bbox=_bbox_union_many([line.bbox for line in ordered_members]),
        angle=host.angle,
        source_index=min(line.source_index for line in ordered_members),
        source_bbox=_bbox_union_many(
            [line.source_bbox or line.bbox for line in ordered_members],
        ),
        ink_bbox=(
            _bbox_union_many(
                [line.ink_bbox for line in ordered_members if line.ink_bbox is not None],
            )
            if any(line.ink_bbox is not None for line in ordered_members)
            else None
        ),
        baseline=host.baseline,
        chars=[char for line in ordered_members for char in line.chars],
        visual_row_id=host.visual_row_id,
        run_index=host.run_index,
        effective_height=host.effective_height,
        em_height=host.em_height or host.effective_height,
        font_signature=host.font_signature,
        font_coverage=host.font_coverage,
        dominant_font_weight=host.dominant_font_weight,
        median_glyph_width=host.median_glyph_width,
        leading_emphasis_width=ordered_members[0].leading_emphasis_width,
        leading_typography_width=ordered_members[0].leading_typography_width,
        paragraph_terminal=ordered_members[-1].paragraph_terminal,
        caption_start=any(member.caption_start for member in ordered_members),
        native_math_words=frozenset(word for member in ordered_members for word in member.native_math_words),
        paragraph_formula_context=any(line.paragraph_formula_context for line in ordered_members),
        split_from_row=any(line.split_from_row for line in ordered_members),
        preserve_split_boundary=any(line.preserve_split_boundary for line in ordered_members),
        semantic_type=host.semantic_type,
        restored_inline_cluster=True,
        compact_formula_cluster=compact_formula_cluster,
        formula_candidate_only=all(line.formula_candidate_only for line in ordered_members),
        style_scale_repaired=any(line.style_scale_repaired for line in ordered_members),
        inline_math_regions=[
            *(region for line in ordered_members for region in line.inline_math_regions),
            *detected_regions,
        ],
    )
    _inherit_unanimous_structural_membership(merged, ordered_members)
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _merge_post_semantic_text_runs(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在容器、公式和标题结束后合并紧贴同基线的普通混合字体 run。"""

    if len(lines) < 2:
        return list(lines)
    local_bboxes = [_rotate_bbox_to_upright(line.bbox, page_size, line.angle) for line in lines]
    layout = build_layout_evidence(lines, page_size, barriers=table_bboxes)
    candidate_index = _post_semantic_candidate_pairs(lines, local_bboxes)
    candidates, heights = candidate_index if candidate_index is not None else (None, None)
    physical_rows: dict[tuple[int, int], tuple[bool, BBox]] = {}
    supporting_rows = sorted((line.bbox[1], index) for index, line in enumerate(lines)) if candidates is not None else []
    supporting_tops = [top for top, _index in supporting_rows]
    parents = list(range(len(lines)))

    def find(index: int) -> int:
        """查找后处理同行合并分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个后处理同行分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first_line in enumerate(lines):
        if first_line.semantic_type is not None:
            continue
        first_bbox = local_bboxes[first_index]
        first_height = heights[first_index] if heights is not None else _line_effective_height(first_line, first_bbox)
        for second_index in candidates[first_index] if candidates is not None else range(first_index + 1, len(lines)):
            second_line = lines[second_index]
            if _caption_crosses_left_text([first_line, second_line]):
                continue
            if second_line.semantic_type is not None or first_line.angle != second_line.angle:
                continue
            if first_line.formula_candidate_only != second_line.formula_candidate_only:
                continue
            if _connection_crosses_table(
                first_line.bbox,
                second_line.bbox,
                table_bboxes,
            ):
                continue
            second_bbox = local_bboxes[second_index]
            second_height = heights[second_index] if heights is not None else _line_effective_height(second_line, second_bbox)
            left, right = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
            pair_height = min(first_height, second_height)
            same_physical_row = first_line.visual_row_id is not None and first_line.visual_row_id == second_line.visual_row_id
            row_key = (first_line.angle, first_line.visual_row_id)
            if same_physical_row and candidates is not None and row_key not in physical_rows:
                members = [
                    index
                    for index, member in enumerate(lines)
                    if member.angle == row_key[0] and member.visual_row_id == row_key[1]
                ]
                physical_rows[row_key] = (
                    all(any(char.isalpha() for char in lines[index].text) for index in members),
                    _bbox_union_many([local_bboxes[index] for index in members]),
                )
            shared_physical_row = same_physical_row and (
                physical_rows[row_key][0]
                if candidates is not None
                else all(
                    any(char.isalpha() for char in member.text)
                    for member in lines
                    if member.angle == first_line.angle and member.visual_row_id == first_line.visual_row_id
                )
            )
            if shared_physical_row:
                row_bounds = (
                    physical_rows[row_key][1]
                    if candidates is not None
                    else _bbox_union_many(
                        [
                            local_bboxes[index]
                            for index, member in enumerate(lines)
                            if member.angle == first_line.angle and member.visual_row_id == first_line.visual_row_id
                        ]
                    )
                )
            else:
                row_bounds = _bbox_union_many([first_bbox, second_bbox])
            justified_row = (
                first_line.font_signature == second_line.font_signature
                and first_line.paragraph_group == second_line.paragraph_group
                and first_line.angle == 0
                and abs(_bbox_center_y(first_bbox) - _bbox_center_y(second_bbox)) < 0.2 * pair_height
                and 0 <= right[0] - left[2] <= (3 if shared_physical_row else 2) * pair_height
                and not layout.separated(first_bbox, second_bbox)
                and _post_semantic_supports_row(
                    lines,
                    first_index,
                    second_index,
                    first_bbox,
                    second_bbox,
                    row_bounds,
                    pair_height,
                    shared_physical_row,
                    supporting_rows if candidates is not None else None,
                    supporting_tops,
                )
            )
            if (
                _post_semantic_same_baseline_geometry(
                    first_bbox,
                    first_height,
                    second_bbox,
                    second_height,
                )
                or justified_row
            ):
                union(first_index, second_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    output: list[_LineItem] = []
    for indices in groups.values():
        if len(indices) == 1:
            output.append(lines[indices[0]])
            continue
        indices.sort(
            key=lambda index: (
                local_bboxes[index][0],
                local_bboxes[index][1],
                lines[index].source_index,
            )
        )
        output.append(
            _merge_same_baseline_group(
                indices,
                lines,
                local_bboxes,
                page_size,
            )
        )
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _post_semantic_candidate_pairs(
    lines: list[_LineItem], local_bboxes: list[BBox]
) -> tuple[IntervalCandidates, list[float]] | None:
    """两条后处理合并路径共用纵向安全超集，保留原始行对枚举顺序。"""

    if len(lines) <= 32 or any(
        type(line) is not _LineItem
        or not _safe_candidate_box(box)
        or not _safe_candidate_box(line.bbox)
        or type(line.angle) is not int
        or type(line.formula_candidate_only) is not bool
        or (line.semantic_type is not None and type(line.semantic_type) is not str)
        or (line.visual_row_id is not None and type(line.visual_row_id) is not int)
        or type(line.text) is not str
        for line, box in zip(lines, local_bboxes, strict=True)
    ):
        return None
    bounds = []
    heights = []
    groups = {}
    for index, (line, box) in enumerate(zip(lines, local_bboxes, strict=True)):
        height = _line_effective_height(line, box)
        if type(height) is not float or not math.isfinite(height) or height > 1e100:
            return None
        low, high = box[1] - 0.2 * height, box[3] + 0.2 * height
        if not math.isfinite(low) or not math.isfinite(high):
            return None
        bounds.append((low, high))
        heights.append(height)
        groups.setdefault((line.angle, line.formula_candidate_only, line.semantic_type), []).append(index)
    return IntervalCandidates(bounds, groups), heights


def _post_semantic_supports_row(
    lines: list[_LineItem],
    first_index: int,
    second_index: int,
    first_bbox: BBox,
    second_bbox: BBox,
    row_bounds: BBox,
    pair_height: float,
    shared_physical_row: bool,
    supporting_rows: list[tuple[float, int]] | None,
    supporting_tops: list[float],
) -> bool:
    """按原条件检查后续正文行；普通几何只查询必要的纵向带。"""

    bottom = max(first_bbox[3], second_bbox[3])
    if supporting_rows is None:
        indices = range(len(lines))
    else:
        start = bisect_right(supporting_tops, bottom)
        end = bisect_right(supporting_tops, bottom + pair_height)
        indices = sorted(index for _top, index in supporting_rows[start:end])
    for index in indices:
        other = lines[index]
        if (
            index not in (first_index, second_index)
            and other is not lines[first_index]
            and other is not lines[second_index]
            and 0 < other.bbox[1] - bottom <= pair_height
            and abs(other.bbox[0] - row_bounds[0]) < (1.5 if shared_physical_row else 1) * pair_height
            and other.bbox[2] >= row_bounds[2] - pair_height
        ):
            return True
    return False


def _post_semantic_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """放宽上下标字号差异，仅合并水平紧贴且垂直充分交叠的普通 run。"""

    pair_height = max(first_height, second_height)
    if (
        min(first_height, second_height) <= 0
        or pair_height
        / min(
            first_height,
            second_height,
        )
        > 1.6
    ):
        return False
    y_overlap = max(
        0.0,
        min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]),
    )
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    horizontal_gap = right_bbox[0] - left_bbox[2]
    return -0.2 * pair_height <= horizontal_gap <= max(2.0, 0.35 * pair_height)


def _can_merge_same_baseline_pair(
    first: _LineItem,
    first_bbox: BBox,
    second: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """判断两个剩余文本 run 是否属于同一条物理基线。"""

    if first.angle != second.angle or _caption_crosses_left_text([first, second]):
        return False
    if first.formula_candidate_only != second.formula_candidate_only:
        return False
    if first.semantic_type != second.semantic_type:
        return False
    if first.visual_row_id == second.visual_row_id and (first.split_from_row or second.split_from_row):
        return False
    if _consecutive_source_row(first, second):
        return not _connection_crosses_table(first.bbox, second.bbox, table_bboxes)
    first_height = _line_effective_height(first, first_bbox)
    second_height = _line_effective_height(second, second_bbox)
    has_compatible_dominant_font = not (
        first.font_signature is None
        or second.font_signature is None
        or first.font_coverage < 0.75
        or second.font_coverage < 0.75
        or first.font_signature != second.font_signature
    )
    if has_compatible_dominant_font and _same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    ):
        return not _connection_crosses_table(first.bbox, second.bbox, table_bboxes)
    return _touching_same_baseline_geometry(
        first_bbox,
        first_height,
        second_bbox,
        second_height,
    ) and not _connection_crosses_table(first.bbox, second.bbox, table_bboxes)


def _consecutive_source_row(first: _LineItem, second: _LineItem) -> bool:
    """以连续源字符和原始行框验证同行，避免替代字形的 ink 修复扩大字体间隙。"""
    if first.angle != 0 or second.angle != 0 or first.baseline is None or second.baseline is None:
        return False
    if first.preserve_split_boundary or second.preserve_split_boundary or not first.chars or not second.chars:
        return False
    first_box, second_box = first.source_bbox, second.source_bbox
    if first_box is None or second_box is None:
        return False
    first_height, second_height = first_box[3] - first_box[1], second_box[3] - second_box[1]
    # 原始大框可能覆盖整条分式，仍须有相近基线才能按普通同行处理。
    if abs(first.baseline - second.baseline) > 0.25 * min(first_height, second_height):
        return False
    if not _same_baseline_geometry(first_box, first_height, second_box, second_height):
        return False
    # 源框与基线已包含文本矩阵的缩放；名义字号比不能代表实际显示大小。
    left, right = sorted((first, second), key=lambda line: line.bbox[0])
    left_indices = [index for char in left.chars for index in char.get("source_indices", (char.get("char_idx"),))]
    right_indices = [index for char in right.chars for index in char.get("source_indices", (char.get("char_idx"),))]
    # 最多容许一个 PDFium 空格；跨行或跨阅读顺序的 run 不以字形距离强行合并。
    if not left_indices or not right_indices or not all(isinstance(index, int) for index in left_indices + right_indices):
        return False
    return 1 <= min(right_indices) - max(left_indices) <= 2


def _touching_same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
) -> bool:
    """为低字体覆盖率 run 提供严格的紧贴同基线几何兜底。"""

    pair_height = max(first_height, second_height)
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    smaller_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / smaller_bbox_height < 0.7:
        return False
    if abs(_bbox_center_y(first_bbox) - _bbox_center_y(second_bbox)) > 0.5 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    return -0.15 * pair_height <= signed_gap <= 0.75


def _same_baseline_geometry(
    first_bbox: BBox,
    first_height: float,
    second_bbox: BBox,
    second_height: float,
    *,
    maximum_gap: float | None = None,
) -> bool:
    """仅依据行高、垂直交叠和水平净空判断两个局部 bbox 是否同基线相邻。"""

    pair_height = max(first_height, second_height)
    if min(first_height, second_height) <= 0 or pair_height / min(first_height, second_height) > 1.35:
        return False
    y_overlap = max(0.0, min(first_bbox[3], second_bbox[3]) - max(first_bbox[1], second_bbox[1]))
    shorter_bbox_height = max(
        0.1,
        min(first_bbox[3] - first_bbox[1], second_bbox[3] - second_bbox[1]),
    )
    if y_overlap / shorter_bbox_height < 0.7 and abs(first_bbox[3] - second_bbox[3]) > 0.25 * pair_height:
        return False
    left_bbox, right_bbox = sorted((first_bbox, second_bbox), key=lambda bbox: bbox[0])
    signed_gap = right_bbox[0] - left_bbox[2]
    gap_limit = max(3.0, 0.75 * pair_height) if maximum_gap is None else maximum_gap
    return -0.25 * pair_height <= signed_gap <= gap_limit


def _bullet_body_typography_member(members: list[_LineItem]) -> _LineItem | None:
    """游离圆点只是项目标记，合行的字体统计应继承右侧主要文字，不能让符号字体覆盖整行正文。"""
    first = members[0]
    if (
        len(members) < 2
        or first.text.strip() not in {"•", "●", "▪"}
        or first.bbox[2] - first.bbox[0] > _line_effective_height(first, first.bbox)
    ):
        return None
    body = max(members[1:], key=lambda line: len(line.text.strip()))
    if len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", body.text)) < 2:
        return None
    return body


def _merge_same_baseline_group(
    indices: list[int],
    lines: list[_LineItem],
    local_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序合并一个同基线分量，并保留全部字符与几何信息。"""

    members = [lines[index] for index in indices]
    body_typography = _bullet_body_typography_member(members)
    content_parts = [members[0].text.strip()]
    for previous_index, current_index in zip(indices, indices[1:]):
        previous_bbox = local_bboxes[previous_index]
        current_bbox = local_bboxes[current_index]
        signed_gap = current_bbox[0] - previous_bbox[2]
        glyph_width = statistics.median(
            [
                width
                for member in (lines[previous_index], lines[current_index])
                if (width := _median_native_glyph_width(member, page_size)) is not None
            ]
            or [1.0]
        )
        separator = "" if signed_gap <= max(0.5, 0.25 * glyph_width) else " "
        content_parts.extend([separator, lines[current_index].text.strip()])

    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in members]),
        angle=members[0].angle,
        source_index=min(member.source_index for member in members),
        source_bbox=_bbox_union_many(
            [member.source_bbox or member.bbox for member in members],
        ),
        ink_bbox=(
            _bbox_union_many(
                [member.ink_bbox for member in members if member.ink_bbox is not None],
            )
            if any(member.ink_bbox is not None for member in members)
            else None
        ),
        baseline=(
            statistics.median(member.baseline for member in members if member.baseline is not None)
            if any(member.baseline is not None for member in members)
            else None
        ),
        chars=[char for member in members for char in member.chars],
        visual_row_id=min(
            (member.visual_row_id for member in members if member.visual_row_id is not None),
            default=None,
        ),
        run_index=min(member.run_index for member in members),
        effective_height=statistics.median(member.effective_height for member in members),
        em_height=statistics.median(member.em_height or member.effective_height for member in members),
        font_signature=(body_typography or members[0]).font_signature,
        font_coverage=body_typography.font_coverage if body_typography else min(member.font_coverage for member in members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight for member in members if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width for member in members if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in members)
        else None,
        leading_emphasis_width=members[0].leading_emphasis_width,
        leading_typography_width=members[0].leading_typography_width,
        paragraph_terminal=members[-1].paragraph_terminal,
        caption_start=any(member.caption_start for member in members),
        native_math_words=frozenset(word for member in members for word in member.native_math_words),
        paragraph_formula_context=any(member.paragraph_formula_context for member in members),
        split_from_row=any(member.split_from_row for member in members),
        preserve_split_boundary=any(member.preserve_split_boundary for member in members),
        semantic_type=members[0].semantic_type,
        restored_inline_cluster=any(member.restored_inline_cluster for member in members),
        compact_formula_cluster=any(member.compact_formula_cluster for member in members),
        formula_candidate_only=all(member.formula_candidate_only for member in members),
        style_scale_repaired=any(member.style_scale_repaired for member in members),
        inline_math_regions=[region for member in members for region in member.inline_math_regions],
    )
    _inherit_unanimous_structural_membership(merged, members)
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def _join_formula_visual_row(
    row: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
) -> str:
    """将一个公式视觉行按局部 x 排序，并按字宽估计几何空格。"""

    ordered = sorted(row, key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not ordered:
        return ""
    parts = [ordered[0][0].text.strip()]
    for previous, current in zip(ordered, ordered[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        gap = current_bbox[0] - previous_bbox[2]
        pair_height = max(
            _line_effective_height(previous_line, previous_bbox),
            _line_effective_height(current_line, current_bbox),
        )
        glyph_widths = [
            width
            for line in (previous_line, current_line)
            if (width := _median_native_glyph_width(line, page_size)) is not None
        ]
        glyph_width = statistics.median(glyph_widths) if glyph_widths else max(1.0, 0.5 * pair_height)
        if gap <= max(0.5, 0.2 * pair_height):
            separator = ""
        else:
            separator = " " * max(1, min(8, int(round(gap / glyph_width))))
        parts.extend([separator, current_line.text.strip()])
    return "".join(parts).strip()


def _restore_dense_split_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
) -> list[_LineItem]:
    """在公式认领后恢复同一栏带内被均匀大空格拆开的密集原生视觉行。"""

    if len(lines) < 3:
        return list(lines)
    lane_keys: dict[int, tuple[int, int]] = {}
    for angle in sorted({line.angle for line in lines}):
        line_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in lines if line.angle == angle]
        if not line_geometry:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(line_geometry, local_page_width, median_height)
        for lane_index, lane in enumerate(lanes):
            if lane.is_span:
                continue
            for line, _bbox in lane.lines:
                lane_keys[line.source_index] = (angle, lane_index)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    restored_lines: list[_LineItem] = []
    for members in row_groups.values():
        if not _can_restore_dense_split_visual_row(
            members,
            page_size,
            table_bboxes,
            lane_keys,
        ):
            continue
        restored_lines.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(restored_lines)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return output


def _demote_runin_title_fragments(lines: list[_LineItem], page_size: tuple[float, float]) -> None:
    """同行正文延续的小标题属于段落内强调，保留原始字体证据供行内样式物化。"""
    layouts = {angle: build_layout_evidence(lines, page_size, angle=angle) for angle in {line.angle for line in lines}}
    for line in lines:
        was_title = line.semantic_type == "paragraph_title"
        if not was_title and not (
            line.semantic_type is None and (line.dominant_font_weight or 0) >= 550 and len(line.text.split()) <= 8
        ):
            continue
        bbox = _rotate_bbox_to_upright(line.ink_bbox or line.bbox, page_size, line.angle)
        height = _line_effective_height(line, bbox)
        for other in lines:
            if other is line or other.angle != line.angle or other.semantic_type is not None:
                continue
            if other.formula_candidate_only or other.compact_formula_cluster:
                continue
            if not was_title and (
                other.dominant_font_weight is None or line.dominant_font_weight < other.dominant_font_weight + 100
            ):
                continue
            other_bbox = _rotate_bbox_to_upright(other.ink_bbox or other.bbox, page_size, other.angle)
            other_height = _line_effective_height(other, other_bbox)
            if height > 1.3 * other_height or len(other.text.strip()) < 3:
                continue
            if (
                abs(_bbox_center_y(bbox) - _bbox_center_y(other_bbox)) <= 0.45 * min(height, other_height)
                and min(bbox[2], other_bbox[2]) - max(bbox[0], other_bbox[0]) <= 0.5 * height
                and not layouts[line.angle].separated(bbox, other_bbox)
                and _same_baseline_geometry(bbox, height, other_bbox, other_height, maximum_gap=5 * max(height, other_height))
            ):
                line.semantic_type = None
                if bbox[0] < other_bbox[0]:
                    line.leading_emphasis_width = bbox[2] - bbox[0]
                    line.leading_typography_width = bbox[2] - bbox[0]
                line.title_suppressed = True
                line.structural_title = False
                line.explicit_section_title = False
                for prefix in lines:
                    if (
                        prefix.semantic_type == "paragraph_title"
                        # 单独字母序号是下一项的前缀，不是上一标题的换行碎片，不能向上传播降级。
                        and re.fullmatch(r"[a-zA-Z][.)]", line.text.strip()) is None
                        and prefix.angle == line.angle
                        and not prefix.explicit_section_title
                        and abs(prefix.bbox[0] - line.bbox[0]) < height
                        and 0 <= line.bbox[1] - prefix.bbox[3] <= height
                        and prefix.font_signature == line.font_signature
                    ):
                        prefix.semantic_type = None
                        prefix.title_suppressed = True
                break


def _merge_metric_heading_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """同排大号标题为数值指标提供版式证据，仅聚合紧邻方向箭头及引用上标，保留下一行说明。"""
    consumed: set[int] = set()
    merged_lines: list[_LineItem] = []
    for value in lines:
        if (
            value.source_index in consumed
            or value.semantic_type not in {None, "paragraph_title"}
            or value.caption_start
            or value.paragraph_terminal
            or value.font_signature is None
            or value.font_coverage < 0.75
            or value.formula_candidate_only
            or value.compact_formula_cluster
            or re.fullmatch(r"\d+(?:[.,]\d+)?\s*(?:[A-Za-z]{1,4}|[%×])", value.text.strip()) is None
        ):
            continue
        bounds = _rotate_bbox_to_upright(value.bbox, page_size, value.angle)
        height = _line_effective_height(value, bounds)
        arrows = []
        for arrow in lines:
            if (
                arrow is value
                or arrow.source_index in consumed
                or arrow.angle != value.angle
                or arrow.semantic_type != "paragraph_title"
                or arrow.font_signature != value.font_signature
                or arrow.caption_start
                or arrow.formula_candidate_only
                or arrow.compact_formula_cluster
                or re.fullmatch(r"[↑↓↗↘]\s*\d{0,3}", arrow.text.strip()) is None
            ):
                continue
            arrow_bounds = _rotate_bbox_to_upright(arrow.bbox, page_size, arrow.angle)
            if (
                0.45 * height <= _line_effective_height(arrow, arrow_bounds) <= 1.05 * height
                and 0 <= arrow_bounds[0] - bounds[2] <= 0.75 * height
                and _bbox_axis_overlap_ratio(bounds, arrow_bounds, axis="y") >= 0.7
            ):
                arrows.append(arrow)
        if len(arrows) != 1:
            continue
        peers = [
            other
            for other in lines
            if other is not value
            and other is not arrows[0]
            and other.angle == value.angle
            and other.semantic_type == "paragraph_title"
            and other.font_signature == value.font_signature
            and not other.caption_start
            and 0.8 * height <= _line_effective_height(other, other.bbox) <= 1.25 * height
            and abs(_bbox_center_y(other.bbox) - _bbox_center_y(value.bbox)) <= 0.35 * height
            and _horizontal_bbox_gap(other.bbox, value.bbox) >= 3 * height
        ]
        if not peers:
            continue
        merged = _merge_dense_split_visual_row([value, arrows[0]], page_size)
        merged.semantic_type = "paragraph_title"
        merged.structural_title = merged.explicit_section_title = True
        merged.title_band_id = value.source_index
        merged.effective_height = merged.em_height = height
        merged_lines.append(merged)
        consumed.update((value.source_index, arrows[0].source_index))
    if not consumed:
        return lines
    return [line for line in lines if line.source_index not in consumed] + merged_lines


def _merge_title_resolved_visual_rows(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """在标题判定后合并同行标题或已降级的混合字体正文 run。"""

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    consumed_source_indices: set[int] = set()
    merged_rows: list[_LineItem] = []
    for members in row_groups.values():
        if len(members) < 2 or not all(member.split_from_row for member in members):
            continue
        if any(member.preserve_split_boundary for member in members):
            continue
        semantic_types = {member.semantic_type for member in members}
        if len(semantic_types) != 1:
            continue
        semantic_type = next(iter(semantic_types))
        font_signatures = {member.font_signature for member in members}
        dense_same_font_text = _is_dense_same_font_two_run_row(
            members,
            page_size,
        )
        sparse_short_prefix_text = _is_sparse_short_prefix_two_run_row(
            members,
            page_size,
        )
        if semantic_type != "paragraph_title" and not (
            semantic_type is None and (len(font_signatures) > 1 or dense_same_font_text or sparse_short_prefix_text)
        ):
            continue
        local_geometry = [
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ]
        local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
        if any(
            not _same_baseline_geometry(
                previous[1],
                _line_effective_height(*previous),
                current[1],
                _line_effective_height(*current),
                maximum_gap=(5.0 if sparse_short_prefix_text else 3.0)
                * max(
                    _line_effective_height(*previous),
                    _line_effective_height(*current),
                ),
            )
            for previous, current in zip(local_geometry, local_geometry[1:])
        ):
            continue
        merged_rows.append(_merge_dense_split_visual_row(members, page_size))
        consumed_source_indices.update(member.source_index for member in members)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(merged_rows)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[1],
            _rotate_bbox_to_upright(line.bbox, page_size, line.angle)[0],
            line.source_index,
        )
    )
    return _merge_numbered_title_fragments(
        output,
        page_size,
    )


def _merge_numbered_title_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """合并 PDF 字体分割导致的章节编号与同基线标题文字。"""

    marker_re = re.compile(
        r"^\d+(?:\s*\.\s*\d+)*\.?$",
    )
    geometry = [
        (
            line,
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            ),
        )
        for line in lines
        if line.semantic_type == "paragraph_title"
    ]
    consumed: set[int] = set()
    merged: list[_LineItem] = []
    for marker, marker_bbox in geometry:
        if marker.source_index in consumed:
            continue
        normalized = re.sub(
            r"\s+",
            " ",
            unicodedata.normalize("NFKC", marker.text),
        ).strip()
        if marker_re.match(normalized) is None:
            continue
        marker_height = _line_effective_height(
            marker,
            marker_bbox,
        )
        companions = [
            (candidate, candidate_bbox)
            for candidate, candidate_bbox in geometry
            if candidate is not marker
            and candidate.source_index not in consumed
            and candidate.angle == marker.angle
            and candidate_bbox[0] >= marker_bbox[2]
            and candidate_bbox[0] - marker_bbox[2] <= 4.0 * marker_height
            and _bbox_axis_overlap_ratio(
                marker_bbox,
                candidate_bbox,
                axis="y",
            )
            >= 0.5
        ]
        if not companions:
            continue
        companion, _companion_bbox = min(
            companions,
            key=lambda item: (
                item[1][0] - marker_bbox[2],
                item[1][1],
            ),
        )
        merged.append(
            _merge_dense_split_visual_row(
                [marker, companion],
                page_size,
            )
        )
        consumed.update({marker.source_index, companion.source_index})
    output = [line for line in lines if line.source_index not in consumed]
    output.extend(merged)
    output.sort(
        key=lambda line: (
            line.angle,
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[1],
            _rotate_bbox_to_upright(
                line.bbox,
                page_size,
                line.angle,
            )[0],
            line.source_index,
        )
    )
    return output


def _is_dense_same_font_two_run_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> bool:
    """检查两个普通文本 run 是否为同字体且占用充分的完整视觉行。"""

    if (
        len(members) != 2
        or not all(member.split_from_row for member in members)
        or any(member.preserve_split_boundary for member in members)
        or any(member.semantic_type is not None for member in members)
        or any(member.font_signature is None for member in members)
        or any(member.font_coverage < 0.75 for member in members)
    ):
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != [0, 1]:
        return False
    if ordered[0].font_signature != ordered[1].font_signature:
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    first, second = local_geometry
    pair_height = max(
        _line_effective_height(*first),
        _line_effective_height(*second),
    )
    if not _same_baseline_geometry(
        first[1],
        _line_effective_height(*first),
        second[1],
        _line_effective_height(*second),
        maximum_gap=3.0 * pair_height,
    ):
        return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= 0.85


def _is_sparse_short_prefix_two_run_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> bool:
    """识别同视觉行中被宽空白拆开的短前缀与宽正文。"""

    if (
        len(members) != 2
        or not all(member.split_from_row for member in members)
        or any(member.preserve_split_boundary for member in members)
        or any(member.semantic_type is not None for member in members)
        or any(member.font_signature is None for member in members)
        or any(member.font_coverage < 0.7 for member in members)
    ):
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != [0, 1]:
        return False
    if not _font_signatures_share_family(
        ordered[0].font_signature,
        ordered[1].font_signature,
    ):
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(
                member.bbox,
                page_size,
                member.angle,
            ),
        )
        for member in ordered
    ]
    local_geometry.sort(
        key=lambda item: (
            item[1][0],
            item[1][1],
            item[0].source_index,
        )
    )
    first, second = local_geometry
    pair_height = max(
        _line_effective_height(*first),
        _line_effective_height(*second),
    )
    local_page_width = page_size[1] if ordered[0].angle in {90, 270} else page_size[0]
    horizontal_gap = second[1][0] - first[1][2]
    return (
        first[1][2] - first[1][0] <= 2.0 * pair_height
        and second[1][2] - second[1][0] >= 0.3 * local_page_width
        and 3.0 * pair_height < horizontal_gap <= 5.0 * pair_height
        and _same_baseline_geometry(
            first[1],
            _line_effective_height(*first),
            second[1],
            _line_effective_height(*second),
            maximum_gap=5.0 * pair_height,
        )
    )


def _can_restore_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
    table_bboxes: list[BBox],
    lane_keys: dict[int, tuple[int, int]],
) -> bool:
    """检查 hard-split run 是否构成同字体且占用充分的完整视觉行。"""

    if (
        len(members) < 2
        or not all(member.split_from_row for member in members)
        or any(member.preserve_split_boundary for member in members)
    ):
        return False
    if len({member.semantic_type for member in members}) != 1:
        return False
    if len({member.formula_candidate_only for member in members}) != 1:
        return False
    ordered = sorted(members, key=lambda member: member.run_index)
    if [member.run_index for member in ordered] != list(range(len(ordered))):
        return False
    member_lane_keys = {lane_keys.get(member.source_index) for member in ordered}
    same_inferred_lane = None not in member_lane_keys and len(member_lane_keys) == 1
    font_signatures = {member.font_signature for member in ordered}
    if len(font_signatures) != 1:
        return False
    if any(_bbox_intersects(member.bbox, table_bbox) for member in ordered for table_bbox in table_bboxes):
        return False

    local_geometry = [
        (
            member,
            _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
        )
        for member in ordered
    ]
    local_geometry.sort(key=lambda item: (item[1][0], item[1][1], item[0].source_index))
    if not same_inferred_lane:
        member_widths = [bbox[2] - bbox[0] for _member, bbox in local_geometry]
        if len(members) == 2 and min(member_widths) > 0.35 * max(member_widths):
            return False
        if 3 <= len(members) <= 6:
            return False
    heights = [_line_effective_height(member, bbox) for member, bbox in local_geometry]
    median_height = statistics.median(heights)
    glyph_widths = [
        width for member, _bbox in local_geometry if (width := _median_native_glyph_width(member, page_size)) is not None
    ]
    median_glyph_width = statistics.median(glyph_widths) if glyph_widths else 0.0
    gap_limit = (
        max(12.0, 1.75 * median_height, 3.0 * median_glyph_width)
        if same_inferred_lane
        else max(8.0, 2.0 * median_height, 2.5 * median_glyph_width)
    )
    for previous, current in zip(local_geometry, local_geometry[1:]):
        if not _same_baseline_geometry(
            previous[1],
            _line_effective_height(*previous),
            current[1],
            _line_effective_height(*current),
            maximum_gap=gap_limit,
        ):
            return False

    union_bbox = _bbox_union_many([bbox for _member, bbox in local_geometry])
    occupied_width = sum(bbox[2] - bbox[0] for _member, bbox in local_geometry)
    minimum_occupancy = (
        0.8
        if len(members) == 2 and same_inferred_lane
        else 0.85
        if len(members) == 2
        else 0.65
        if not same_inferred_lane
        else 0.65
    )
    return occupied_width / max(0.1, union_bbox[2] - union_bbox[0]) >= minimum_occupancy


def _merge_dense_split_visual_row(
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> _LineItem:
    """按局部 x 顺序恢复密集视觉行，正净空使用单空格，重叠片段直接连接。"""

    ordered_geometry = sorted(
        (
            (
                member,
                _rotate_bbox_to_upright(member.bbox, page_size, member.angle),
            )
            for member in members
        ),
        key=lambda item: (item[1][0], item[1][1], item[0].source_index),
    )
    content_parts = [ordered_geometry[0][0].text.strip()]
    for previous, current in zip(ordered_geometry, ordered_geometry[1:]):
        separator = "" if current[1][0] <= previous[1][2] else " "
        content_parts.extend([separator, current[0].text.strip()])

    ordered_members = [member for member, _bbox in ordered_geometry]
    body_typography = _bullet_body_typography_member(ordered_members)
    merged = _LineItem(
        text="".join(content_parts).strip(),
        bbox=_bbox_union_many([member.bbox for member in ordered_members]),
        angle=ordered_members[0].angle,
        source_index=min(member.source_index for member in ordered_members),
        source_bbox=_bbox_union_many(
            [member.source_bbox or member.bbox for member in ordered_members],
        ),
        ink_bbox=(
            _bbox_union_many(
                [member.ink_bbox for member in ordered_members if member.ink_bbox is not None],
            )
            if any(member.ink_bbox is not None for member in ordered_members)
            else None
        ),
        baseline=(
            statistics.median(member.baseline for member in ordered_members if member.baseline is not None)
            if any(member.baseline is not None for member in ordered_members)
            else None
        ),
        chars=[char for member in ordered_members for char in member.chars],
        visual_row_id=ordered_members[0].visual_row_id,
        run_index=0,
        effective_height=statistics.median(_line_effective_height(member, bbox) for member, bbox in ordered_geometry),
        em_height=statistics.median(_line_effective_height(member, bbox) for member, bbox in ordered_geometry),
        font_signature=(body_typography or ordered_members[0]).font_signature,
        font_coverage=body_typography.font_coverage
        if body_typography
        else min(member.font_coverage for member in ordered_members),
        dominant_font_weight=statistics.median(
            member.dominant_font_weight for member in ordered_members if member.dominant_font_weight is not None
        )
        if any(member.dominant_font_weight is not None for member in ordered_members)
        else None,
        median_glyph_width=statistics.median(
            member.median_glyph_width for member in ordered_members if member.median_glyph_width is not None
        )
        if any(member.median_glyph_width is not None for member in ordered_members)
        else None,
        leading_emphasis_width=ordered_members[0].leading_emphasis_width,
        leading_typography_width=ordered_members[0].leading_typography_width,
        paragraph_terminal=ordered_members[-1].paragraph_terminal,
        caption_start=any(member.caption_start for member in ordered_members),
        native_math_words=frozenset(word for member in ordered_members for word in member.native_math_words),
        paragraph_formula_context=any(member.paragraph_formula_context for member in ordered_members),
        split_from_row=False,
        preserve_split_boundary=any(member.preserve_split_boundary for member in ordered_members),
        semantic_type=ordered_members[0].semantic_type,
        restored_inline_cluster=any(member.restored_inline_cluster for member in ordered_members),
        compact_formula_cluster=any(member.compact_formula_cluster for member in ordered_members),
        formula_candidate_only=all(member.formula_candidate_only for member in ordered_members),
        style_scale_repaired=any(member.style_scale_repaired for member in ordered_members),
        inline_math_regions=[region for member in ordered_members for region in member.inline_math_regions],
    )
    _inherit_unanimous_structural_membership(merged, ordered_members)
    if merged.chars:
        _fill_native_typography(merged, page_size)
    return merged


def merge_text_line_clusters(
    lines: list[_LineItem], page_size: tuple[float, float], table_bboxes: list[BBox]
) -> list[_LineItem]:
    """共享同基线、重叠簇和第二次同基线合并的确定性闭包。"""
    lines = _merge_same_baseline_text_lines(lines, page_size, table_bboxes)
    lines = _merge_overlapping_inline_text_clusters(lines, page_size, table_bboxes)
    return _merge_same_baseline_text_lines(lines, page_size, table_bboxes)
