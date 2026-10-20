"""提供文本栏带、行距和行连接的共享布局判定。"""

from __future__ import annotations

import math
import re
import statistics
from typing import Sequence

from ....foundation._text import is_hyphen_at_line_end
from ....schema import BBox
from ...._compute_backend import get_native
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_intersects,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
)
from .models import _LineItem, _LocalAxisLine, _TextLane
from .inline.types import PDF_FONT_ITALIC_FLAG
from .typography import _normalized_font_family

_TIGHT_OUTPUT_PADDING = 1.0


def _title_fonts_compatible(first: _LineItem, second: _LineItem) -> bool:
    """检查标题字体和字重是否兼容；低字体覆盖率时仍保留可靠字重证据。"""

    font_conflicts = (
        first.font_signature is not None
        and second.font_signature is not None
        and first.font_coverage >= 0.75
        and second.font_coverage >= 0.75
        and first.font_signature != second.font_signature
    )
    weight_conflicts = _font_weights_conflict(first, second)
    return not (font_conflicts or weight_conflicts)


def _font_signatures_share_family(
    first: tuple[str, int] | None,
    second: tuple[str, int] | None,
) -> bool:
    """判断两个可靠字体签名是否仅因 PDF 子集前缀或描述标志不同。"""

    first_family = _normalized_font_family(first)
    second_family = _normalized_font_family(second)
    return first_family is not None and second_family is not None and first_family == second_family


def _font_signatures_share_emphasis_family(first: tuple[str, int], second: tuple[str, int]) -> bool:
    """仅在已确认的正文强调续行中忽略通用字重/斜体后缀，不改变标题或其他字体兼容规则。"""
    style_suffix = r"(?:semibold|demibold|regular|regu|roman|bold|light|medium|italic|ital|oblique)+$"
    first_family = re.sub(style_suffix, "", _normalized_font_family(first) or "")
    second_family = re.sub(style_suffix, "", _normalized_font_family(second) or "")
    return bool(first_family) and first_family == second_family


def _font_weights_conflict(first: _LineItem, second: _LineItem) -> bool:
    """判断两行是否存在足以构成段落硬边界的显著字重差异。"""

    return (
        first.dominant_font_weight is not None
        and second.dominant_font_weight is not None
        and abs(first.dominant_font_weight - second.dominant_font_weight) >= 100.0
        and max(first.dominant_font_weight, second.dominant_font_weight)
        >= 1.15 * min(first.dominant_font_weight, second.dominant_font_weight)
    )


def _should_connect_semantic_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """只用几何、字体和障碍连接同类型语义行，避免标题内容影响聚合。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    if previous_line.semantic_type != current_line.semantic_type:
        return False
    if (
        current_line.semantic_type == "paragraph_title"
        and current_line.explicit_section_title
        and current_line.title_band_id is not None
        and current_line.title_band_id != previous_line.title_band_id
    ):
        # 已确认的独立编号标题带构成永久边界，连续两级标题不能因紧排而粘连。
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    shared_title_band = previous_line.title_band_id is not None and previous_line.title_band_id == current_line.title_band_id
    confirmed_title_band = shared_title_band and previous_line.structural_title and current_line.structural_title
    if max(previous_height, current_height) / min(previous_height, current_height) > 1.35 and not confirmed_title_band:
        return False
    vertical_gap = _effective_text_row_gap(previous, current)
    if not -0.25 * pair_height <= vertical_gap <= max(1.25 * pair_height, regular_gap + 0.75 * pair_height):
        return False
    if previous_line.semantic_type == "paragraph_title" and vertical_gap > 0.5 * pair_height and not shared_title_band:
        return False
    font_conflicts = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature != current_line.font_signature
    )
    uncertain_document_title_font = (
        previous_line.semantic_type == "doc_title"
        and min(previous_line.font_coverage, current_line.font_coverage) < 0.85
        and (
            previous_line.dominant_font_weight is None
            or current_line.dominant_font_weight is None
            or abs(previous_line.dominant_font_weight - current_line.dominant_font_weight) < 100.0
        )
    )
    if (
        font_conflicts
        and not uncertain_document_title_font
        and not (shared_title_band and previous_line.structural_title and current_line.structural_title)
    ):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    centered_pair = abs(_bbox_center_x(previous_bbox) - _bbox_center_x(current_bbox)) <= 0.15 * lane_width
    aligned_pair = abs(previous_bbox[0] - current_bbox[0]) <= 0.75 * pair_height
    overlapping_pair = _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") >= 0.35
    return centered_pair or aligned_pair or overlapping_pair


def _line_style_scale(line: _LineItem, local_bbox: BBox) -> float:
    """返回 canonical 字体尺度，缺失时兼容旧有效行高与局部 bbox。"""

    return max(
        0.1,
        line.em_height
        if line.style_scale_repaired and line.em_height > 0
        else line.effective_height or (local_bbox[3] - local_bbox[1]),
    )


def _line_canonical_style_scale(line: _LineItem, local_bbox: BBox) -> float:
    """忽略语义选择标记，直接返回 tight/origin 校准后的字体尺度。"""

    return max(
        0.1,
        line.em_height if line.em_height > 0 else line.effective_height or (local_bbox[3] - local_bbox[1]),
    )


# 兼容既有私有入口，直接绑定同一实现，避免高频布局检查多经过一层 Python 转发。
_line_effective_height = _line_style_scale


def _line_layout_height(_line: _LineItem, local_bbox: BBox) -> float:
    """返回 canonical 布局包络高度，供公式与视觉容器空间判断使用。"""

    return max(0.1, local_bbox[3] - local_bbox[1])


def _line_tight_output_bbox(
    line: _LineItem,
    page_size: tuple[float, float],
) -> BBox | None:
    """把可靠 tight 字形并集四边各扩 1pt，并裁剪到页面范围。"""

    ink_bbox = _coerce_bbox(line.ink_bbox)
    if ink_bbox is None:
        return None
    return _clip_bbox(
        (
            ink_bbox[0] - _TIGHT_OUTPUT_PADDING,
            ink_bbox[1] - _TIGHT_OUTPUT_PADDING,
            ink_bbox[2] + _TIGHT_OUTPUT_PADDING,
            ink_bbox[3] + _TIGHT_OUTPUT_PADDING,
        ),
        page_size,
    )


def _lines_tight_output_bbox(
    lines: Sequence[_LineItem],
    page_size: tuple[float, float],
) -> BBox | None:
    """合并多行 tight+1pt 候选；缺失 tight 的成员继续使用原 layout bbox。"""

    output_bboxes: list[BBox] = []
    changed = False
    for line in lines:
        candidate = _line_tight_output_bbox(line, page_size)
        output_bboxes.append(candidate or line.bbox)
        changed = changed or candidate is not None
    if not changed or not output_bboxes:
        return None
    return _bbox_union_many(output_bboxes)


def _effective_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """按前一行顶边与有效行高计算净空，避免高数学字形拉长 bbox 底边。"""

    previous_line, previous_bbox = previous
    _current_line, current_bbox = current
    if previous_line.restored_inline_cluster:
        # 二维文本簇的 bbox 底边是真实分母边界；同时截断深度重叠，避免相邻分式互相成为段落屏障。
        return max(
            current_bbox[1] - previous_bbox[3],
            -0.25 * _line_effective_height(previous_line, previous_bbox),
        )
    return current_bbox[1] - (previous_bbox[1] + _line_effective_height(previous_line, previous_bbox))


def _effective_body_text_row_gap(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
) -> float:
    """正文连接优先使用 origin 基线节奏，缺证据时回退既有 bbox 净空。"""

    previous_line, previous_bbox = previous
    current_line, _current_bbox = current
    if (
        previous_line.baseline is not None
        and current_line.baseline is not None
        and current_line.baseline > previous_line.baseline
    ):
        previous_scale = _line_effective_height(previous_line, previous_bbox)
        current_scale = _line_effective_height(*current)
        pitch = current_line.baseline - previous_line.baseline
        if (
            0.5 * min(previous_scale, current_scale)
            <= pitch
            <= 3.0
            * max(
                previous_scale,
                current_scale,
            )
        ):
            return pitch - previous_scale
    return _effective_text_row_gap(previous, current)


def _lane_batch_native():
    """参考函数被替换时完整保留 Python 读取和调用行为。"""
    native = get_native()
    if native is None or any(
        actual is not expected
        for actual, expected in zip(
            (
                _line_effective_height,
                _line_style_scale,
                _best_lane_coverage,
                _fits_only_one_lane_ordered,
                _bbox_center_x,
                _bbox_center_y,
            ),
            _LANE_BATCH_REFERENCE_FUNCTIONS,
        )
    ):
        return None
    return native


def _supported_lane_intervals_native(line_geometry, width, median_height):
    """完整输入只读一次，锚点与中位数由 Rust 批量生成。"""
    native = _lane_batch_native()
    return None if native is None else native.supported_lane_intervals_owned(line_geometry, width, median_height, _LineItem)


def _supported_lane_intervals_python(line_geometry, local_page_width, median_height):
    """保留原贪心锚点、统计中位数和首个支持区间的完整参考。"""
    anchor_tolerance = max(3.0, 0.75 * median_height)
    anchor_geometry = [
        item for item in line_geometry if item[0].semantic_type not in {"header", "footer", "page_number", "aside_text"}
    ]
    regular_lines = [
        item
        for item in anchor_geometry
        if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.15 * local_page_width)
    ]
    left_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(regular_lines, key=lambda value: value[1][0]):
        if not left_clusters:
            left_clusters.append([item])
            continue
        cluster_left = statistics.median(member[1][0] for member in left_clusters[-1])
        if abs(item[1][0] - cluster_left) <= anchor_tolerance:
            left_clusters[-1].append(item)
        else:
            left_clusters.append([item])

    supported_intervals = [
        (
            statistics.median(item[1][0] for item in cluster),
            statistics.median(item[1][2] for item in cluster),
            len(cluster),
        )
        for cluster in left_clusters
        if len(cluster) >= 3
    ]
    supported_intervals.sort(key=lambda interval: interval[0])
    filtered_intervals: list[tuple[float, float, int]] = []
    for interval in supported_intervals:
        if not filtered_intervals:
            filtered_intervals.append(interval)
            continue
        previous = filtered_intervals[-1]
        minimum_gutter = max(6.0, 0.75 * median_height)
        if interval[0] - previous[1] >= minimum_gutter:
            filtered_intervals.append(interval)
        elif interval[2] > previous[2]:
            filtered_intervals[-1] = interval

    if not filtered_intervals:
        source = regular_lines or line_geometry
        filtered_intervals = [
            (
                min(item[1][0] for item in source),
                max(item[1][2] for item in source),
                len(source),
            )
        ]

    return anchor_geometry, filtered_intervals


def _lane_assignments_native(line_geometry, lanes, tolerance, nested=None):
    """分栏时重新读取当前几何，不复用先前阶段的成员或判定结果。"""
    native = _lane_batch_native()
    return (
        None
        if native is None
        else native.lane_assignments_owned(
            line_geometry,
            lanes,
            tolerance,
            nested,
            _LineItem,
            _TextLane,
        )
    )


def _infer_text_lanes(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
    *,
    recalculate_intervals: bool = True,
) -> list[_TextLane]:
    """从重复左右边缘推断稳定栏带，并按需用已分配成员重算边界。"""

    anchor_tolerance = max(3.0, 0.75 * median_height)
    supported = _supported_lane_intervals_native(line_geometry, local_page_width, median_height)
    if supported is None:
        anchor_geometry, filtered_intervals = _supported_lane_intervals_python(line_geometry, local_page_width, median_height)
    else:
        filtered_intervals, anchor_indices = supported
        anchor_geometry = [line_geometry[i] for i in anchor_indices]

    nested_column_band = None
    if len(filtered_intervals) == 1 and anchor_geometry:
        nested_outer_interval = (
            (
                min(item[1][0] for item in anchor_geometry),
                max(item[1][2] for item in anchor_geometry),
            )
            if recalculate_intervals
            else filtered_intervals[0][:2]
        )
        nested_column_band = _infer_nested_column_band(
            anchor_geometry,
            local_page_width,
            median_height,
            nested_outer_interval,
            enhanced=recalculate_intervals,
        )
    if nested_column_band is not None:
        nested_lanes, band_top, band_bottom = nested_column_band
        nested_ordered_lanes = sorted(nested_lanes, key=lambda lane: lane.left)
        fallback_lane = _TextLane(
            left=filtered_intervals[0][0],
            right=filtered_intervals[0][1],
        )
        span_lines: list[tuple[_LineItem, BBox]] = []
        destinations = _lane_assignments_native(line_geometry, nested_lanes, anchor_tolerance, (band_top, band_bottom))
        if destinations is not None:
            for item, target in zip(line_geometry, destinations):
                if target is None:
                    span_lines.append(item)
                elif target == len(nested_lanes):
                    fallback_lane.lines.append(item)
                else:
                    nested_lanes[target].lines.append(item)
        else:
            for item in line_geometry:
                line, bbox = item
                center_y = _bbox_center_y(bbox)
                if (
                    line.semantic_type in {"header", "footer", "page_number", "page_footnote", "aside_text"}
                    or not band_top <= center_y <= band_bottom
                ):
                    fallback_lane.lines.append(item)
                    continue
                best_lane, best_coverage, second_coverage = _best_lane_coverage(
                    bbox,
                    nested_lanes,
                )
                fits_only_one_lane = _fits_only_one_lane_ordered(
                    bbox,
                    best_lane,
                    nested_ordered_lanes,
                    anchor_tolerance,
                )
                if second_coverage >= 0.2 and not fits_only_one_lane:
                    span_lines.append(item)
                    continue
                if best_coverage >= 0.5 or fits_only_one_lane:
                    best_lane.lines.append(item)
                else:
                    span_lines.append(item)
        lanes = [lane for lane in nested_lanes if lane.lines]
        if recalculate_intervals:
            _expand_nested_lane_intervals_from_members(
                lanes,
                anchor_tolerance,
            )
        if fallback_lane.lines:
            lanes.append(fallback_lane)
        if span_lines:
            lanes.append(
                _TextLane(
                    left=min(item[1][0] for item in span_lines),
                    right=max(item[1][2] for item in span_lines),
                    lines=span_lines,
                    is_span=True,
                )
            )
        _reattach_span_lane_continuations(lanes, median_height)
        _reattach_cross_lane_short_tails(lanes, median_height)
        return lanes

    lanes = [_TextLane(left=left, right=right) for left, right, _support in filtered_intervals]
    ordered_lanes = sorted(lanes, key=lambda lane: lane.left)
    span_lines: list[tuple[_LineItem, BBox]] = []
    destinations = _lane_assignments_native(line_geometry, lanes, anchor_tolerance)
    if destinations is not None:
        for item, target in zip(line_geometry, destinations):
            if target is None:
                span_lines.append(item)
            else:
                lanes[target].lines.append(item)
    else:
        for item in line_geometry:
            bbox = item[1]
            best_lane, best_coverage, second_coverage = _best_lane_coverage(bbox, lanes)
            fits_only_one_lane = _fits_only_one_lane_ordered(
                bbox,
                best_lane,
                ordered_lanes,
                anchor_tolerance,
            )
            # 同时覆盖两个稳定正文栏的行仍属于跨栏内容；只进入单侧栏且未越过栏沟的
            # 宽正文行则回到该栏，避免窄图注把正文错误挤入 span lane。
            if second_coverage >= 0.2 and not fits_only_one_lane:
                span_lines.append(item)
                continue
            if len(lanes) == 1 or fits_only_one_lane:
                best_lane.lines.append(item)
            else:
                span_lines.append(item)

    if recalculate_intervals:
        _expand_nested_lane_intervals_from_members(
            lanes,
            anchor_tolerance,
        )
    if span_lines:
        lanes.append(
            _TextLane(
                left=min(item[1][0] for item in span_lines),
                right=max(item[1][2] for item in span_lines),
                lines=span_lines,
                is_span=True,
            )
        )
    _reattach_span_lane_continuations(lanes, median_height)
    _reattach_cross_lane_short_tails(lanes, median_height)
    return lanes


def _best_lane_coverage(
    bbox: BBox,
    lanes: list[_TextLane],
) -> tuple[_TextLane, float, float]:
    """一次遍历求最佳栏覆盖与次高覆盖，平局保留原有首个最佳栏。"""

    line_width = max(0.1, bbox[2] - bbox[0])
    best_lane = lanes[0]
    best_coverage = -1.0
    second_coverage = -1.0
    for lane in lanes:
        coverage = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left)) / line_width
        if coverage > best_coverage:
            second_coverage = best_coverage
            best_coverage = coverage
            best_lane = lane
        elif coverage > second_coverage:
            second_coverage = coverage
    return best_lane, best_coverage, second_coverage


def _fits_only_one_lane_ordered(
    bbox: BBox,
    best_lane: _TextLane,
    ordered: list[_TextLane],
    tolerance: float,
) -> bool:
    """在调用方预排序的栏序列上判断宽行是否只停留在一个栏内。"""

    lane_index = ordered.index(best_lane)
    if lane_index > 0 and bbox[0] < ordered[lane_index - 1].right - tolerance:
        return False
    if lane_index + 1 < len(ordered) and bbox[2] > ordered[lane_index + 1].left - 0.25 * tolerance:
        return False
    return best_lane.left - tolerance <= _bbox_center_x(bbox) <= best_lane.right + max(tolerance, bbox[2] - best_lane.right)


def _fits_only_one_lane(
    bbox: BBox,
    best_lane: _TextLane,
    lanes: list[_TextLane],
    tolerance: float,
) -> bool:
    """保留旧内部入口；独立调用仍自行排序，输出与预排序版完全一致。"""

    return _fits_only_one_lane_ordered(
        bbox,
        best_lane,
        sorted(lanes, key=lambda lane: lane.left),
        tolerance,
    )


_LANE_BATCH_REFERENCE_FUNCTIONS = (
    _line_effective_height,
    _line_style_scale,
    _best_lane_coverage,
    _fits_only_one_lane_ordered,
    _bbox_center_x,
    _bbox_center_y,
)


def _expand_nested_lane_intervals_from_members(
    lanes: list[_TextLane],
    tolerance: float,
) -> None:
    """按已归属成员扩展局部栏边界，并在相邻栏相交前保留稳定栏沟。"""

    for lane in lanes:
        alignment_tolerance = max(1.0, 0.5 * tolerance)
        body_members = [bbox for line, bbox in lane.lines if line.semantic_type is None]
        if not body_members:
            continue
        aligned_members = [
            bbox
            for bbox in body_members
            if abs(bbox[0] - lane.left) <= alignment_tolerance or abs(bbox[2] - lane.right) <= alignment_tolerance
        ]
        if not aligned_members:
            continue
        # 页眉、页码和标题不参与；同时只让至少一侧锚点稳定的正文扩张栏宽，
        # 避免页面后续另一版式区段把当前局部栏整体拉宽。
        lane.left = min(lane.left, min(bbox[0] for bbox in aligned_members))
        lane.right = max(lane.right, max(bbox[2] for bbox in aligned_members))
    ordered = sorted(lanes, key=lambda lane: lane.left)
    for left_lane, right_lane in zip(ordered, ordered[1:]):
        if left_lane.right < right_lane.left - tolerance:
            continue
        midpoint = (left_lane.right + right_lane.left) / 2.0
        left_lane.right = min(left_lane.right, midpoint)
        right_lane.left = max(right_lane.left, midpoint)


def _infer_nested_column_band(
    line_geometry: list[tuple[_LineItem, BBox]],
    local_page_width: float,
    median_height: float,
    outer_interval: tuple[float, float],
    *,
    enhanced: bool = False,
) -> tuple[list[_TextLane], float, float] | None:
    """在全宽版心内查找仅占局部纵向区间的并列正文栏。"""

    outer_width = max(0.1, outer_interval[1] - outer_interval[0])
    candidates = [
        item
        for item in line_geometry
        if item[0].semantic_type is None
        and max(
            4.0 * _line_effective_height(*item),
            0.12 * local_page_width,
        )
        <= item[1][2] - item[1][0]
        <= 0.62 * outer_width
    ]
    if len(candidates) < 6:
        return None

    center_tolerance = max(2.0 * median_height, 0.06 * local_page_width)
    center_clusters: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(candidates, key=lambda value: _bbox_center_x(value[1])):
        center = _bbox_center_x(item[1])
        target = next(
            (
                cluster
                for cluster in center_clusters
                if abs(center - statistics.median(_bbox_center_x(member[1]) for member in cluster)) <= center_tolerance
            ),
            None,
        )
        if target is None:
            center_clusters.append([item])
        else:
            target.append(item)

    supported = [cluster for cluster in center_clusters if len(cluster) >= 3]
    supported.sort(key=lambda cluster: statistics.median(_bbox_center_x(item[1]) for item in cluster))
    best_pair: (
        tuple[
            tuple[int, float, float],
            list[tuple[_LineItem, BBox]],
            list[tuple[_LineItem, BBox]],
            tuple[float, float],
            tuple[float, float],
        ]
        | None
    ) = None
    cluster_pairs = (
        [
            (left_cluster, right_cluster)
            for left_index, left_cluster in enumerate(supported[:-1])
            for right_cluster in supported[left_index + 1 :]
        ]
        if enhanced
        else list(zip(supported, supported[1:]))
    )
    for left_cluster, right_cluster in cluster_pairs:
        left_interval = (
            statistics.median(item[1][0] for item in left_cluster),
            statistics.median(item[1][2] for item in left_cluster),
        )
        right_interval = (
            statistics.median(item[1][0] for item in right_cluster),
            statistics.median(item[1][2] for item in right_cluster),
        )
        gutter = right_interval[0] - left_interval[1]
        common_top = max(
            min(item[1][1] for item in left_cluster),
            min(item[1][1] for item in right_cluster),
        )
        common_bottom = min(
            max(item[1][3] for item in left_cluster),
            max(item[1][3] for item in right_cluster),
        )
        combined_width = right_interval[1] - left_interval[0]
        if (
            gutter < max(6.0, 0.75 * median_height)
            or common_bottom - common_top < 2.0 * median_height
            or combined_width < 0.55 * outer_width
        ):
            continue
        score = (
            min(len(left_cluster), len(right_cluster)),
            common_bottom - common_top,
            gutter,
        )
        candidate_pair = (
            score,
            left_cluster,
            right_cluster,
            left_interval,
            right_interval,
        )
        if best_pair is None or candidate_pair[0] > best_pair[0]:
            best_pair = candidate_pair
    if best_pair is None:
        return None

    _score, left_cluster, right_cluster, left_interval, right_interval = best_pair
    band_top = min(item[1][1] for item in [*left_cluster, *right_cluster]) - 0.5 * median_height
    band_bottom = max(item[1][3] for item in [*left_cluster, *right_cluster]) + 0.5 * median_height
    return (
        [
            _TextLane(left=left_interval[0], right=left_interval[1]),
            _TextLane(left=right_interval[0], right=right_interval[1]),
        ],
        band_top,
        band_bottom,
    )


def _reattach_span_lane_continuations(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把紧随稳定跨栏多行之后的单栏宽短尾行迁回对应 span lane。"""

    regular_lanes = [lane for lane in lanes if not lane.is_span]
    span_lanes = [lane for lane in lanes if lane.is_span]
    if len(regular_lanes) < 2 or not span_lanes:
        return

    for span_lane in span_lanes:
        _reattach_repeated_indented_span_tails(
            span_lane,
            regular_lanes,
            median_height,
        )
        while True:
            span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            candidates: list[
                tuple[
                    float,
                    float,
                    _TextLane,
                    tuple[_LineItem, BBox],
                ]
            ] = []
            for regular_lane in regular_lanes:
                for candidate in regular_lane.lines:
                    candidate_line, candidate_bbox = candidate
                    preceding = [
                        item
                        for item in span_lane.lines
                        if item[0].semantic_type == candidate_line.semantic_type and item[1][1] < candidate_bbox[1]
                    ]
                    if len(preceding) < 2:
                        continue
                    previous, last = preceding[-2:]
                    previous_height = _line_effective_height(*previous)
                    last_height = _line_effective_height(*last)
                    if (
                        abs(previous[1][0] - last[1][0]) > 0.75 * median_height
                        or max(previous_height, last_height) / min(previous_height, last_height) > 1.35
                        or not _title_fonts_compatible(previous[0], last[0])
                        or not -0.25 * median_height <= _effective_text_row_gap(previous, last) <= 0.75 * median_height
                    ):
                        continue
                    gap = _effective_text_row_gap(last, candidate)
                    candidate_height = _line_effective_height(*candidate)
                    if (
                        not -0.25 * median_height <= gap <= 0.75 * median_height
                        or abs(candidate_bbox[0] - last[1][0]) > 0.75 * median_height
                        or max(last_height, candidate_height) / min(last_height, candidate_height) > 1.35
                        or not _title_fonts_compatible(last[0], candidate_line)
                    ):
                        continue
                    has_parallel_peer = any(
                        other_line is not candidate_line
                        and _bbox_axis_overlap_ratio(
                            candidate_bbox,
                            other_bbox,
                            axis="y",
                        )
                        >= 0.5
                        for lane in regular_lanes
                        for other_line, other_bbox in lane.lines
                    )
                    if has_parallel_peer:
                        continue
                    candidates.append(
                        (
                            candidate_bbox[1],
                            max(0.0, gap),
                            regular_lane,
                            candidate,
                        )
                    )
            if not candidates:
                break
            _top, _gap, regular_lane, candidate = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            regular_lane.lines.remove(candidate)
            span_lane.lines.append(candidate)
            span_lane.left = min(span_lane.left, candidate[1][0])
            span_lane.right = max(span_lane.right, candidate[1][2])
            span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))


def _lane_accepts_short_tail(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """用同一组前序行证据判断当前栏或其他栏是否能够接纳正文短尾。"""

    candidate_line, candidate_bbox = candidate
    preceding = [
        item for item in lane.lines if item[0].semantic_type == candidate_line.semantic_type and item[1][1] < candidate_bbox[1]
    ]
    if not preceding:
        return False
    # 同位置前序行按栏内稳定排序取 source_index 较小者，避免后续插入触发排序后改变已用证据。
    previous = max(preceding, key=lambda item: (item[1][1], item[1][0], -item[0].source_index))
    return _short_tail_accepts_previous(candidate, lane, median_height, previous)


def _short_tail_accepts_previous(candidate, lane, median_height, previous) -> bool:
    """复用已确定的原规则前序行，保持字体、间距和视觉行门槛不变。"""
    candidate_line, candidate_bbox = candidate
    previous_line, previous_bbox = previous
    pair_height = max(
        _line_effective_height(*previous),
        _line_effective_height(*candidate),
        median_height,
    )
    lane_width = max(0.1, lane.right - lane.left)
    if (
        previous_bbox[2] - previous_bbox[0] < 0.65 * lane_width
        or candidate_bbox[2] - candidate_bbox[0] > 0.85 * lane_width
        or candidate_bbox[0] < lane.left - 0.75 * pair_height
        or candidate_bbox[2] > lane.right + 0.75 * pair_height
        or abs(candidate_bbox[0] - previous_bbox[0]) > 0.75 * pair_height
        or not _title_fonts_compatible(previous_line, candidate_line)
    ):
        return False
    gap = _effective_body_text_row_gap(previous, candidate)
    if not -0.25 * pair_height <= gap <= 0.9 * pair_height:
        return False
    if (
        previous_line.visual_row_id is not None
        and candidate_line.visual_row_id is not None
        and not 0 < candidate_line.visual_row_id - previous_line.visual_row_id <= 2
    ):
        return False
    return True


_TAIL_REFERENCE_FUNCTIONS = (
    _line_effective_height,
    _line_style_scale,
    _short_tail_accepts_previous,
    _title_fonts_compatible,
    _effective_body_text_row_gap,
    _font_weights_conflict,
)


def _reattach_cross_lane_short_tails(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """按纵向事件复用各栏最近前序行；特殊输入交给原逐次扫描实现。"""
    # 只有一个栏带时不存在“跨栏”归属，直接跳过输入校验和前序扫描；
    # 这不改变成员、顺序或边界，也能让常见单栏页面避开整页重复遍历。
    if len(lanes) < 2:
        return
    if _short_tail_lane_events_native(lanes, median_height):
        return
    pending = []
    seen_lines, seen_indices = set(), set()
    for lane_index, lane in enumerate(lanes):
        for ordinal, row in enumerate(lane.lines):
            line, bbox = row
            if (
                type(line) is not _LineItem
                or type(line.source_index) is not int
                or id(line) in seen_lines
                or line.source_index in seen_indices
                or type(bbox) not in (tuple, list)
                or len(bbox) != 4
                or (line.semantic_type is not None and type(line.semantic_type) is not str)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in bbox)
            ):
                return _reattach_cross_lane_short_tails_python(lanes, median_height)
            seen_lines.add(id(line))
            seen_indices.add(line.source_index)
            if line.semantic_type is None:
                pending.append((lane_index, row, ordinal))
    pending.sort(key=lambda item: (item[1][1][1], item[1][1][0], item[1][0].source_index))
    if _short_tail_native(lanes, pending, median_height):
        return
    previous = [None] * len(lanes)
    previous_keys = [None] * len(lanes)
    append_ordinal = sum(len(lane.lines) for lane in lanes)
    position = 0
    while position < len(pending):
        end = position + 1
        top = pending[position][1][1][1]
        while end < len(pending) and pending[end][1][1][1] == top:
            end += 1
        settled = []
        for source, candidate, ordinal in pending[position:end]:
            matches = [
                index
                for index, lane in enumerate(lanes)
                if previous[index] is not None and _short_tail_accepts_previous(candidate, lane, median_height, previous[index])
            ]
            target = matches[0] if len(matches) == 1 else source
            if target != source:
                lanes[source].lines.remove(candidate)
                lanes[target].lines.append(candidate)
                lanes[target].lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
                ordinal = append_ordinal
                append_ordinal += 1
            settled.append((target, candidate, ordinal))
        # 严格较低 y 才能成为下一组的前序行，同高候选不互相影响。
        for target, row, ordinal in settled:
            key = (row[1][1], row[1][0], -row[0].source_index, -ordinal)
            if previous_keys[target] is None or key > previous_keys[target]:
                previous_keys[target], previous[target] = key, row
        position = end


def _short_tail_lane_events_native(lanes, median_height):
    """验证和几何快照一并交给 Rust；完成快照后只迁移原始行对象。"""
    native = get_native()
    if native is None or any(
        actual is not expected
        for actual, expected in zip(
            (
                _line_effective_height,
                _line_style_scale,
                _short_tail_accepts_previous,
                _title_fonts_compatible,
                _effective_body_text_row_gap,
                _font_weights_conflict,
            ),
            _TAIL_REFERENCE_FUNCTIONS,
        )
    ):
        return False
    moves = native.short_tail_lane_events(lanes, median_height, _TextLane, _LineItem)
    if moves is None:
        return False
    for source, row, target in moves:
        lanes[source].lines.remove(row)
        lanes[target].lines.append(row)
        lanes[target].lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    return True


def _short_tail_native(lanes, pending, median_height):
    """预计算每行字体和尺度后批量结算事件；特殊数值维持既有 Python 遍历。"""
    native = get_native()
    if native is None:
        return False
    if (
        _line_effective_height is not _TAIL_REFERENCE_FUNCTIONS[0]
        or _line_style_scale is not _TAIL_REFERENCE_FUNCTIONS[1]
        or _short_tail_accepts_previous is not _TAIL_REFERENCE_FUNCTIONS[2]
        or _title_fonts_compatible is not _TAIL_REFERENCE_FUNCTIONS[3]
        or _effective_body_text_row_gap is not _TAIL_REFERENCE_FUNCTIONS[4]
        or _font_weights_conflict is not _TAIL_REFERENCE_FUNCTIONS[5]
    ):
        return False
    bounds = [(lane.left, lane.right) for lane in lanes]
    moves = native.short_tail_destinations_owned(pending, bounds, median_height)
    if moves is None:
        return False
    for index, target in moves:
        source, row, _ordinal = pending[index]
        lanes[source].lines.remove(row)
        lanes[target].lines.append(row)
        lanes[target].lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    return True


def _reattach_cross_lane_short_tails_python(
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """按前序依赖顺序，把具有唯一栏归属的正文短尾迁回对应栏。"""

    # 栏边界固定，且匹配只依赖 y0 严格更小的行；先确定上方行后，每行至多迁移一次。
    # 保留原始对象及其来源栏，不能用 source_index 合并或跳过不同的文本行。
    pending = [(lane, candidate) for lane in lanes for candidate in lane.lines if candidate[0].semantic_type is None]
    pending.sort(key=lambda item: (item[1][1][1], item[1][1][0], item[1][0].source_index))
    for source_lane, candidate in pending:
        # 当前栏也必须参与唯一性判断；多个栏均能接纳时保留原位，避免双向迁移。
        matches = [lane for lane in lanes if _lane_accepts_short_tail(candidate, lane, median_height)]
        if len(matches) != 1 or matches[0] is source_lane:
            continue
        target_lane = matches[0]
        source_lane.lines.remove(candidate)
        target_lane.lines.append(candidate)
        target_lane.lines.sort(
            key=lambda item: (item[1][1], item[1][0], item[0].source_index),
        )


def _reattach_repeated_indented_span_tails(
    span_lane: _TextLane,
    regular_lanes: list[_TextLane],
    median_height: float,
) -> None:
    """识别重复的跨栏首行与缩进短尾，并把短尾统一迁回跨栏栏带。"""

    span_rows = sorted(
        span_lane.lines,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    )
    matches: list[tuple[_TextLane, tuple[_LineItem, BBox], tuple[_LineItem, BBox], float]] = []
    for span_index, span_row in enumerate(span_rows):
        span_line, span_bbox = span_row
        next_span_top = span_rows[span_index + 1][1][1] if span_index + 1 < len(span_rows) else float("inf")
        for regular_lane in regular_lanes:
            for candidate in regular_lane.lines:
                candidate_line, candidate_bbox = candidate
                if candidate_line.semantic_type != span_line.semantic_type:
                    continue
                gap = _effective_text_row_gap(span_row, candidate)
                indent = candidate_bbox[0] - span_bbox[0]
                span_height = _line_effective_height(*span_row)
                candidate_height = _line_effective_height(*candidate)
                if (
                    candidate_bbox[1] <= span_bbox[1]
                    or candidate_bbox[1] >= next_span_top
                    or not -0.25 * median_height <= gap <= 0.75 * median_height
                    or not 0.75 * median_height <= indent <= 6.0 * median_height
                    or max(span_height, candidate_height) / min(span_height, candidate_height) > 1.35
                    or not _title_fonts_compatible(span_line, candidate_line)
                ):
                    continue
                has_parallel_peer = any(
                    other_line is not candidate_line and _bbox_axis_overlap_ratio(candidate_bbox, other_bbox, axis="y") >= 0.5
                    for lane in regular_lanes
                    for other_line, other_bbox in lane.lines
                )
                if not has_parallel_peer:
                    matches.append((regular_lane, candidate, span_row, indent))

    if len(matches) < 2:
        return
    median_indent = statistics.median(match[3] for match in matches)
    supported = [match for match in matches if abs(match[3] - median_indent) <= max(0.75 * median_height, 0.25 * median_indent)]
    if len(supported) < 2:
        return
    for regular_lane, candidate, _span_row, _indent in supported:
        if candidate not in regular_lane.lines:
            continue
        regular_lane.lines.remove(candidate)
        span_lane.lines.append(candidate)
        span_lane.left = min(span_lane.left, candidate[1][0])
        span_lane.right = max(span_lane.right, candidate[1][2])
    span_lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))


def _estimate_lane_gap(lane: _TextLane) -> tuple[float, float]:
    """仅为本次行距估计提取一次行高，排序仍作用于原栏带成员列表。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None:
        return _estimate_lane_gap_python(lane)
    lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    result = native.lane_gap(
        [bbox for _line, bbox in lane.lines],
        [_line_effective_height(line, bbox) for line, bbox in lane.lines],
        [line.restored_inline_cluster for line, _bbox in lane.lines],
        [
            previous[0].visual_row_id == current[0].visual_row_id and (previous[0].split_from_row or current[0].split_from_row)
            for previous, current in zip(lane.lines, lane.lines[1:])
        ],
    )
    return result if result is not None else _estimate_lane_gap_python(lane)


def _estimate_lane_gap_python(lane: _TextLane) -> tuple[float, float]:
    """从栏带内兼容相邻行的较小间隙簇估计常规净空和 MAD。"""

    lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
    heights = [_line_effective_height(line, bbox) for line, bbox in lane.lines]
    median_height = statistics.median(heights) if heights else 1.0
    gaps: list[float] = []
    for previous, current in zip(lane.lines, lane.lines[1:]):
        previous_line, previous_bbox = previous
        current_line, current_bbox = current
        if previous_line.visual_row_id == current_line.visual_row_id and (
            previous_line.split_from_row or current_line.split_from_row
        ):
            continue
        previous_height = _line_effective_height(previous_line, previous_bbox)
        current_height = _line_effective_height(current_line, current_bbox)
        if max(previous_height, current_height) / min(previous_height, current_height) > 1.35:
            continue
        pair_height = max(previous_height, current_height)
        gap = _effective_text_row_gap(previous, current)
        if gap < -0.25 * pair_height or gap > 2.0 * pair_height:
            continue
        if (
            _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5
            and abs(previous_bbox[0] - current_bbox[0]) > 1.5 * median_height
        ):
            continue
        # PDF 字符框常在相邻基线间产生极小重叠；按零净空计入常规行距统计。
        gaps.append(max(0.0, gap))

    if not gaps:
        return 0.35 * median_height, 0.0
    sorted_gaps = sorted(gaps)
    lower_count = max(1, math.ceil(len(sorted_gaps) * 0.6))
    lower_gaps = sorted_gaps[:lower_count]
    regular_gap = statistics.median(lower_gaps)
    gap_mad = statistics.median(abs(gap - regular_gap) for gap in lower_gaps)
    return regular_gap, gap_mad


def _is_structural_typography_gap(
    previous_height: float,
    current_height: float,
    vertical_gap: float,
    regular_gap: float,
    gap_mad: float,
    *,
    reliable_style_change: bool = False,
) -> bool:
    """判断异常段间净空是否同时具有行高或可靠字体层级变化。"""

    pair_height = max(previous_height, current_height)
    minimum_height = max(0.1, min(previous_height, current_height))
    prominent_gap = vertical_gap > regular_gap + max(
        0.75 * pair_height,
        3.0 * gap_mad,
    )
    return prominent_gap and (pair_height / minimum_height >= 1.12 or reliable_style_change)


def _should_connect_text_rows(
    previous: tuple[_LineItem, BBox],
    current: tuple[_LineItem, BBox],
    lane: _TextLane,
    regular_gap: float,
    gap_mad: float,
    table_bboxes: list[BBox],
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """综合局部间距、首行缩进、字体和障碍判断两个相邻视觉行是否同段。"""

    previous_line, previous_bbox = previous
    current_line, current_bbox = current
    previous_height = _line_effective_height(previous_line, previous_bbox)
    current_height = _line_effective_height(current_line, current_bbox)
    pair_height = max(previous_height, current_height)
    lane_width = max(0.1, lane.right - lane.left)
    previous_width = previous_bbox[2] - previous_bbox[0]
    current_width = current_bbox[2] - current_bbox[0]
    vertical_gap = _effective_body_text_row_gap(previous, current)
    if previous_line.visual_row_id == current_line.visual_row_id and (
        previous_line.split_from_row or current_line.split_from_row
    ):
        return False
    if current_height < 0.88 * previous_height and vertical_gap > regular_gap + max(0.25 * previous_height, 3.0 * gap_mad):
        return False
    both_fill_lane = previous_width >= 0.8 * lane_width and current_width >= 0.8 * lane_width
    aligned_left_edges = abs(previous_bbox[0] - current_bbox[0]) <= 0.5 * pair_height
    current_returns_to_lane_left = (
        abs(current_bbox[0] - lane.left) <= 0.75 * pair_height
        and -0.5 * pair_height <= previous_bbox[0] - lane.left <= 2.0 * pair_height
    )
    before_cjk, after_cjk = previous_line.native_cjk_style, current_line.native_cjk_style
    mixed_cjk_continuation = (
        before_cjk is not None
        and after_cjk is not None
        and before_cjk[:3] == after_cjk[:3]
        and before_cjk[3] > 0
        and 0.9 <= after_cjk[3] / before_cjk[3] <= 1.1
        and 1.25 <= max(previous_height, current_height) / min(previous_height, current_height) <= 1.45
        and previous_width >= 0.8 * lane_width
        and current_width >= 0.5 * lane_width
        and current_returns_to_lane_left
        and not previous_line.paragraph_terminal
        and not _font_weights_conflict(previous_line, current_line)
        and -0.25 * pair_height <= vertical_gap <= regular_gap + 0.35 * pair_height
    )
    reliable_font_match = (
        previous_line.font_signature is None
        or current_line.font_signature is None
        or previous_line.font_coverage < 0.75
        or current_line.font_coverage < 0.75
        or previous_line.font_signature == current_line.font_signature
        or _font_signatures_share_family(
            previous_line.font_signature,
            current_line.font_signature,
        )
        or (current_width <= 0.5 * lane_width and previous_line.font_signature[1] == current_line.font_signature[1])
    )
    previous_indent = previous_bbox[0] - lane.left
    # 极端字体矩阵校正后，旧字框高度不再夸大缩进；真实同行左缘与常规基线间距仍约束续行。
    local_metric_pair = all(
        line.style_scale_repaired
        and line.source_bbox is not None
        and line.source_bbox[3] - line.source_bbox[1] >= 4 * _line_effective_height(line, bbox)
        for line, bbox in (previous, current)
    )
    repeated_indent_continuation = (
        previous_indent >= max(5.0, (1.25 if local_metric_pair else 1.8) * pair_height)
        and abs(current_bbox[0] - previous_bbox[0]) <= 0.5 * pair_height
        and reliable_font_match
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.75 * pair_height, 3.0 * gap_mad)
    )
    # 逗号未终止的满行正文可接同字体家族的粗体枚举短尾；真实标题、留白和字号变化仍拒绝。
    emphasized_enumeration_tail = (
        previous_line.semantic_type is None
        and current_line.semantic_type is None
        and previous_line.text.rstrip().endswith((",", "，"))
        and (current_line.paragraph_terminal or bool(re.search(r"[.!?。！？]$", current_line.text.rstrip())))
        and len(current_line.text.split()) <= 10
        and previous_line.font_signature is not None
        and current_line.font_signature is not None
        and _font_signatures_share_emphasis_family(previous_line.font_signature, current_line.font_signature)
        and max(previous_height, current_height) <= 1.2 * min(previous_height, current_height)
        and -0.1 * pair_height <= vertical_gap <= 0.4 * pair_height
    )
    # 未收句的满栏正文后，连词或小写词开头的短斜体收句仍是正文续行；独立大写标签和公式不适用。
    italic_sentence_tail = (
        previous_line.semantic_type is None
        and current_line.semantic_type is None
        and not previous_line.paragraph_terminal
        and re.search(r"[.!?。！？:：;；]$", previous_line.text.rstrip()) is None
        and not previous_line.formula_candidate_only
        and not current_line.formula_candidate_only
        and not current_line.compact_formula_cluster
        and (current_line.paragraph_terminal or bool(re.search(r"[.!?。！？]$", current_line.text.rstrip())))
        and re.match(r"(?:[+&]\s+|[a-z])", current_line.text.strip()) is not None
        and 1 <= len(re.findall(r"[A-Za-z]{2,}", current_line.text)) <= 10
        and previous_line.font_signature is not None
        and current_line.font_signature is not None
        and current_line.font_signature[1] & PDF_FONT_ITALIC_FLAG
        and not previous_line.font_signature[1] & PDF_FONT_ITALIC_FLAG
        and _font_signatures_share_emphasis_family(previous_line.font_signature, current_line.font_signature)
        and not _font_weights_conflict(previous_line, current_line)
        and max(previous_height, current_height) <= 1.2 * min(previous_height, current_height)
        and -0.1 * pair_height <= vertical_gap <= 0.4 * pair_height
    )
    safe_short_tail = (
        previous_width >= 0.75 * lane_width
        and current_width <= 0.7 * lane_width
        and (aligned_left_edges or current_returns_to_lane_left)
        and (reliable_font_match or emphasized_enumeration_tail or italic_sentence_tail)
        and (not _font_weights_conflict(previous_line, current_line) or emphasized_enumeration_tail)
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.75 * pair_height, 3.0 * gap_mad)
    )
    height_ratio = max(previous_height, current_height) / min(previous_height, current_height)
    font_style_changed = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_signature[1] != current_line.font_signature[1]
        and not _font_signatures_share_family(
            previous_line.font_signature,
            current_line.font_signature,
        )
    )
    reliable_style_conflict = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and (
            (
                previous_line.font_signature != current_line.font_signature
                and not _font_signatures_share_family(
                    previous_line.font_signature,
                    current_line.font_signature,
                )
            )
            or _font_weights_conflict(previous_line, current_line)
        )
    )
    fallback_font_continuation = (
        previous_line.font_signature is not None
        and current_line.font_signature is not None
        and previous_line.font_coverage >= 0.75
        and current_line.font_coverage >= 0.75
        and previous_line.font_signature[0] != current_line.font_signature[0]
        and previous_line.font_signature[1] == current_line.font_signature[1]
        and aligned_left_edges
        and height_ratio <= 1.25
        and not _font_weights_conflict(previous_line, current_line)
        and -0.25 * pair_height <= vertical_gap <= regular_gap + max(0.35 * pair_height, 3.0 * gap_mad)
    )
    if (
        font_style_changed
        and _font_weights_conflict(previous_line, current_line)
        and not fallback_font_continuation
        and not mixed_cjk_continuation
        and not emphasized_enumeration_tail
    ):
        # 显式样式位与显著字重同时变化仍是硬边界，不能被满栏几何放宽。
        return False
    if (
        not is_hyphen_at_line_end(previous_line.text)
        and _is_structural_typography_gap(
            previous_height,
            current_height,
            vertical_gap,
            regular_gap,
            gap_mad,
            reliable_style_change=reliable_style_conflict,
        )
        and not fallback_font_continuation
    ):
        # 图注到正文等排版层级转换即使同栏满行，也不能被常规续行规则重新吸收。
        if not mixed_cjk_continuation:
            return False
    full_width_continuation = (
        both_fill_lane
        and aligned_left_edges
        and not font_style_changed
        and vertical_gap <= regular_gap + max(0.75 * min(previous_height, current_height), 3.0 * gap_mad)
    )
    if height_ratio > 1.35 and not safe_short_tail and not full_width_continuation:
        # 满栏混合字体可跨字号续接，但显式正体/斜体等样式边界仍保持原分段语义。
        if (not both_fill_lane or not aligned_left_edges or font_style_changed) and not mixed_cjk_continuation:
            return False

    if vertical_gap < -0.25 * pair_height:
        return False
    if (
        _bbox_axis_overlap_ratio(previous_bbox, current_bbox, axis="x") < 0.5
        and abs(previous_bbox[0] - current_bbox[0]) > 1.5 * pair_height
        and not safe_short_tail
    ):
        return False
    if _connection_crosses_table(previous_line.bbox, current_line.bbox, table_bboxes):
        return False
    if _horizontal_rule_separates_rows(previous_bbox, current_bbox, lane, axis_lines):
        return False
    if mixed_cjk_continuation:
        # 同汉字字形尺度证明日期/英文缩写首行的 dominant font 变化不构成段界，首行缩进仍保留。
        return True

    gap_limit = max(
        regular_gap + max(0.5 * pair_height, 3.0 * gap_mad),
        1.1 * pair_height,
    )
    # 排版断词可以跳过缩进、字体和短行规则，但仍须限制在邻近物理行内，
    # 避免页内远距离的 “cross-” 与后续标题被误拼为同一段。
    if is_hyphen_at_line_end(previous_line.text):
        return vertical_gap <= max(gap_limit, 1.8 * pair_height)
    if vertical_gap > gap_limit:
        return False

    terminal_previous = bool(re.search(r"[.!?。！？:：;；][\]\)}）】》”’'\"]*$", previous_line.text.rstrip()))
    sparse_lane = sum(line.semantic_type is None for line, _bbox in lane.lines) <= 6
    if (
        terminal_previous
        and not repeated_indent_continuation
        and ((sparse_lane and vertical_gap > 0.65 * pair_height) or vertical_gap > regular_gap + 0.5 * pair_height)
    ):
        return False

    # 局部版心可能比整栏推断边界更靠左，缩进需同时参考上一物理行。
    local_lane_left = min(lane.left, previous_bbox[0])
    local_lane_width = max(0.1, lane.right - local_lane_left)
    next_indent = current_bbox[0] - local_lane_left
    previous_fill = max(0.0, previous_bbox[2] - local_lane_left) / local_lane_width
    if (
        next_indent >= max(5.0, 0.65 * pair_height)
        and (previous_fill <= 0.8 or terminal_previous)
        and not safe_short_tail
        and not repeated_indent_continuation
    ):
        # 已确认的同左缘短尾优先于栏左缘缩进，避免参考文献冒号后的末行被切断。
        return False

    abnormal_gap = vertical_gap > regular_gap + max(0.25 * pair_height, 3.0 * gap_mad)
    if (
        reliable_style_conflict
        and (abnormal_gap or min(previous_width, current_width) <= 0.7 * lane_width)
        and not both_fill_lane
        and not safe_short_tail
        and not fallback_font_continuation
    ):
        return False
    if abnormal_gap and min(previous_width, current_width) <= 0.65 * lane_width and not safe_short_tail:
        return False
    return True


def _horizontal_rule_separates_rows(
    previous_bbox: BBox,
    current_bbox: BBox,
    lane: _TextLane,
    axis_lines: list[_LocalAxisLine],
) -> bool:
    """检查两个相邻文本行之间是否存在覆盖当前栏带的长水平规则线。"""

    if current_bbox[1] <= previous_bbox[3]:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    for axis_line in axis_lines:
        if axis_line.orientation != "horizontal":
            continue
        line_y = _bbox_center_y(axis_line.bbox)
        if not previous_bbox[3] <= line_y <= current_bbox[1]:
            continue
        overlap = max(0.0, min(axis_line.bbox[2], lane.right) - max(axis_line.bbox[0], lane.left))
        if overlap / lane_width >= 0.6:
            return True
    return False


def _connection_crosses_table(
    first_bbox: BBox,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """检查两行中心连接区域是否穿过已确认表格。"""

    first_center = (_bbox_center_x(first_bbox), _bbox_center_y(first_bbox))
    second_center = (_bbox_center_x(second_bbox), _bbox_center_y(second_bbox))
    connector = _coerce_bbox(
        (
            min(first_center[0], second_center[0]) - 0.1,
            min(first_center[1], second_center[1]) - 0.1,
            max(first_center[0], second_center[0]) + 0.1,
            max(first_center[1], second_center[1]) + 0.1,
        )
    )
    return connector is not None and any(_bbox_intersects(connector, table_bbox) for table_bbox in table_bboxes)
