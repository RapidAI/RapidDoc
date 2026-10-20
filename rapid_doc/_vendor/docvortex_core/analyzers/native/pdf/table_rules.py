"""PDF 规则线与文本行候选；保留原有认领顺序与判定规则。"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import statistics
import math
import sys
from collections import OrderedDict
from typing import Any

from ....document.pdf._document import PDFPathInfo
from ....schema import BBox
from .geometry import (
    _bbox_area,
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _point_in_bbox,
    _rotate_bbox_from_upright,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from . import geometry as _fragment_geometry
from .models import (
    _Fragment,
    _LineItem,
    _LocalAxisLine,
    _PageSource,
    _PreparedPage,
    _SharedLineIndexSet,
    _TableCandidate,
    _TableContinuationGrid,
    _VisualRow,
)
from .table_annotations import (
    _PreparedTableNoteBodyMetrics,
    _PreparedTableCoreRows,
    _PreparedMarkerCache,
    _PreparedTableNoteRow,
    _build_table_annotation,
    _prepare_annotation_geometry,
    _collect_caption_rows,
    _collect_footnote_rows,
    _find_table_caption,
    _merge_table_candidate_annotations,
    _prepare_table_note_body_metrics,
    _prepare_table_note_rows,
    _prepare_table_core_rows,
    _prepare_marker_line_context,
)
from .table_rows import _clip_visual_row_to_corridor


@dataclass(slots=True)
class _RuleCorridorRow:
    """保存精确横向走廊裁剪后的行，以及原算法用于纵向准入的几何。"""

    source_bbox: BBox
    gate_center_y: float
    row: _VisualRow


@dataclass(slots=True)
class _RuleIntervalPrefix:
    """仅保留当前横线起点下的一份精确行序列和相邻区间分配。"""

    rows: tuple[_VisualRow, ...]
    rules: tuple[_LocalAxisLine, ...]
    groups: list[list[_VisualRow]]


@dataclass(slots=True)
class _RuleBandIndex:
    """保存同走廊的原行身份及其在完整横线组中的首个相邻区间。"""

    rows: tuple[_VisualRow, ...]
    positions: dict[int, int]
    bands: tuple[int, ...]
    on_rule: tuple[bool, ...]
    rule_count: int
    native: Any = None
    shared_core_rows: Any = None

    def interval(self, rows: list[_VisualRow]) -> tuple[int, int] | None:
        """只准入与原走廊完全相同、连续且保序的行引用。"""

        if not rows:
            return 0, 0
        start = self.positions.get(id(rows[0]))
        if start is None or start + len(rows) > len(self.rows):
            return None
        if any(row is not self.rows[start + offset] for offset, row in enumerate(rows)):
            return None
        return start, start + len(rows)


class _RuleIntervalGroups(list):
    """保留普通分组列表行为，并携带同一次原生扫描产生的区间文本证据。"""

    def __init__(self, groups, rows, rules, height, accepted, segments=None, short_intervals=False):
        """证据仅供原输入列表和同一行高查询复用，不缓存外部可变对象的跨调用状态。"""
        super().__init__(groups)
        self.source_rows = rows
        self.source_rules = rules
        self.height = height
        self.accepted = accepted
        self.segments = segments
        self.short_intervals = short_intervals


@dataclass(slots=True)
class _StableColumnPrefixState:
    """保存稳定列聚类的完整前缀状态，供严格前缀输入继续计算。"""

    row_ids: tuple[int, ...]
    clusters_by_alignment: Any


@dataclass(slots=True)
class _StableColumnCache:
    """保存稳定列精确结果与当前起点下可安全续算的前缀状态。"""

    results: dict[tuple[float, tuple[int, ...]], tuple[int, float]] = field(default_factory=dict)
    prefixes: dict[tuple[float, int], _StableColumnPrefixState] = field(default_factory=dict)
    prepared_rows: dict[int, tuple[_VisualRow, Any]] = field(default_factory=dict)
    owned_geometry: bool = False
    native_geometry: Any = None


@dataclass(slots=True)
class _RuleCandidateContext:
    """同一候选组共享只读输入和网格索引，避免每个区间复制页面成员。"""

    rows: list
    lines: list
    page_size: tuple
    angle: int
    median_height: float
    axis_lines: list
    excluded_bboxes: list
    marker_cache: dict
    body_metrics: _PreparedTableNoteBodyMetrics
    marker_prepared: _PreparedMarkerCache = field(default_factory=_PreparedMarkerCache)
    grids: list | None = None
    grid_components: list | None = None
    grid_members: dict = field(default_factory=dict)
    core_indexes: dict = field(default_factory=dict)
    marker_line_context: tuple[bool, dict] | None = None
    row_bounds: dict = field(default_factory=dict)
    annotation_geometry: Any = None


@dataclass(slots=True)
class _RuleCandidateDraft:
    """保存轻量候选描述，评分排序之后再创建注释及成员集合。"""

    context: _RuleCandidateContext
    boundaries: list
    rows: list
    caption: _LineItem | None
    note_rows: list
    has_notes: bool
    score: float
    core_query: Any = None
    prepared_candidate: Any = None

    def materialize(self) -> _TableCandidate:
        """按参考规则物化当前候选，并复用本组网格成员后立即交给合并器。"""
        context = self.context
        if context.annotation_geometry is None:
            context.annotation_geometry = _prepare_annotation_geometry(context.rows) or False
        candidate = _expand_rule_table_candidate(
            self.boundaries,
            self.rows,
            context.rows,
            context.lines,
            context.page_size,
            context.angle,
            context.median_height,
            self.caption,
            self.has_notes,
            context.marker_cache,
            self.note_rows,
            context.body_metrics,
            self.core_query,
            context.annotation_geometry or None,
            self.prepared_candidate,
        )
        candidate.score = self.score
        if context.grids is None:
            context.grids = [
                box
                for box in _connected_rule_grid_bboxes(
                    context.axis_lines,
                    context.median_height,
                    components=context.grid_components,
                )
                if not any(_bbox_overlap_in_smaller(box, excluded) >= 0.5 for excluded in context.excluded_bboxes)
            ]
        return _expand_candidates_to_connected_rule_grids(
            [candidate],
            context.rows,
            context.page_size,
            context.angle,
            context.median_height,
            context.axis_lines,
            context.excluded_bboxes,
            prepared_grid_bboxes=context.grids,
            grid_member_cache=context.grid_members,
        )[0]


def _build_fragments(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_Fragment]:
    """将精修后的原生 run 转换成表格单元候选。"""

    fragments: list[_Fragment] = []
    for line in lines:
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        fragments.append(
            _Fragment(
                text=line.text,
                bbox=line.bbox,
                local_bbox=local_bbox,
                line_index=line.source_index,
                # 复用原生粗行身份，避免同一字符行内不同 cell
                # 因轻微基线差异被误拆成多行。
                visual_row_id=line.visual_row_id,
            )
        )
    return fragments


_FRAGMENT_GROUP_REFERENCE_FUNCTIONS = (_bbox_center_y, statistics.fmean)
_RULE_MEAN_REFERENCE = statistics.mean


def _mean_rule_positions(values):
    """以精确整数比率批量求均值；非普通浮点、特殊值及替换的统计函数保留原路径。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is not None and statistics.mean is _RULE_MEAN_REFERENCE:
        result = native.exact_float_mean(values)
        if result is not None:
            return result
    return statistics.mean(values)


_FRAGMENT_ROW_REFERENCES = (
    _fragment_geometry._bbox_union,
    _bbox_union,
    _bbox_union_many,
    _Fragment.__getattribute__,
    _VisualRow,
    sum,
)


def _cluster_fragment_rows(fragments, median_height):
    """仅对普通片段使用批量分组，特殊对象和数值完整保留参考路径。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None or (
        _bbox_center_y is not _FRAGMENT_GROUP_REFERENCE_FUNCTIONS[0]
        or statistics.fmean is not _FRAGMENT_GROUP_REFERENCE_FUNCTIONS[1]
        or type(median_height) not in (int, float)
        or not math.isfinite(median_height)
    ):
        return _cluster_fragment_rows_python(fragments, median_height)
    if (
        _fragment_geometry._bbox_union,
        _bbox_union,
        _bbox_union_many,
        _Fragment.__getattribute__,
        _VisualRow,
        sum,
    ) == _FRAGMENT_ROW_REFERENCES:
        rows = native.fragment_rows_owned(fragments, median_height, _Fragment, _VisualRow, statistics.fmean, sum)
        if rows is not None:
            return rows
    items = []
    for fragment in fragments:
        box = fragment.local_bbox
        if (
            type(fragment) is not _Fragment
            or type(box) not in (tuple, list)
            or len(box) != 4
            or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 2**50 for value in box)
            or (
                fragment.visual_row_id is not None
                and (type(fragment.visual_row_id) is not int or not -(2**63) <= fragment.visual_row_id < 2**63)
            )
        ):
            return _cluster_fragment_rows_python(fragments, median_height)
        items.append((_bbox_center_y(box), box[0], fragment.visual_row_id))
    indices = native.fragment_row_groups(items, max(2.0, median_height * 0.5), statistics.fmean)
    if indices is None:
        return _cluster_fragment_rows_python(fragments, median_height)
    return _materialize_fragment_rows([[fragments[i] for i in group] for group in indices])


def _cluster_fragment_rows_python(
    fragments: list[_Fragment],
    median_height: float,
) -> list[_VisualRow]:
    """优先复用原生视觉行身份，其余片段按中心线容差聚成表格行。"""

    tolerance = max(2.0, median_height * 0.5)
    native_groups: dict[int, list[_Fragment]] = {}
    geometric_fragments: list[_Fragment] = []
    for fragment in fragments:
        if fragment.visual_row_id is None:
            geometric_fragments.append(fragment)
        else:
            native_groups.setdefault(fragment.visual_row_id, []).append(fragment)

    # 先锁定同一原生粗行拆出的 run，再允许不同粗行按基线几何合并；
    # 旋转表格常把同一数据行的各 cell 分成多个 pdftext 粗行，不能只依赖 row id。
    seed_groups = [*native_groups.values(), *[[fragment] for fragment in geometric_fragments]]
    prepared_seed_groups = [
        (
            seed_group,
            statistics.fmean(_bbox_center_y(item.local_bbox) for item in seed_group),
            min(item.local_bbox[0] for item in seed_group),
        )
        for seed_group in seed_groups
    ]
    prepared_seed_groups.sort(key=lambda item: (item[1], item[2]))
    grouped: list[list[_Fragment]] = []
    group_centers: list[float] = []
    for seed_group, center_y, _left in prepared_seed_groups:
        target_index: int | None = None
        for group_index, group_center in enumerate(group_centers):
            if abs(center_y - group_center) <= tolerance:
                target_index = group_index
                break
        if target_index is None:
            grouped.append(list(seed_group))
            group_centers.append(center_y)
        else:
            target_group = grouped[target_index]
            target_group.extend(seed_group)
            # 仅在组成员变化时使用原有 fmean 重新计算，后续比较直接复用精确结果。
            group_centers[target_index] = statistics.fmean(_bbox_center_y(item.local_bbox) for item in target_group)

    return _materialize_fragment_rows(grouped)


def _materialize_fragment_rows(grouped):
    """在输出边界沿用原框合并、Python sum 与稳定排序，不能近似最终中心。"""
    rows: list[_VisualRow] = []
    for group in grouped:
        group.sort(key=lambda item: item.local_bbox[0])
        bbox = _bbox_union_many([item.local_bbox for item in group])
        visual_row_ids = {item.visual_row_id for item in group if item.visual_row_id is not None}
        rows.append(
            _VisualRow(
                fragments=group,
                center_y=sum(_bbox_center_y(item.local_bbox) for item in group) / len(group),
                bbox=bbox,
                visual_row_id=next(iter(visual_row_ids)) if len(visual_row_ids) == 1 else None,
            )
        )
    rows.sort(key=lambda row: row.center_y)
    return rows


def _build_rule_table_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    *,
    path_infos: list[PDFPathInfo] | None = None,
    excluded_bboxes: list[BBox] | None = None,
    caption_candidates: list[tuple[_LineItem, BBox]] | None = None,
    defer_materialization: bool = False,
) -> list[_TableCandidate] | list[_RuleCandidateDraft]:
    """枚举同跨度横线边界区间，再以连续多列文本分布确认表格。"""

    candidates: list[_TableCandidate] = []
    path_infos = path_infos or []
    excluded_bboxes = excluded_bboxes or []
    stable_column_cache = _StableColumnCache(owned_geometry=True)
    prepared_path_infos = _prepare_fill_band_infos(path_infos, page_size, angle)
    native_fill_bands = _prepare_native_fill_bands(prepared_path_infos)
    vertical_axis_lines = [line for line in axis_lines if line.orientation == "vertical"]
    corridor_cache: dict[tuple[float, float], list[_RuleCorridorRow]] = {}
    note_corridor_cache: dict[tuple[float, float], tuple[list[_PreparedTableNoteRow], bool]] = {}
    prepared_note_body_metrics = _prepare_table_note_body_metrics(
        lines,
        page_size,
        angle,
    )
    marker_cache: dict[tuple[int, str], bool] = {}
    drafts: list[_RuleCandidateDraft] = []
    context = _RuleCandidateContext(
        rows,
        lines,
        page_size,
        angle,
        median_height,
        axis_lines,
        excluded_bboxes,
        marker_cache,
        prepared_note_body_metrics,
    )
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        active_first_index = -1
        interval_prefix: _RuleIntervalPrefix | None = None
        band_indexes: OrderedDict[tuple[float, float], _RuleBandIndex | None] = OrderedDict()
        # 保留历史算法的候选集合与枚举顺序；性能优化只复用候选内部的纯计算结果。
        for first_index, bottom_index in _iter_rule_spans(len(rule_group)):
            if first_index != active_first_index:
                stable_column_cache.prefixes.clear()
                if stable_column_cache.native_geometry is not None:
                    stable_column_cache.native_geometry.clear_prefixes()
                interval_prefix = None
                active_first_index = first_index
            top_rule = rule_group[first_index]
            bottom_rule = rule_group[bottom_index]
            interval_rules = rule_group[first_index : bottom_index + 1]
            boundary_rules = [top_rule, bottom_rule]
            rule_bbox = _bbox_union_many([line.bbox for line in boundary_rules])
            core_rows = _rows_inside_rule_interval(
                rows,
                rule_bbox,
                excluded_bboxes,
                corridor_cache=corridor_cache,
            )
            caption_line = _find_table_caption(
                lines,
                rule_bbox,
                page_size,
                angle,
                median_height,
                caption_candidates,
            )
            caption_anchored_compact_grid = (
                caption_line is not None
                and len(interval_rules) >= 3
                and rule_bbox[3] - rule_bbox[1] <= 0.15 * (page_size[0] if angle in {90, 270} else page_size[1])
            )
            corridor_key = (rule_bbox[0], rule_bbox[2])
            if corridor_key not in band_indexes:
                corridor_rows = [item.row for item in corridor_cache[corridor_key]]
                band_indexes[corridor_key] = (
                    _prepare_rule_band_index(corridor_rows, rule_group)
                    if len(corridor_rows) >= 8 and len(rule_group) >= 4
                    else None
                )
                if len(band_indexes) > 16:
                    band_indexes.popitem(last=False)
            else:
                band_indexes.move_to_end(corridor_key)
            interval_row_groups = _partition_rows_by_prepared_bands(
                core_rows, interval_rules, first_index, band_indexes[corridor_key], median_height
            )
            if interval_row_groups is None:
                interval_row_groups, interval_prefix = _partition_rows_by_rule_intervals_cached(
                    core_rows, interval_rules, interval_prefix
                )
            if (
                not _every_rule_interval_has_multi_cell_row(
                    core_rows,
                    interval_rules,
                    median_height,
                    interval_row_groups,
                )
                and not caption_anchored_compact_grid
            ):
                continue
            if (
                not _rule_intervals_are_column_compatible(
                    core_rows,
                    interval_rules,
                    median_height,
                    stable_column_cache,
                    interval_row_groups,
                )
                and not caption_anchored_compact_grid
            ):
                continue

            # 框只属于本次只读构建；每个区间仍独立执行完整行带规则。
            fill_band_count = (
                native_fill_bands.count(rule_bbox, median_height)
                if native_fill_bands is not None and _fill_band_rules_unchanged()
                else None
            )
            if fill_band_count is None:
                fill_band_count = _count_repeated_fill_bands(
                    path_infos,
                    rule_bbox,
                    page_size,
                    angle,
                    median_height,
                    prepared_path_infos,
                )
            aligned_vertical_count = _count_aligned_vertical_rules(
                axis_lines,
                rule_bbox,
                median_height,
                vertical_axis_lines,
            )

            row_segments = (
                interval_row_groups.segments
                if type(interval_row_groups) is _RuleIntervalGroups
                and interval_row_groups.source_rows is core_rows
                and interval_row_groups.height == median_height
                and interval_row_groups.segments is not None
                and _continuous_table_row_segments is _CONTINUOUS_SEGMENTS_REFERENCE
                else _continuous_table_row_segments(core_rows, median_height)
            )
            accepted: tuple[list[_VisualRow], list[_VisualRow], int, float] | None = None
            for row_segment in row_segments:
                dense_rows = [row for row in row_segment if len(row.fragments) >= 2]
                compact_grid_columns = (
                    _compact_fully_ruled_grid_column_count(
                        row_segment,
                        dense_rows,
                        interval_rules,
                        axis_lines,
                        rule_bbox,
                        median_height,
                    )
                    if len(dense_rows) == 2
                    else 0
                )
                stable_columns, column_coverage = _count_stable_columns(
                    dense_rows,
                    median_height,
                    stable_column_cache,
                    allow_prefix_reuse=True,
                )
                if compact_grid_columns > 0:
                    # 两行样本容易把左右/中心锚点误算成不同稳定列，使用物理网格列数。
                    stable_columns = compact_grid_columns
                caption_supported_compact_rows = (
                    caption_anchored_compact_grid and len(dense_rows) >= 2 and stable_columns >= 3 and column_coverage >= 0.5
                )
                if len(dense_rows) < 3 and compact_grid_columns == 0 and not caption_supported_compact_rows:
                    continue
                # 真表格的多单元行会在整个数据带内反复出现；少数图题、图例和
                # 坐标刻度偶然形成的多列行不能支撑一大片正文区域。
                if len(dense_rows) / len(row_segment) < 0.2:
                    continue
                if stable_columns < 2 or column_coverage < 0.5:
                    continue
                if _looks_like_page_column_prose(
                    row_segment,
                    dense_rows,
                    stable_columns,
                    fill_band_count,
                    aligned_vertical_count,
                    rule_bbox,
                ):
                    continue
                if not _table_segment_reaches_boundaries(
                    row_segment,
                    rule_bbox,
                    median_height,
                ):
                    continue
                corridor_key = (rule_bbox[0], rule_bbox[2])
                if corridor_key not in context.row_bounds:
                    context.row_bounds[corridor_key] = _RuleRowBounds([item.row for item in corridor_cache[corridor_key]])
                if not _table_rows_align_with_rule_span(
                    row_segment,
                    rule_bbox,
                    median_height,
                    context.row_bounds[corridor_key],
                ):
                    continue
                result = (
                    row_segment,
                    dense_rows,
                    stable_columns,
                    column_coverage,
                )
                if accepted is None or (
                    len(row_segment),
                    len(dense_rows),
                    stable_columns,
                    column_coverage,
                ) > (
                    len(accepted[0]),
                    len(accepted[1]),
                    accepted[2],
                    accepted[3],
                ):
                    accepted = result
            if accepted is None:
                continue

            accepted_rows, dense_rows, stable_columns, _coverage = accepted
            corridor_key = (rule_bbox[0], rule_bbox[2])
            note_context = note_corridor_cache.get(corridor_key)
            if note_context is None:
                prepared_note_rows = _prepare_table_note_rows(
                    rows,
                    lines,
                    rule_bbox,
                    median_height,
                    page_size,
                    angle,
                )
                note_context = (
                    prepared_note_rows,
                    any(row.explicit_note or row.auxiliary_marker is not None for row in prepared_note_rows),
                )
                note_corridor_cache[corridor_key] = note_context
            prepared_note_rows, has_possible_table_notes = note_context
            score = float(2 + len(dense_rows) + stable_columns + min(fill_band_count, 8))
            if defer_materialization:
                band_index = band_indexes[corridor_key]
                if corridor_key not in context.core_indexes:
                    if context.marker_line_context is None:
                        context.marker_line_context = _prepare_marker_line_context(lines)
                    marker_safe, marker_source_lines = context.marker_line_context
                    # 构建阶段内部行序列冻结为 tuple，允许两个只读索引安全共享切片校验。
                    corridor_rows = (
                        band_index.rows if band_index is not None else tuple(item.row for item in corridor_cache[corridor_key])
                    )
                    context.core_indexes[corridor_key] = _prepare_table_core_rows(
                        corridor_rows,
                        lines,
                        prepared_note_body_metrics,
                        context.marker_prepared,
                        marker_safe=marker_safe,
                        source_lines=marker_source_lines,
                    )
                core_index = context.core_indexes[corridor_key]
                interval = core_index.interval(accepted_rows) if core_index is not None else None
                core_query = (core_index, *interval) if interval is not None else None
                drafts.append(
                    _RuleCandidateDraft(
                        context,
                        boundary_rules,
                        accepted_rows,
                        caption_line,
                        prepared_note_rows,
                        has_possible_table_notes,
                        score,
                        core_query,
                        _prepared_rule_candidate_query(band_indexes[corridor_key], accepted_rows, core_query),
                    )
                )
                continue
            candidate = _expand_rule_table_candidate(
                boundary_rules,
                accepted_rows,
                rows,
                lines,
                page_size,
                angle,
                median_height,
                caption_line,
                has_possible_table_notes,
                marker_cache,
                prepared_note_rows,
                prepared_note_body_metrics,
                prepared_candidate=_prepared_rule_candidate_query(band_indexes[corridor_key], accepted_rows),
            )
            candidate.score = score
            candidates.append(candidate)
    if defer_materialization:
        # 延迟物化时先冻结网格连通分量，供闭合网格检测和后续 owned
        # 合并共用一次结果；没有候选的页面保持零额外扫描。
        if drafts:
            context.grid_components = _connected_rule_grid_components(axis_lines, median_height)
        return drafts
    return _expand_candidates_to_connected_rule_grids(
        candidates,
        rows,
        page_size,
        angle,
        median_height,
        axis_lines,
        excluded_bboxes,
    )


def _expand_candidates_to_connected_rule_grids(
    candidates: list[_TableCandidate],
    rows: list[_VisualRow],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    excluded_bboxes: list[BBox],
    *,
    prepared_grid_bboxes: list[BBox] | None = None,
    grid_member_cache: dict[BBox, frozenset[int]] | None = None,
) -> list[_TableCandidate]:
    """把已确认候选沿连续横边界和贯穿竖轨扩展到完整物理网格。"""

    grid_bboxes = (
        prepared_grid_bboxes
        if prepared_grid_bboxes is not None
        else [
            grid_bbox
            for grid_bbox in _connected_rule_grid_bboxes(axis_lines, median_height)
            if not any(_bbox_overlap_in_smaller(grid_bbox, excluded_bbox) >= 0.5 for excluded_bbox in excluded_bboxes)
        ]
    )
    if not grid_bboxes:
        return candidates

    tolerance = max(2.0, median_height)
    if grid_member_cache is None:
        grid_member_cache = {}
    for candidate in candidates:
        if candidate.core_bbox is None:
            continue
        core_local_bbox = _rotate_bbox_to_upright(
            candidate.core_bbox,
            page_size,
            angle,
        )
        matches = [
            grid_bbox
            for grid_bbox in grid_bboxes
            if _bbox_axis_overlap_ratio(
                core_local_bbox,
                grid_bbox,
                axis="x",
            )
            >= 0.9
            and core_local_bbox[3] >= grid_bbox[1] - tolerance
            and core_local_bbox[1] <= grid_bbox[3] + tolerance
        ]
        if not matches:
            continue
        grid_bbox = max(
            matches,
            key=lambda bbox: (
                min(core_local_bbox[3], bbox[3]) - max(core_local_bbox[1], bbox[1]),
                _bbox_area(bbox),
            ),
        )
        expanded_core_bbox = _bbox_union(core_local_bbox, grid_bbox)
        candidate.local_bbox = _bbox_union(candidate.local_bbox, grid_bbox)
        candidate.core_bbox = _rotate_bbox_from_upright(
            expanded_core_bbox,
            page_size,
            angle,
        )
        candidate.bbox = _rotate_bbox_from_upright(
            candidate.local_bbox,
            page_size,
            angle,
        )
        grid_members = grid_member_cache.get(expanded_core_bbox)
        if grid_members is None:
            grid_members = frozenset(
                fragment.line_index
                for row in rows
                if expanded_core_bbox[1] <= row.center_y <= expanded_core_bbox[3]
                for fragment in row.fragments
                if _point_in_bbox(
                    (
                        _bbox_center_x(fragment.local_bbox),
                        _bbox_center_y(fragment.local_bbox),
                    ),
                    expanded_core_bbox,
                )
            )
            grid_member_cache[expanded_core_bbox] = grid_members
        candidate.line_indices = _SharedLineIndexSet(
            grid_members,
            candidate.line_indices,
        )
        for annotation in candidate.annotations:
            candidate.line_indices.difference_update(annotation.line_indices)
    return candidates


def _build_closed_rule_grid_candidates(
    rows: list[_VisualRow],
    lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    axis_lines: list[_LocalAxisLine],
    excluded_bboxes: list[BBox],
    caption_candidates: list[tuple[_LineItem, BBox]] | None = None,
    *,
    grid_components: list[list[_LocalAxisLine]] | None = None,
    previous_table_grid: _TableContinuationGrid | None = None,
    page_index: int | None = None,
    leading_obstacles: list[BBox] | None = None,
    form_bboxes: list[BBox] | None = None,
) -> list[_TableCandidate]:
    """用闭合物理网格接纳含空行或仅有表头文本的稀疏表格。"""

    candidates: list[_TableCandidate] = []
    row_interval_index = _build_row_interval_index(rows)
    components = grid_components if grid_components is not None else _connected_rule_grid_components(axis_lines, median_height)
    # 复用方传入的分量仍由上面的调用/检测边界生成；这里只迭代，不重复扫描竖轨。
    for component in components:
        grid_bbox = _bbox_union_many([rule.bbox for rule in component])
        if any(_bbox_overlap_in_smaller(grid_bbox, excluded_bbox) >= 0.5 for excluded_bbox in excluded_bboxes):
            continue
        # 只有包住网格的 Form 才能否决表格，单元格内部的小图标不能反向吞掉父表。
        if any(_bbox_overlap_in_first(grid_bbox, form_bbox) >= 0.9 for form_bbox in form_bboxes or []):
            continue
        core_rows = _rows_inside_rule_interval(
            rows,
            grid_bbox,
            excluded_bboxes,
            row_interval_index,
        )
        if not core_rows:
            continue

        vertical_positions = _closed_grid_vertical_track_positions(
            component,
            axis_lines,
            median_height,
        )
        if len(vertical_positions) < 2:
            continue
        edge_tolerance = max(2.0, 0.25 * median_height)
        if (
            abs(vertical_positions[0] - grid_bbox[0]) > edge_tolerance
            or abs(vertical_positions[-1] - grid_bbox[2]) > edge_tolerance
        ):
            continue

        if len(component) == 2:
            if len(vertical_positions) < 3:
                continue
            occupied_columns = _count_occupied_closed_grid_columns(
                core_rows,
                vertical_positions,
            )
            if occupied_columns < 2 and not _matches_previous_table_grid(
                previous_table_grid,
                page_index,
                angle,
                page_size,
                median_height,
                grid_bbox,
                vertical_positions,
                rows,
                leading_obstacles or [],
            ):
                continue

        caption_line = _find_table_caption(
            lines,
            grid_bbox,
            page_size,
            angle,
            median_height,
            caption_candidates,
        )
        candidate = _expand_rule_table_candidate(
            [component[0], component[-1]],
            core_rows,
            rows,
            lines,
            page_size,
            angle,
            median_height,
            caption_line,
        )
        candidate.score = float(100 + len(component) + len(vertical_positions))
        candidates.append(candidate)
    return candidates


def _matches_previous_table_grid(
    previous: _TableContinuationGrid | None,
    page_index: int | None,
    angle: int,
    page_size: tuple[float, float],
    em: float,
    bbox: BBox,
    positions: list[float],
    rows: list[_VisualRow],
    obstacles: list[BBox],
) -> bool:
    """仅相邻页首个完整网格与前页末表全部列轨匹配时，接纳单列有字的续表。"""
    if previous is None or page_index != previous.page_index + 1 or angle != previous.angle:
        return False
    width = page_size[0] if angle in (0, 180) else page_size[1]
    tolerance = max(previous.tolerance, max(2.0, 0.25 * em) / width)
    if len(positions) != len(previous.positions) or any(
        abs(position / width - prior) > tolerance for position, prior in zip(positions, previous.positions)
    ):
        return False
    edge_tolerance = max(2.0, 0.25 * em)
    return not any(row.bbox[3] < bbox[1] - edge_tolerance for row in rows) and not any(
        box[3] < bbox[1] - edge_tolerance for box in obstacles
    )


def _trailing_table_continuation_grid(source: _PageSource, prepared: _PreparedPage) -> _TableContinuationGrid | None:
    """从末表最后物理行提取列轨，允许此前存在合并单元格。"""
    tables = [block for block in prepared.fixed_blocks if block["type"] == "table" and block.get("angle", 0) == 0]
    if source.page_index is None or not tables:
        return None
    table = max(tables, key=lambda block: block["bbox"][3])
    bbox = table["bbox"]
    heights = [line.effective_height for line in source.lines if line.effective_height > 0]
    em = statistics.median(heights) if heights else 1.0
    tolerance = max(2.0, 0.25 * em)
    marginals = {"header", "footer", "page_number"}
    other_boxes = [block["bbox"] for block in prepared.fixed_blocks if block is not table and block["type"] not in marginals]
    other_boxes.extend(line.bbox for line in prepared.remaining_lines if line.semantic_type not in marginals)
    if any(box[3] > bbox[3] + tolerance for box in other_boxes):
        return None
    axis_lines = _transform_axis_lines(source.drawing_lines, source.page_size, 0)
    horizontal = sorted(
        [
            rule
            for rule in axis_lines
            if rule.orientation == "horizontal"
            and abs(rule.bbox[0] - bbox[0]) <= tolerance
            and abs(rule.bbox[2] - bbox[2]) <= tolerance
            and bbox[1] - tolerance <= _bbox_center_y(rule.bbox) <= bbox[3] + tolerance
        ],
        key=lambda rule: _bbox_center_y(rule.bbox),
    )
    if (
        len(horizontal) < 2
        or max(abs(_bbox_center_y(horizontal[0].bbox) - bbox[1]), abs(_bbox_center_y(horizontal[-1].bbox) - bbox[3]))
        > tolerance
    ):
        return None
    # Pagination continues the terminal cell structure, not necessarily the
    # structure of merged headers higher up the same table. Keep the existing
    # coverage threshold, but measure it over the final full-width rule band.
    trailing_band = horizontal[-2:]
    top, bottom = (_bbox_center_y(rule.bbox) for rule in trailing_band)
    if bottom - top < max(em, 2.0 * tolerance):
        return None
    positions = _closed_grid_vertical_track_positions(trailing_band, axis_lines, em)
    position_tolerance = max(1.0, 0.1 * em)
    positions = [
        position for position in positions
        if any(rule.orientation == "vertical"
               and abs(_bbox_center_x(rule.bbox) - position) <= position_tolerance
               and rule.bbox[1] <= bottom + tolerance and rule.bbox[3] >= bottom - tolerance
               for rule in axis_lines)
    ]
    if len(positions) < 3 or max(abs(positions[0] - bbox[0]), abs(positions[-1] - bbox[2])) > tolerance:
        return None
    return _TableContinuationGrid(
        source.page_index,
        0,
        tuple(position / source.page_size[0] for position in positions),
        tolerance / source.page_size[0],
    )


def _native_multicolumn_grid_bboxes(source: _PageSource, em: float, region: BBox) -> list[BBox]:
    """用完整连通网格证明表格，并要求图形候选实际跨越有字单元格。"""
    axis_lines = _transform_axis_lines(source.drawing_lines, source.page_size, 0)
    fragments = _build_fragments([line for line in source.lines if line.angle == 0], source.page_size)
    rows = _cluster_fragment_rows(fragments, em)
    output = []
    tolerance = max(2.0, 0.25 * em)
    for component in _connected_rule_grid_components(axis_lines, em):
        if len(component) < 4:
            continue
        bbox = _bbox_union_many([rule.bbox for rule in component])
        if _bbox_overlap_in_smaller(region, bbox) < 0.5:
            continue
        if any(_bbox_overlap_in_smaller(bbox, form) >= 0.5 for form in source.form_bboxes):
            continue
        outer = _closed_grid_vertical_track_positions(component, axis_lines, em)
        if len(outer) < 2 or max(abs(outer[0] - bbox[0]), abs(outer[-1] - bbox[2])) > tolerance:
            continue
        proven_bands = []
        region_bands = []
        for top, bottom in zip(component, component[1:]):
            band_center = (_bbox_center_y(top.bbox) + _bbox_center_y(bottom.bbox)) / 2
            positions = _closed_grid_vertical_track_positions([top, bottom], axis_lines, em)
            band_rows = [row for row in rows if _bbox_center_y(top.bbox) < row.center_y < _bbox_center_y(bottom.bbox)]
            if _count_occupied_closed_grid_columns(band_rows, positions) >= 2:
                proven_bands.append(positions)
                if region[1] < band_center < region[3]:
                    # 只统计候选内部的文字与列，避免借相邻单元格为单格图表提供证据。
                    region_rows = [_clip_visual_row_to_corridor(row, region, margin=0.0) for row in band_rows]
                    region_rows = [row for row in region_rows if row is not None]
                    covered_columns = sum(region[0] < (left + right) / 2 < region[2] for left, right in zip(positions, positions[1:]))
                    if covered_columns >= 2 and _count_occupied_closed_grid_columns(region_rows, positions) >= 2:
                        region_bands.append(positions)
        # 合并表头的列轨可能每行都不同。候选有足够的二维范围且确实跨列时，
        # 到完整连通网格中检查重复数据行，不要求第二行数据也已被图形裁框包含。
        if not region_bands or (len(region_bands) < 2 and region[3] - region[1] < 3 * em):
            continue
        if any(
            len(first) == len(second) and all(abs(a - b) <= tolerance for a, b in zip(first, second))
            for index, first in enumerate(proven_bands)
            for second in proven_bands[index + 1 :]
        ):
            output.append(bbox)
    # A graphic may span several separate tables. Their proven occupied grids
    # must cover most of the graphic, rather than borrowing a nearby table.
    intersections = [(max(box[0], region[0]), max(box[1], region[1]),
                      min(box[2], region[2]), min(box[3], region[3])) for box in output]
    x_edges = sorted({x for box in intersections for x in (box[0], box[2])})
    covered = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        intervals = []
        for top, bottom in sorted((box[1], box[3]) for box in intersections if box[0] <= left and box[2] >= right):
            if intervals and top <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], bottom)
            else:
                intervals.append([top, bottom])
        covered += (right - left) * sum(bottom - top for top, bottom in intervals)
    return output if covered >= 0.7 * _bbox_area(region) else []


def _closed_grid_vertical_track_positions(
    horizontal_rules: list[_LocalAxisLine],
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[float]:
    """收集覆盖首末横边界中心跨度至少九成的竖轨并合并重复路径。"""

    top = _bbox_center_y(horizontal_rules[0].bbox)
    bottom = _bbox_center_y(horizontal_rules[-1].bbox)
    grid_height = max(0.1, bottom - top)
    left = min(rule.bbox[0] for rule in horizontal_rules)
    right = max(rule.bbox[2] for rule in horizontal_rules)
    edge_tolerance = max(2.0, 0.25 * median_height)
    tracks = []
    for line in axis_lines:
        if line.orientation != "vertical":
            continue
        overlap = max(
            0.0,
            min(line.bbox[3], bottom) - max(line.bbox[1], top),
        )
        position = _bbox_center_x(line.bbox)
        if overlap > 0 and left - edge_tolerance <= position <= right + edge_tolerance:
            tracks.append((position, max(top, line.bbox[1]), min(bottom, line.bbox[3])))

    position_tolerance = max(1.0, 0.1 * median_height)
    position_groups = []
    for track in sorted(tracks):
        if position_groups and abs(track[0] - statistics.mean(t[0] for t in position_groups[-1])) <= position_tolerance:
            position_groups[-1].append(track)
        else:
            position_groups.append([track])
    positions = []
    for group in position_groups:
        intervals = []
        for start, end in sorted((t[1], t[2]) for t in group):
            if intervals and start <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        if sum(end - start for start, end in intervals) >= 0.9 * grid_height:
            positions.append(statistics.mean(t[0] for t in group))
    return positions


def _count_occupied_closed_grid_columns(
    rows: list[_VisualRow],
    vertical_positions: list[float],
) -> int:
    """按文本片段中心统计闭合网格中实际有文字的物理列数。"""

    occupied_columns: set[int] = set()
    for row in rows:
        for fragment in row.fragments:
            center_x = _bbox_center_x(fragment.local_bbox)
            matching_columns = [
                index
                for index, (left, right) in enumerate(zip(vertical_positions, vertical_positions[1:]))
                if left < center_x < right
            ]
            if len(matching_columns) == 1:
                occupied_columns.add(matching_columns[0])
    return len(occupied_columns)


def _connected_rule_grid_bboxes(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
    *,
    components: list[list[_LocalAxisLine]] | None = None,
) -> list[BBox]:
    """把端点一致且由外轨或至少两条列轨贯穿的相邻横线组成网格框。"""

    # components 是同一页面内已冻结的私有复用输入；未提供时保持原签名
    # 和原计算，直接调用本函数的测试/旧调用路径不受影响。
    if components is None:
        components = _connected_rule_grid_components(axis_lines, median_height)
    return [_bbox_union_many([rule.bbox for rule in component]) for component in components]


def _connected_rule_grid_components(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """保留连续网格的横边界成员，供精确外轨和横边界数量校验。"""

    output: list[list[_LocalAxisLine]] = []
    for rule_group in _group_long_horizontal_rules(axis_lines, median_height):
        components: list[list[_LocalAxisLine]] = []
        for rule in rule_group:
            if not components or not _rule_bands_share_grid_tracks(
                components[-1][-1],
                rule,
                axis_lines,
                median_height,
            ):
                components.append([rule])
            else:
                components[-1].append(rule)
        output.extend(component for component in components if len(component) >= 2)
    return output


def _connected_horizontal_rule_bboxes(
    source: _PageSource,
) -> set[BBox]:
    """返回参与常规横排闭合网格的原始水平线框，供上游避免删除真实表格边界。"""

    angle_lines = [line for line in source.lines if line.angle == 0]
    fragments = _build_fragments(angle_lines, source.page_size)
    if not fragments:
        return set()
    local_axis_lines = _transform_axis_lines(
        source.drawing_lines,
        source.page_size,
        0,
    )
    return {
        rule.original_bbox
        for component in _connected_rule_grid_components(
            local_axis_lines,
            _median_fragment_height(fragments),
        )
        for rule in component
        if rule.orientation == "horizontal"
    }


def _rule_bands_share_grid_tracks(
    top_rule: _LocalAxisLine,
    bottom_rule: _LocalAxisLine,
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> bool:
    """校验相邻横边界的跨度，并确认其间存在连续外框或稳定列分隔线。"""

    top_width = max(0.1, top_rule.bbox[2] - top_rule.bbox[0])
    bottom_width = max(0.1, bottom_rule.bbox[2] - bottom_rule.bbox[0])
    overlap_left = max(top_rule.bbox[0], bottom_rule.bbox[0])
    overlap_right = min(top_rule.bbox[2], bottom_rule.bbox[2])
    overlap_width = max(0.0, overlap_right - overlap_left)
    endpoint_tolerance = max(4.0, 2.0 * median_height)
    if (
        overlap_width / max(top_width, bottom_width) < 0.9
        or abs(top_rule.bbox[0] - bottom_rule.bbox[0]) > endpoint_tolerance
        or abs(top_rule.bbox[2] - bottom_rule.bbox[2]) > endpoint_tolerance
    ):
        return False

    top_y = _bbox_center_y(top_rule.bbox)
    bottom_y = _bbox_center_y(bottom_rule.bbox)
    track_tolerance = max(1.0, 0.25 * median_height)
    raw_positions = [
        _bbox_center_x(line.bbox)
        for line in axis_lines
        if line.orientation == "vertical"
        and line.bbox[1] <= top_y + track_tolerance
        and line.bbox[3] >= bottom_y - track_tolerance
        and overlap_left - track_tolerance <= _bbox_center_x(line.bbox) <= overlap_right + track_tolerance
    ]
    position_groups: list[list[float]] = []
    for position in sorted(raw_positions):
        if position_groups and abs(position - _mean_rule_positions(position_groups[-1])) <= track_tolerance:
            position_groups[-1].append(position)
        else:
            position_groups.append([position])
    positions = [_mean_rule_positions(group) for group in position_groups]
    has_outer_tracks = any(abs(position - overlap_left) <= endpoint_tolerance for position in positions) and any(
        abs(position - overlap_right) <= endpoint_tolerance for position in positions
    )
    interior_tracks = [
        position for position in positions if overlap_left + track_tolerance < position < overlap_right - track_tolerance
    ]
    return has_outer_tracks or len(interior_tracks) >= 2


def _iter_rule_spans(rule_count: int):
    """按历史顺序枚举全部横线边界组合，不删除任何内部子候选。"""

    for first_index in range(max(0, rule_count - 1)):
        for bottom_index in range(first_index + 1, rule_count):
            yield first_index, bottom_index


def _group_long_horizontal_rules(
    axis_lines: list[_LocalAxisLine],
    median_height: float,
) -> list[list[_LocalAxisLine]]:
    """按近似左右端点聚合长横线，并去除同位置重复路径。"""

    minimum_length = max(40.0, 10.0 * median_height)
    horizontal_lines = [
        line for line in axis_lines if line.orientation == "horizontal" and line.bbox[2] - line.bbox[0] >= minimum_length
    ]
    endpoint_tolerance = max(4.0, 2.0 * median_height)
    span_groups: list[list[_LocalAxisLine]] = []
    for line in sorted(horizontal_lines, key=lambda item: (item.bbox[0], item.bbox[2], item.bbox[1])):
        target = next(
            (
                group
                for group in span_groups
                if abs(line.bbox[0] - group[0].bbox[0]) <= endpoint_tolerance
                and abs(line.bbox[2] - group[0].bbox[2]) <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            span_groups.append([line])
        else:
            target.append(line)

    output: list[list[_LocalAxisLine]] = []
    for span_group in span_groups:
        unique_lines: list[_LocalAxisLine] = []
        for line in sorted(span_group, key=lambda item: _bbox_center_y(item.bbox)):
            if any(abs(_bbox_center_y(line.bbox) - _bbox_center_y(item.bbox)) <= 1.0 for item in unique_lines):
                continue
            unique_lines.append(line)
        if len(unique_lines) >= 2:
            output.append(unique_lines)
    return output


def _build_rule_corridor_rows(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    excluded_bboxes: list[BBox],
) -> list[_RuleCorridorRow]:
    """按精确左右边界预裁剪视觉行，保留原算法纵向准入所需的中间几何。"""

    output: list[_RuleCorridorRow] = []
    for row in rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=0.0)
        if clipped_row is None:
            continue
        fragments = [
            fragment
            for fragment in clipped_row.fragments
            if not any(
                _point_in_bbox(
                    (
                        _bbox_center_x(fragment.local_bbox),
                        _bbox_center_y(fragment.local_bbox),
                    ),
                    excluded_bbox,
                )
                for excluded_bbox in excluded_bboxes
            )
        ]
        if not fragments:
            continue
        output.append(
            _RuleCorridorRow(
                source_bbox=row.bbox,
                gate_center_y=clipped_row.center_y,
                row=_VisualRow(
                    fragments=fragments,
                    center_y=sum(_bbox_center_y(fragment.local_bbox) for fragment in fragments) / len(fragments),
                    bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
                    visual_row_id=clipped_row.visual_row_id,
                ),
            )
        )
    return output


def _rows_inside_rule_interval(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    excluded_bboxes: list[BBox],
    row_interval_index: tuple[list[tuple[int, _VisualRow]], list[float]] | None = None,
    corridor_cache: dict[tuple[float, float], list[_RuleCorridorRow]] | None = None,
) -> list[_VisualRow]:
    """截取边界走廊内文本行，并移除已由强图形核心覆盖的片段。"""

    if corridor_cache is not None:
        corridor_key = (rule_bbox[0], rule_bbox[2])
        corridor_rows = corridor_cache.get(corridor_key)
        if corridor_rows is None:
            corridor_rows = _build_rule_corridor_rows(
                rows,
                rule_bbox,
                excluded_bboxes,
            )
            corridor_cache[corridor_key] = corridor_rows
        return [
            item.row
            for item in corridor_rows
            if item.source_bbox[1] <= rule_bbox[3]
            and item.source_bbox[3] >= rule_bbox[1]
            and rule_bbox[1] <= item.gate_center_y <= rule_bbox[3]
        ]

    output: list[_VisualRow] = []
    if row_interval_index is None:
        indexed_rows = list(enumerate(rows))
    else:
        records, starts = row_interval_index
        upper = bisect_right(starts, rule_bbox[3])
        indexed_rows = [(index, row) for index, row in records[:upper] if row.bbox[3] >= rule_bbox[1]]
        indexed_rows.sort(key=lambda item: item[0])
    for _index, row in indexed_rows:
        clipped_row = _clip_visual_row_to_corridor(row, rule_bbox, margin=0.0)
        if clipped_row is None or not rule_bbox[1] <= clipped_row.center_y <= rule_bbox[3]:
            continue
        fragments = [
            fragment
            for fragment in clipped_row.fragments
            if not any(
                _point_in_bbox(
                    (
                        _bbox_center_x(fragment.local_bbox),
                        _bbox_center_y(fragment.local_bbox),
                    ),
                    excluded_bbox,
                )
                for excluded_bbox in excluded_bboxes
            )
        ]
        if not fragments:
            continue
        output.append(
            _VisualRow(
                fragments=fragments,
                center_y=sum(_bbox_center_y(fragment.local_bbox) for fragment in fragments) / len(fragments),
                bbox=_bbox_union_many([fragment.local_bbox for fragment in fragments]),
                visual_row_id=clipped_row.visual_row_id,
            )
        )
    return output


def _build_row_interval_index(
    rows: list[_VisualRow],
) -> tuple[list[tuple[int, _VisualRow]], list[float]]:
    """建立按行上边界排序的安全超集索引，保留原始行顺序供后续处理。"""
    records = sorted(
        enumerate(rows),
        key=lambda item: (item[1].bbox[1], item[1].bbox[3], item[0]),
    )
    return records, [row.bbox[1] for _index, row in records]


def _partition_rows_by_rule_intervals(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
) -> list[list[_VisualRow]]:
    """一次把候选行分配到相邻横线闭区间，保持共享边界行同时属于两侧区间。"""

    if len(rule_group) < 2:
        return []
    centers = [_bbox_center_y(rule.bbox) for rule in rule_group]
    groups: list[list[_VisualRow]] = [[] for _ in range(len(rule_group) - 1)]
    for row in rows:
        center_y = row.center_y
        position = bisect_left(centers, center_y)
        if position < len(centers) and centers[position] == center_y:
            if position > 0:
                groups[position - 1].append(row)
            if position < len(groups):
                groups[position].append(row)
            continue
        interval_index = position - 1
        if 0 <= interval_index < len(groups):
            groups[interval_index].append(row)
    return groups


def _prepare_rule_band_index(rows: list[_VisualRow], rules: list[_LocalAxisLine]) -> _RuleBandIndex | None:
    """为一条精确走廊计算完整横线组的首区间，异常和重复横线回退。"""

    if len({id(row) for row in rows}) != len(rows) or any(
        type(row.center_y) is not float or not math.isfinite(row.center_y) for row in rows
    ):
        return None
    centers = []
    for rule in rules:
        box = rule.bbox
        if (
            type(box) not in (tuple, list)
            or len(box) != 4
            or any(type(value) is not float or not math.isfinite(value) for value in box)
        ):
            return None
        center = _bbox_center_y(box)
        if not math.isfinite(center) or centers and center <= centers[-1]:
            return None
        centers.append(center)
    native = _prepare_native_rule_candidates(rows, centers, rules)
    bands = [bisect_left(centers, row.center_y) for row in rows] if native is None else []
    return _RuleBandIndex(
        tuple(rows),
        {id(row): index for index, row in enumerate(rows)},
        tuple(bands),
        tuple(position < len(centers) and centers[position] == row.center_y for position, row in zip(bands, rows))
        if native is None
        else (),
        len(centers),
        native,
    )


def _prepare_native_rule_candidates(rows, centers, rules):
    """仅为普通有限数据准备共享原生走廊，特殊对象在计算开始前留给参考路径。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None or not hasattr(native, "PreparedRuleCandidates"):
        return None
    if any(type(rule) is not _LocalAxisLine for rule in rules):
        return None
    packed = []
    for row in rows:
        if (
            type(row) is not _VisualRow
            or type(row.fragments) is not list
            or type(row.bbox) not in (tuple, list)
            or len(row.bbox) != 4
            or any(type(value) is not float or not math.isfinite(value) for value in row.bbox)
        ):
            return None
        members = []
        for fragment in row.fragments:
            if (
                type(fragment) is not _Fragment
                or type(fragment.line_index) is not int
                or not -(2**63) <= fragment.line_index < 2**63
            ):
                return None
            members.append(fragment.line_index)
        packed.append((row.center_y, len(row.fragments), row.bbox, members))
    return native.PreparedRuleCandidates(centers, packed)


def _prepared_rule_candidate_query(index, rows, prepared_core=None):
    """共享走廊身份只验证一次，后续复用已由 core.interval 验证的保序切片。"""
    if index is None or index.native is None:
        return None
    if prepared_core is not None:
        core, start, end = prepared_core
        if type(core) is _PreparedTableCoreRows and type(core.rows) is tuple:
            shared = index.shared_core_rows
            if shared is None or shared[0] is not core or shared[1] is not core.rows:
                # 仅缓存内部不可变 tuple；字段替换会因身份变化重新验证，外部可变列表不准入。
                same_rows = core.rows is index.rows or (
                    len(core.rows) == len(index.rows)
                    and all(actual is expected for actual, expected in zip(core.rows, index.rows, strict=True))
                )
                shared = index.shared_core_rows = (core, core.rows, same_rows)
            if shared[2]:
                return (index, start, end) if start < end else None
    interval = index.interval(rows)
    return (index, *interval) if interval is not None and interval[0] < interval[1] else None


def _prepared_rule_candidate_core(prepared, rules, *, owned=False):
    """原生一次归并核心成员与坐标来源，Python 保留原坐标身份及集合契约。"""
    if prepared is None or not rules:
        return None
    boxes = [rule.bbox for rule in rules]
    if any(
        type(box) not in (tuple, list)
        or len(box) != 4
        or any(type(value) is not float or not math.isfinite(value) for value in box)
        for box in boxes
    ):
        return None
    index, start, end = prepared
    members, sources = (index.native.owned_core if owned else index.native.core)(start, end, boxes)
    bbox = tuple(
        (boxes[source] if source < len(boxes) else index.rows[source - len(boxes)].bbox)[axis]
        for axis, source in enumerate(sources)
    )
    if owned:
        from ._native_table_merge import _NativeCoreLineSet

        return _NativeCoreLineSet(members), bbox
    return {member for member in members}, bbox


def _partition_rows_by_prepared_bands(
    rows: list[_VisualRow],
    rules: list[_LocalAxisLine],
    first_index: int,
    index: _RuleBandIndex | None,
    median_height: float | None = None,
) -> list[list[_VisualRow]] | None:
    """仅对连续且保序的走廊行复用首区间，仍让边界行双归属。"""

    if index is None or first_index < 0 or first_index + len(rules) > index.rule_count:
        return None
    interval = index.interval(rows)
    if interval is None:
        return None
    start, end = interval
    if index.native is not None:
        evidence_safe = type(median_height) is float and math.isfinite(median_height)
        packed_groups, accepted, packed_segments, short = index.native.partition_with_segments(
            start, end, first_index, len(rules), median_height if evidence_safe else 0.0
        )
        groups = [[index.rows[position] for position in group] for group in packed_groups]
        segments = [[index.rows[position] for position in group] for group in packed_segments]
        return _RuleIntervalGroups(groups, rows, rules, median_height, accepted, segments, short) if evidence_safe else groups
    groups: list[list[_VisualRow]] = [[] for _ in range(max(0, len(rules) - 1))]
    for row, band, on_rule in zip(rows, index.bands[start:end], index.on_rule[start:end], strict=True):
        local = band - first_index
        if on_rule:
            if 0 <= local - 1 < len(groups):
                groups[local - 1].append(row)
            if 0 <= local < len(groups):
                groups[local].append(row)
        elif 0 <= local - 1 < len(groups):
            groups[local - 1].append(row)
    return groups


def _partition_rows_by_rule_intervals_cached(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
    previous: _RuleIntervalPrefix | None,
) -> tuple[list[list[_VisualRow]], _RuleIntervalPrefix]:
    """严格同一行序列与递增横线前缀时只补末区间，其余执行原分配。"""

    reusable = (
        previous is not None
        and len(rule_group) == len(previous.rules) + 1
        and len(rows) == len(previous.rows)
        and all(row is old for row, old in zip(rows, previous.rows, strict=True))
        and all(rule is old for rule, old in zip(rule_group[:-1], previous.rules, strict=True))
        and all(type(row.center_y) is float and math.isfinite(row.center_y) for row in rows)
        and all(
            type(rule.bbox) in (tuple, list)
            and len(rule.bbox) == 4
            and all(type(value) is float and math.isfinite(value) for value in rule.bbox)
            for rule in rule_group
        )
    )
    if reusable:
        centers = [_bbox_center_y(rule.bbox) for rule in rule_group]
        reusable = all(first < second for first, second in zip(centers, centers[1:]))
    if reusable:
        low, high = centers[-2:]
        groups = [*previous.groups, [row for row in rows if low <= row.center_y <= high]]
    else:
        groups = _partition_rows_by_rule_intervals(rows, rule_group)
    return groups, _RuleIntervalPrefix(tuple(rows), tuple(rule_group), groups)


def _every_rule_interval_has_multi_cell_row(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
    median_height: float,
    interval_row_groups: list[list[_VisualRow]] | None = None,
) -> bool:
    """要求候选跨过的每个相邻横线区间都存在至少一行多单元文本。"""

    if len(rule_group) < 2:
        return False
    if (
        type(median_height) is float
        and type(interval_row_groups) is _RuleIntervalGroups
        and interval_row_groups.source_rows is rows
        and interval_row_groups.source_rules is rule_group
        and interval_row_groups.height == median_height
    ):
        return interval_row_groups.accepted
    groups = interval_row_groups
    if groups is None:
        groups = _partition_rows_by_rule_intervals(rows, rule_group)
    for interval_index, ((top_rule, bottom_rule), interval_rows) in enumerate(zip(zip(rule_group, rule_group[1:]), groups)):
        top = _bbox_center_y(top_rule.bbox)
        bottom = _bbox_center_y(bottom_rule.bbox)
        if any(len(row.fragments) >= 2 for row in interval_rows):
            continue
        # 紧邻顶边界的合并表头可能由 pdftext 输出为一个短 fragment；
        # 只放宽高度很小的首区间，避免把远处章节标题接到表格上。
        if interval_index == 0 and interval_rows and bottom - top <= 2.5 * median_height:
            continue
        return False
    return True


def _rule_intervals_are_column_compatible(
    rows: list[_VisualRow],
    rule_group: list[_LocalAxisLine],
    median_height: float,
    stable_column_cache: _StableColumnCache | None = None,
    interval_row_groups: list[list[_VisualRow]] | None = None,
) -> bool:
    """拒绝跨过长篇栏式正文、导致稳定列数明显塌缩的多表合并区间。"""

    # 同一次原生闭区间扫描已证明全部间距不超过六倍行高时，不再逐候选重复读取每根横线。
    if (
        type(interval_row_groups) is _RuleIntervalGroups
        and interval_row_groups.source_rows is rows
        and interval_row_groups.source_rules is rule_group
        and interval_row_groups.height == median_height
        and interval_row_groups.short_intervals
        and _bbox_center_y is _SHORT_INTERVAL_CENTER_REFERENCE
    ):
        return True

    interval_heights = [
        _bbox_center_y(bottom_rule.bbox) - _bbox_center_y(top_rule.bbox)
        for top_rule, bottom_rule in zip(rule_group, rule_group[1:])
    ]
    # 原逻辑对不高于六倍行高的所有区间都会无条件跳过后续兼容性检查；
    # 因此全部为短区间时可直接返回，避免先计算永远不会参与裁决的列统计。
    if interval_heights and all(height <= 6.0 * median_height for height in interval_heights):
        return True

    groups = interval_row_groups
    if groups is None:
        groups = _partition_rows_by_rule_intervals(rows, rule_group)
    profiles: list[tuple[int, float, int, float]] = []
    for band_rows, interval_height in zip(groups, interval_heights):
        interval_rows = [row for row in band_rows if len(row.fragments) >= 2]
        if _is_multiline_description_cell_band(band_rows, median_height):
            profiles.append((2, interval_height, len(band_rows), 1.0))
            continue
        stable_columns, column_coverage = _count_stable_columns(
            interval_rows,
            median_height,
            stable_column_cache,
        )
        profiles.append(
            (
                stable_columns,
                interval_height,
                len(interval_rows),
                column_coverage,
            )
        )
    maximum_columns = max(
        (columns for columns, _height, _row_count, _coverage in profiles),
        default=0,
    )
    for interval_index, (columns, interval_height, row_count, coverage) in enumerate(profiles):
        # 紧凑首区间可能只是跨列表头；一旦区间明显高于普通表头，
        # 也必须具有连续多单元格行，不能无条件跨过正文连接两张表。
        if interval_index == 0 and interval_height <= 2.5 * median_height:
            continue
        if interval_height <= 6.0 * median_height:
            continue
        minimum_rows = max(2, int(interval_height / max(8.0 * median_height, 0.1)))
        if row_count < minimum_rows or coverage < 0.5:
            return False
        if columns < max(2, int(0.5 * maximum_columns)):
            return False
    return True


def _is_multiline_description_cell_band(rows: list[_VisualRow], height: float) -> bool:
    """确认短左单元格与多行右单元格，不能用物理续行次数否定逻辑表格行。"""
    if len(rows) < 3 or len(rows[0].fragments) != 2:
        return False
    first, second = sorted(rows[0].fragments, key=lambda fragment: fragment.local_bbox[0])
    left, right = first.local_bbox, second.local_bbox
    if left[2] - left[0] > 0.45 * (right[2] - left[0]) or right[0] - left[2] < 0.5 * height:
        return False
    for previous, row in zip(rows, rows[1:]):
        if row.center_y - previous.center_y > 2.0 * height:
            return False
        if len(row.fragments) != 1:
            return False
        bbox = row.fragments[0].local_bbox
        if abs(bbox[0] - right[0]) > 0.75 * height or bbox[2] > right[2] + height:
            return False
    return True


def _continuous_table_row_segments(
    rows: list[_VisualRow],
    median_height: float,
) -> list[list[_VisualRow]]:
    """按物理行距切分边界区间，保留单元格换行参与连续性判断。"""

    segments: list[list[_VisualRow]] = []
    for row in sorted(rows, key=lambda item: item.center_y):
        if not segments or max(0.0, row.bbox[1] - segments[-1][-1].bbox[3]) > 3.0 * median_height:
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


_CONTINUOUS_SEGMENTS_REFERENCE = _continuous_table_row_segments


def _table_segment_reaches_boundaries(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
) -> bool:
    """要求数据行链分别贴近最近的上下边界，排除页眉线和远处章节标题。"""

    if not rows:
        return False
    maximum_gap = 2.5 * median_height
    top_gap = max(0.0, rows[0].bbox[1] - rule_bbox[1])
    bottom_gap = max(0.0, rule_bbox[3] - rows[-1].bbox[3])
    return top_gap <= maximum_gap and bottom_gap <= maximum_gap


class _RuleRowBounds:
    """保存走廊行框索引，只接受身份与顺序完全连续的查询。"""

    def __init__(self, rows):
        """一次构建有限行框极值树，特殊值和小页面不建立索引。"""
        from ...._compute_backend import get_native

        self.rows = rows
        self.positions = {id(row): i for i, row in enumerate(rows)}
        self.native = None
        native = get_native()
        if (
            native is not None
            and len(rows) >= 32
            and len(self.positions) == len(rows)
            and all(
                type(row.bbox) in (tuple, list)
                and len(row.bbox) == 4
                and all(type(v) is float and math.isfinite(v) for v in row.bbox)
                for row in rows
            )
        ):
            self.native = native.TableRowGeometry([row.bbox for row in rows])

    def bbox(self, rows):
        """保持重复和非连续行的原遍历；极值来源仍引用原坐标对象。"""
        if self.native is None or not rows:
            return None
        start = self.positions.get(id(rows[0]))
        if (
            start is None
            or start + len(rows) > len(self.rows)
            or any(row is not self.rows[start + offset] for offset, row in enumerate(rows))
        ):
            return None
        indices = self.native.union_indices(start, start + len(rows))
        return tuple(self.rows[index].bbox[axis] for axis, index in enumerate(indices))


def _table_rows_align_with_rule_span(
    rows: list[_VisualRow],
    rule_bbox: BBox,
    median_height: float,
    prepared_bounds: _RuleRowBounds | None = None,
) -> bool:
    """校验数据行总体跨度与横线走廊重叠，拒绝仅在边缘偶遇的多列文本。"""

    if not rows:
        return False
    rows_bbox = prepared_bounds.bbox(rows) if prepared_bounds is not None else None
    if rows_bbox is None:
        rows_bbox = _bbox_union_many([row.bbox for row in rows])
    rule_width = max(0.1, rule_bbox[2] - rule_bbox[0])
    rows_width = max(0.1, rows_bbox[2] - rows_bbox[0])
    overlap = max(
        0.0,
        min(rows_bbox[2], rule_bbox[2]) - max(rows_bbox[0], rule_bbox[0]),
    )
    return overlap / min(rule_width, rows_width) >= 0.9 and rows_width >= max(8.0 * median_height, 0.25 * rule_width)


def _count_aligned_vertical_rules(
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
    vertical_axis_lines: list[_LocalAxisLine] | None = None,
) -> int:
    """统计贯穿候选主要高度且位于横线跨度内的竖向分隔线。"""

    required_height = max(4.0 * median_height, 0.5 * (rule_bbox[3] - rule_bbox[1]))
    candidates = vertical_axis_lines if vertical_axis_lines is not None else axis_lines
    return sum(
        line.orientation == "vertical"
        and rule_bbox[0] - median_height <= _bbox_center_x(line.bbox) <= rule_bbox[2] + median_height
        and line.bbox[3] - line.bbox[1] >= required_height
        and _bbox_axis_overlap_ratio(line.bbox, rule_bbox, axis="y") >= 0.8
        for line in candidates
    )


def _compact_fully_ruled_grid_column_count(
    row_segment: list[_VisualRow],
    dense_rows: list[_VisualRow],
    interval_rules: list[_LocalAxisLine],
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
) -> int:
    """以完整横竖边界确认两行紧凑网格，并返回物理列数，失败时返回零。"""

    rule_height = max(0.1, rule_bbox[3] - rule_bbox[1])
    if len(row_segment) != 2 or len(dense_rows) != 2 or len(interval_rules) < 3 or rule_height > 6.0 * median_height:
        return 0

    vertical_positions = _full_height_vertical_rule_positions(
        axis_lines,
        rule_bbox,
        median_height,
    )
    if len(vertical_positions) < 3:
        return 0

    edge_tolerance = max(1.5, 0.25 * median_height)
    left_boundary = min(
        vertical_positions,
        key=lambda position: abs(position - rule_bbox[0]),
    )
    right_boundary = min(
        vertical_positions,
        key=lambda position: abs(position - rule_bbox[2]),
    )
    if (
        abs(left_boundary - rule_bbox[0]) > edge_tolerance
        or abs(right_boundary - rule_bbox[2]) > edge_tolerance
        or right_boundary <= left_boundary
    ):
        return 0

    grid_boundaries = [position for position in vertical_positions if left_boundary <= position <= right_boundary]
    if len(grid_boundaries) < 3:
        return 0
    grid_intervals = list(zip(grid_boundaries, grid_boundaries[1:]))

    occupied_columns: list[set[int]] = []
    for row in dense_rows:
        row_columns: list[int] = []
        for fragment in row.fragments:
            fragment_center = _bbox_center_x(fragment.local_bbox)
            matching_columns = [index for index, (left, right) in enumerate(grid_intervals) if left <= fragment_center <= right]
            if len(matching_columns) != 1:
                return 0
            column_index = matching_columns[0]
            if column_index in row_columns:
                return 0
            row_columns.append(column_index)
        if len(row_columns) < 2:
            return 0
        occupied_columns.append(set(row_columns))

    if len(occupied_columns[0] & occupied_columns[1]) < 2:
        return 0
    return len(grid_intervals)


def _full_height_vertical_rule_positions(
    axis_lines: list[_LocalAxisLine],
    rule_bbox: BBox,
    median_height: float,
) -> list[float]:
    """收集覆盖紧凑候选主要高度的竖线中心，并合并同位置重复路径。"""

    rule_height = max(0.1, rule_bbox[3] - rule_bbox[1])
    raw_positions: list[float] = []
    for line in axis_lines:
        if line.orientation != "vertical":
            continue
        overlap = max(
            0.0,
            min(line.bbox[3], rule_bbox[3]) - max(line.bbox[1], rule_bbox[1]),
        )
        if (
            overlap / rule_height < 0.8
            or line.bbox[3] - line.bbox[1] < 0.8 * rule_height
            or not rule_bbox[0] - median_height <= _bbox_center_x(line.bbox) <= rule_bbox[2] + median_height
        ):
            continue
        raw_positions.append(_bbox_center_x(line.bbox))

    deduplicated: list[list[float]] = []
    position_tolerance = max(1.0, 0.1 * median_height)
    for position in sorted(raw_positions):
        if deduplicated and abs(position - _mean_rule_positions(deduplicated[-1])) <= position_tolerance:
            deduplicated[-1].append(position)
        else:
            deduplicated.append([position])
    return [_mean_rule_positions(group) for group in deduplicated]


def _looks_like_page_column_prose(
    rows: list[_VisualRow],
    dense_rows: list[_VisualRow],
    stable_columns: int,
    fill_band_count: int,
    aligned_vertical_count: int,
    rule_bbox: BBox,
) -> bool:
    """用双栏占宽率识别夹在远横线间的普通并排正文。"""

    if stable_columns != 2 or fill_band_count >= 2 or aligned_vertical_count > 0 or len(dense_rows) / len(rows) < 0.55:
        return False
    corridor_width = max(0.1, rule_bbox[2] - rule_bbox[0])
    occupied_ratios = [
        sum(fragment.local_bbox[2] - fragment.local_bbox[0] for fragment in row.fragments) / corridor_width
        for row in dense_rows
    ]
    return statistics.median(occupied_ratios) >= 0.75


def _prepare_fill_band_infos(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    angle: int,
) -> list[tuple[PDFPathInfo, BBox]]:
    """预先转换可见填充 Path 的坐标，避免每个横线区间重复旋转。"""
    return [
        (path_info, _rotate_bbox_to_upright(path_info.bbox, page_size, angle))
        for path_info in path_infos
        if path_info.form_depth == 0 and path_info.fill_visible
    ]


def _count_repeated_fill_bands(
    path_infos: list[PDFPathInfo],
    rule_bbox: BBox,
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    prepared_path_infos: list[tuple[PDFPathInfo, BBox]] | None = None,
) -> int:
    """统计区间内左右端点和高度重复的填充行带，并对重叠 Path 去重。"""

    minimum_width = max(8.0 * median_height, 0.3 * (rule_bbox[2] - rule_bbox[0]))
    candidates: list[BBox] = []
    prepared = prepared_path_infos
    if prepared is None:
        prepared = _prepare_fill_band_infos(path_infos, page_size, angle)
    for _path_info, bbox in prepared:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if (
            width < minimum_width
            or not 0.25 * median_height <= height <= 3.0 * median_height
            or _bbox_center_y(bbox) < rule_bbox[1]
            or _bbox_center_y(bbox) > rule_bbox[3]
            or _bbox_axis_overlap_ratio(bbox, rule_bbox, axis="x") < 0.8
        ):
            continue
        if any(_bbox_overlap_in_smaller(bbox, item) >= 0.9 for item in candidates):
            continue
        candidates.append(bbox)

    endpoint_tolerance = max(3.0, median_height)
    groups: list[list[BBox]] = []
    for bbox in candidates:
        target = next(
            (
                group
                for group in groups
                if abs(bbox[0] - group[0][0]) <= endpoint_tolerance
                and abs(bbox[2] - group[0][2]) <= endpoint_tolerance
                and abs((bbox[3] - bbox[1]) - (group[0][3] - group[0][1])) <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            groups.append([bbox])
        else:
            target.append(bbox)
    return max((len(group) for group in groups), default=0)


_FILL_BAND_REFERENCES = (
    _count_repeated_fill_bands,
    _prepare_fill_band_infos,
    _bbox_center_y,
    _bbox_axis_overlap_ratio,
    _bbox_overlap_in_smaller,
    max,
    min,
    abs,
)


def _fill_band_rules_unchanged():
    """仅在原规则及数值函数均未替换时复用构建内快照，替换规则仍逐区间调用。"""
    return (
        _count_repeated_fill_bands,
        _prepare_fill_band_infos,
        _bbox_center_y,
        _bbox_axis_overlap_ratio,
        _bbox_overlap_in_smaller,
        max,
        min,
        abs,
    ) == _FILL_BAND_REFERENCES


def _prepare_native_fill_bands(prepared):
    """只为检测器当前只读页面构建准备自有框；重新构建或特殊输入不复用旧状态。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is None
        or not hasattr(native, "prepare_fill_bands")
        or not _fill_band_rules_unchanged()
        or type(prepared) is not list
        or any(type(item) is not tuple or len(item) != 2 for item in prepared)
    ):
        return None
    return native.prepare_fill_bands([box for _path, box in prepared])


def _longest_dense_multi_cell_rows(
    rows: list[_VisualRow],
    median_height: float,
) -> list[_VisualRow]:
    """返回行距不超过四倍行高的最长连续多单元格文本段。"""

    segments: list[list[_VisualRow]] = []
    for row in (item for item in rows if len(item.fragments) >= 2):
        if not segments or row.center_y - segments[-1][-1].center_y > 4.0 * median_height:
            segments.append([row])
        else:
            segments[-1].append(row)
    return max(segments, key=len, default=[])


def _expand_rule_table_candidate(
    rule_group: list[_LocalAxisLine],
    core_rows: list[_VisualRow],
    all_rows: list[_VisualRow],
    all_lines: list[_LineItem],
    page_size: tuple[float, float],
    angle: int,
    median_height: float,
    caption_line: _LineItem | None,
    has_possible_table_notes: bool = True,
    marker_cache: dict[tuple[int, str], bool] | None = None,
    prepared_note_rows: list[_PreparedTableNoteRow] | None = None,
    prepared_note_body_metrics: _PreparedTableNoteBodyMetrics | None = None,
    prepared_core: Any = None,
    prepared_annotation_geometry: Any = None,
    prepared_candidate: Any = None,
    *,
    owned: bool = False,
) -> _TableCandidate:
    """合并横线核心与上下注释，并保留注释的独立行身份。"""

    rule_bbox = _bbox_union_many([line.bbox for line in rule_group])
    native_core = (
        _prepared_rule_candidate_core(prepared_candidate, rule_group, owned=True)
        if owned
        else _prepared_rule_candidate_core(prepared_candidate, rule_group)
    )
    if owned and native_core is None:
        raise ValueError("owned candidate requires a prepared core")
    core_line_indices = (
        native_core[0] if native_core is not None else {fragment.line_index for row in core_rows for fragment in row.fragments}
    )
    caption_rows = _collect_caption_rows(all_rows, caption_line, rule_bbox, median_height)
    footnote_rows = (
        _collect_footnote_rows(
            all_rows,
            all_lines,
            rule_bbox,
            median_height,
            core_line_indices,
            page_size,
            angle,
            marker_cache,
            prepared_note_rows,
            prepared_note_body_metrics,
            prepared_core,
        )
        if has_possible_table_notes
        else []
    )
    if native_core is not None:
        core_local_bbox = native_core[1]
        indexed_bbox = True
    else:
        core_rows_bbox = prepared_core[0].bbox(prepared_core[1], prepared_core[2]) if prepared_core is not None else None
        indexed_bbox = core_rows_bbox is not None
        if core_rows_bbox is None:
            core_rows_bbox = _bbox_union_many([row.bbox for row in core_rows])
        core_local_bbox = _bbox_union(rule_bbox, core_rows_bbox)
    caption_annotation = _build_table_annotation(
        "caption",
        caption_rows,
        excluded_line_indices=core_line_indices,
        excluded_local_bbox=core_local_bbox,
        prepared_geometry=prepared_annotation_geometry,
    )
    footnote_annotation = _build_table_annotation(
        "footnote",
        footnote_rows,
        excluded_line_indices=core_line_indices,
        prepared_geometry=prepared_annotation_geometry,
    )
    annotations = [annotation for annotation in (caption_annotation, footnote_annotation) if annotation is not None]
    annotation_line_indices = (
        set().union(
            *(annotation.line_indices for annotation in annotations),
        )
        if annotations
        else set()
    )
    if indexed_bbox and all(
        type(value) is float and math.isfinite(value) for row in (*caption_rows, *footnote_rows) for value in row.bbox
    ):
        # 有限极值并集幂等，外层核心始终先参与比较，保留相等坐标的原对象。
        local_bbox = core_local_bbox
        for row in (*caption_rows, *footnote_rows):
            local_bbox = _bbox_union(local_bbox, row.bbox)
    else:
        included_rows = [*caption_rows, *core_rows, *footnote_rows]
        local_bbox = _bbox_union(core_local_bbox, _bbox_union_many([row.bbox for row in included_rows]))
    if owned:
        return (
            _rotate_bbox_from_upright(local_bbox, page_size, angle),
            local_bbox,
            _rotate_bbox_from_upright(core_local_bbox, page_size, angle),
            core_line_indices,
            annotations,
        )
    return _TableCandidate(
        bbox=_rotate_bbox_from_upright(local_bbox, page_size, angle),
        local_bbox=local_bbox,
        angle=angle,
        score=0.0,
        core_bbox=_rotate_bbox_from_upright(core_local_bbox, page_size, angle),
        # 表体成员与注释成员保持互斥；物化失败时会显式把无效注释放回表体投影。
        line_indices=core_line_indices - annotation_line_indices,
        annotations=annotations,
    )


def _new_stable_column_clusters() -> dict[str, list[dict[str, Any]]]:
    """创建与原稳定列算法一致的三组空聚类状态。"""

    return {alignment: [] for alignment in ("left", "center", "right")}


def _extend_stable_column_clusters(
    clusters_by_alignment: dict[str, list[dict[str, Any]]],
    rows: list[_VisualRow],
    *,
    start_row_index: int,
    tolerance: float,
) -> None:
    """按原顺序把新增行续入已有聚类状态，不改变均值和首命中规则。"""

    for alignment in ("left", "center", "right"):
        clusters = clusters_by_alignment[alignment]
        for row_index, row in enumerate(rows, start=start_row_index):
            for fragment in row.fragments:
                left, _top, right, _bottom = fragment.local_bbox
                if alignment == "left":
                    anchor = left
                elif alignment == "center":
                    anchor = (left + right) / 2
                else:
                    anchor = right
                cluster: dict[str, Any] | None = None
                for item in clusters:
                    if abs(anchor - float(item["mean"])) <= tolerance:
                        cluster = item
                        break
                if cluster is None:
                    clusters.append({"mean": anchor, "values": [anchor], "rows": {row_index}})
                else:
                    cluster["values"].append(anchor)
                    cluster["rows"].add(row_index)
                    # 保留原实现的 sum/len 计算方式，避免改变临界浮点聚类归属。
                    cluster["mean"] = sum(cluster["values"]) / len(cluster["values"])


def _stable_column_result(
    clusters_by_alignment: dict[str, list[dict[str, Any]]],
    row_count: int,
) -> tuple[int, float]:
    """从完整聚类状态计算原有稳定列数量与最低覆盖率。"""

    if row_count <= 0:
        return 0, 0.0

    best_result = (0, 0.0)
    for alignment in ("left", "center", "right"):
        clusters = clusters_by_alignment[alignment]
        stable_coverages: list[float] = []
        for cluster in clusters:
            coverage = len(cluster["rows"]) / row_count
            if coverage >= 0.5:
                stable_coverages.append(coverage)
        result = (
            len(stable_coverages),
            min(stable_coverages) if stable_coverages else 0.0,
        )
        # 仅在结果严格更优时更新，平局时保留既有的左对齐优先级。
        if result > best_result:
            best_result = result
    return best_result


def _count_stable_columns(
    rows: list[_VisualRow],
    median_height: float,
    cache: _StableColumnCache | None = None,
    *,
    allow_prefix_reuse: bool = False,
) -> tuple[int, float]:
    """复用走廊锚点与严格前缀的原生累计状态，特殊数值保留 Python 求和行为。"""
    from ...._compute_backend import get_native

    native = get_native()
    if native is None or sys.implementation.name != "cpython" or not (3, 10) <= sys.version_info[:2] <= (3, 14):
        return _count_stable_columns_python(rows, median_height, cache, allow_prefix_reuse=allow_prefix_reuse)
    if type(median_height) is not float or not math.isfinite(median_height):
        return _count_stable_columns_python(rows, median_height)
    # 只有检测器独占且不写入几何的构建阶段启用；普通缓存调用仍保留原有可检查的 Python 状态。
    if (
        cache is not None
        and cache.owned_geometry
        and (
            _VisualRow.__getattribute__,
            getattr(_VisualRow, "fragments", None),
            _Fragment.__getattribute__,
            getattr(_Fragment, "local_bbox", None),
        )
        == _COLUMN_GEOMETRY_REFERENCES
    ):
        if cache.native_geometry is None:
            cache.native_geometry = native.NativeColumnCache(sys.version_info >= (3, 12))
        result = cache.native_geometry.count(rows, median_height, allow_prefix_reuse, _VisualRow, _Fragment)
        if result is not None:
            return result
        # 特殊输入可能执行调用方代码；完整退回后不再复用此前普通字段的准备状态。
        cache.owned_geometry = False
        cache.native_geometry = None
    row_ids = tuple(id(row) for row in rows)
    key = (median_height, row_ids)
    if cache is not None and key in cache.results:
        return cache.results[key]
    prefix_key = (median_height, row_ids[0]) if row_ids and allow_prefix_reuse else None
    prefix = cache.prefixes.get(prefix_key) if cache is not None and prefix_key is not None else None
    reuse = (
        prefix is not None
        and isinstance(prefix.clusters_by_alignment, native.StableColumnClusters)
        and len(prefix.row_ids) < len(row_ids)
        and row_ids[: len(prefix.row_ids)] == prefix.row_ids
    )
    start = len(prefix.row_ids) if reuse else 0
    packed = []
    for row in rows[start:]:
        existing = cache.prepared_rows.get(id(row)) if cache is not None else None
        if existing is None:
            values = []
            for fragment in row.fragments:
                left, _top, right, _bottom = fragment.local_bbox
                if type(left) is not float or type(right) is not float or not math.isfinite(left) or not math.isfinite(right):
                    values = None
                    break
                values.append((left, right))
            if cache is not None:
                # 强引用限定于本次构建，避免行对象销毁后地址被另一个对象复用。
                cache.prepared_rows[id(row)] = (row, values)
        else:
            values = existing[1]
        if values is None:
            return _count_stable_columns_python(rows, median_height)
        packed.append(values)
    state = prefix.clusters_by_alignment if reuse else native.StableColumnClusters(sys.version_info >= (3, 12))
    result = state.extend(packed, max(3.0, median_height * 0.75))
    if result is None:
        if cache is not None and prefix_key is not None:
            cache.prefixes.pop(prefix_key, None)
        return _count_stable_columns_python(rows, median_height)
    if cache is not None:
        cache.results[key] = result
        if prefix_key is not None and (prefix is None or reuse):
            cache.prefixes[prefix_key] = _StableColumnPrefixState(row_ids, state)
    return result


def _count_stable_columns_python(
    rows: list[_VisualRow],
    median_height: float,
    cache: _StableColumnCache | None = None,
    *,
    allow_prefix_reuse: bool = False,
) -> tuple[int, float]:
    """分别聚类片段左边界、中心和右边界，并对严格前缀输入续算已有状态。"""

    row_ids = tuple(id(row) for row in rows)
    cache_key = (median_height, row_ids)
    if cache is not None:
        cached = cache.results.get(cache_key)
        if cached is not None:
            return cached

    tolerance = max(3.0, median_height * 0.75)
    clusters_by_alignment: dict[str, list[dict[str, Any]]]
    start_row_index = 0
    prefix_key: tuple[float, int] | None = None
    prefix_state: _StableColumnPrefixState | None = None
    reused_prefix = False

    if cache is not None and allow_prefix_reuse and row_ids:
        prefix_key = (median_height, row_ids[0])
        prefix_state = cache.prefixes.get(prefix_key)
        if (
            prefix_state is not None
            and len(prefix_state.row_ids) < len(row_ids)
            and row_ids[: len(prefix_state.row_ids)] == prefix_state.row_ids
        ):
            clusters_by_alignment = prefix_state.clusters_by_alignment
            start_row_index = len(prefix_state.row_ids)
            reused_prefix = True
        else:
            clusters_by_alignment = _new_stable_column_clusters()
    else:
        clusters_by_alignment = _new_stable_column_clusters()

    _extend_stable_column_clusters(
        clusters_by_alignment,
        rows[start_row_index:],
        start_row_index=start_row_index,
        tolerance=tolerance,
    )
    result = _stable_column_result(clusters_by_alignment, len(rows))

    if cache is not None:
        cache.results[cache_key] = result
        if allow_prefix_reuse and row_ids and prefix_key is not None:
            if prefix_state is None or reused_prefix:
                cache.prefixes[prefix_key] = _StableColumnPrefixState(
                    row_ids=row_ids,
                    clusters_by_alignment=clusters_by_alignment,
                )
    return result


def _merge_table_candidates(candidates: list[_TableCandidate]) -> list[_TableCandidate]:
    """合并同方向且明显重叠的横线候选，避免同一表格重复输出。"""

    merged: list[_TableCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if isinstance(candidate, _RuleCandidateDraft):
            candidate = candidate.materialize()
        target = next(
            (
                item
                for item in merged
                if item.angle == candidate.angle and _bbox_overlap_in_smaller(candidate.bbox, item.bbox) >= 0.2
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        if target.score >= 100 and candidate.score < 100:
            # A weak rule band cannot grow a confirmed grid across prose or
            # another table, even when a decorative page frame connects them.
            continue
        target.bbox = _bbox_union(target.bbox, candidate.bbox)
        target.local_bbox = _bbox_union(target.local_bbox, candidate.local_bbox)
        if target.core_bbox is None:
            target.core_bbox = candidate.core_bbox
        elif candidate.core_bbox is not None:
            target.core_bbox = _bbox_union(target.core_bbox, candidate.core_bbox)
        if isinstance(candidate.line_indices, _SharedLineIndexSet) and not isinstance(
            target.line_indices,
            _SharedLineIndexSet,
        ):
            # 首个高分候选可能来自闭合网格路径并持有普通 set；转换为候选的
            # 精确共享基底表示后，后续同网格候选即可合并差集而不重扫全部成员。
            target.line_indices = _SharedLineIndexSet.from_exact_values(
                candidate.line_indices.base,
                target.line_indices,
            )
        target.line_indices.update(candidate.line_indices)
        _merge_table_candidate_annotations(target, candidate)
        target.score = max(target.score, candidate.score)
    return sorted(merged, key=lambda item: (item.bbox[1], item.bbox[0]))


def _median_fragment_height(fragments: list[_Fragment]) -> float:
    """返回正向文本片段高度的中位数。"""

    heights = [
        fragment.local_bbox[3] - fragment.local_bbox[1]
        for fragment in fragments
        if fragment.local_bbox[3] > fragment.local_bbox[1]
    ]
    return max(0.1, float(statistics.median(heights)) if heights else 1.0)


def _merge_owned_table_candidates(candidates):
    """检测器独占候选先走自有原生流，特殊输入保持原有可变对象参考语义。"""
    from ._native_table_merge import merge_owned

    result = merge_owned(candidates)
    return _merge_table_candidates(candidates) if result is None else result


_NATIVE_DRAFT_MATERIALIZE = _RuleCandidateDraft.materialize
_SHORT_INTERVAL_CENTER_REFERENCE = _bbox_center_y
_COLUMN_GEOMETRY_REFERENCES = (
    _VisualRow.__getattribute__,
    _VisualRow.fragments,
    _Fragment.__getattribute__,
    _Fragment.local_bbox,
)

_NATIVE_MERGE_RULES = {
    function.__name__: function
    for function in (
        _merge_table_candidates,
        _expand_rule_table_candidate,
        _build_table_annotation,
        _collect_footnote_rows,
        _collect_caption_rows,
        _merge_table_candidate_annotations,
        _expand_candidates_to_connected_rule_grids,
        _prepared_rule_candidate_core,
        _bbox_area,
        _bbox_axis_overlap_ratio,
        _bbox_overlap_in_smaller,
        _bbox_union,
        _rotate_bbox_from_upright,
        _rotate_bbox_to_upright,
        _point_in_bbox,
        _bbox_center_x,
        _bbox_center_y,
    )
}
