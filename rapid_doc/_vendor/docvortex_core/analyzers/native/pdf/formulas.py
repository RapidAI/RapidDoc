"""按空间关系检测并物化原生 PDF 公式块。"""

from __future__ import annotations

import re
import math
import statistics
import unicodedata
from dataclasses import dataclass, replace
from contextvars import ContextVar
from functools import wraps
from typing import Any

from ....document.pdf._document import PDFPathInfo
from ....document.pdf.text._contracts import Char
from ....foundation._text import build_tagged_formula_content
from ....schema import BBox
from ...._compute_backend import get_native
from .geometry import (
    _bbox_axis_overlap_ratio,
    _bbox_center_x,
    _bbox_center_y,
    _bbox_distance,
    _bbox_intersects,
    _bbox_overlap_in_first,
    _bbox_overlap_in_smaller,
    _bbox_union,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _expand_bbox,
    _rotate_bbox_to_upright,
    _transform_axis_lines,
)
from .line_layout import (
    _connection_crosses_table,
    _infer_text_lanes,
    _line_effective_height,
    _line_style_scale,
    _line_tight_output_bbox,
    _lines_tight_output_bbox,
)
from .line_merging import _join_formula_visual_row, _merge_overlapping_inline_cluster
from .models import _AxisLine, _FormulaAnchor, _LineItem, _PageSource, _TextLane
from .native_text import _sanitize_pdf_control_text
from ._native_geometry import glyph_flags as _unmapped_glyph_flags
from .text_roles import has_prose, prose_residue, publication_text
from .layout_evidence import build_layout_evidence

_FORMULA_NUMBER_SUFFIX_RE = re.compile(r"^(?P<prefix>.*?)(?P<marker>[(（﹙][^()（）﹙﹚\r\n]+[)）﹚])\s*$")
_FORMULA_NUMBER_MARKER_RE = re.compile(
    r"^[（(﹙]\s*(?:(?:[A-Za-z][.]?)?\d+(?:[.\-]\d+)*)\s*[)）﹚]$",
)
_FORMULA_OPERATOR_CHARS = frozenset("=<>≤≥≠≈∝∈∑∫√±×÷−")


_FORMULA_PAGE_MARGIN_RATIO = 0.05
_VECTOR_FORMULA_COMPLEX_SEGMENTS = 8
_VECTOR_FORMULA_MIN_PATHS = 5
_VECTOR_FORMULA_MIN_COMPLEX_PATHS = 5
_VECTOR_FORMULA_MIN_COMPLEX_RATIO = 0.5
_VECTOR_FORMULA_NUMBER_MIN_PATHS = 3
_VECTOR_FORMULA_NUMBER_MAX_PATHS = 6

_FORMULA_TEXT_FEATURES = ContextVar("docvortex_formula_text_features", default=None)


def _formula_scope(function, fresh):
    """页面入口隔离纯文字特征，内部 canonical 重放复用相同不可变字符串的计算。"""

    @wraps(function)
    def scoped(*args, **kwargs):
        """仅缓存纯字符串特征；成员、字体和几何在每个阶段重新读取。"""
        current = _FORMULA_TEXT_FEATURES.get()
        token = _FORMULA_TEXT_FEATURES.set({} if fresh or current is None else current)
        try:
            return function(*args, **kwargs)
        finally:
            _FORMULA_TEXT_FEATURES.reset(token)

    return scoped


def _formula_page_scope(function):
    """独立调用建立页面上下文，处于同页完成阶段时复用纯文字特征。"""
    return _formula_scope(function, False)


def _formula_fresh_page_scope(function):
    """每页最终处理建立独立上下文，完成或异常后立即释放有界特征表。"""
    return _formula_scope(function, True)


def _formula_text_feature(function):
    """只按本次页面输入的普通字符串复用纯特征，文字修改后自然使用新键。"""

    @wraps(function)
    def feature(text):
        """自定义字符串和没有页面上下文的调用仍直接执行参考函数。"""
        cache = _FORMULA_TEXT_FEATURES.get()
        if cache is None or type(text) is not str:
            return function(text)
        key = (function, text)
        if key not in cache:
            if len(cache) >= 4096:
                cache.pop(next(iter(cache)))
            cache[key] = function(text)
        return cache[key]

    return feature


@dataclass(slots=True)
class _VectorPathComponent:
    """保存同栏邻接 Path 形成的矢量组件。"""

    lane_index: int
    path_infos: list[PDFPathInfo]
    bbox: BBox


@dataclass(slots=True)
class _VectorFormulaCandidate:
    """保存已通过主体校验、等待吸收横线和编号的矢量公式。"""

    lane_index: int
    bbox: BBox
    path_source_indices: set[int]
    has_number: bool = False


def _build_vector_formula_blocks(
    source: _PageSource,
    container_blocks: list[dict[str, Any]],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """从根层填充 Path 构建空内容公式，并唯一认领可提取的公式编号。"""

    available_lines = [line for line in source.lines if line.angle == 0 and line.source_index not in claimed_line_indices]
    if len(available_lines) < 3 or not source.path_infos:
        return [], set()

    line_geometry = [(line, line.bbox) for line in available_lines]
    effective_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    median_height = statistics.median(effective_heights) if effective_heights else 0.0
    if median_height <= 0:
        return [], set()
    lanes = [
        lane
        for lane in _infer_text_lanes(line_geometry, source.page_size[0], median_height)
        if not lane.is_span and len(lane.lines) >= 3
    ]
    if not lanes:
        return [], set()

    components = _build_vector_path_components(
        source.path_infos,
        lanes,
        median_height,
    )
    container_bboxes = [bbox for block in container_blocks if (bbox := _coerce_bbox(block.get("bbox"))) is not None]
    inline_sources = {
        id(component): _inline_vector_formula_prose_sources(
            component, lanes[component.lane_index], median_height, container_bboxes
        )
        for component in components
    }
    candidates = [
        _VectorFormulaCandidate(
            lane_index=component.lane_index,
            bbox=component.bbox,
            path_source_indices={item.source_index for item in component.path_infos},
        )
        for component in components
        if inline_sources[id(component)]
        or _is_vector_formula_core(
            component,
            lanes[component.lane_index],
            median_height,
            source.page_size,
            container_bboxes,
        )
    ]
    if not candidates:
        return [], set()

    _attach_vector_formula_rules(candidates, components, median_height)
    _attach_vector_formula_path_numbers(candidates, components, lanes, median_height)
    claimed_number_indices = _attach_vector_formula_text_numbers(
        candidates,
        lanes,
        median_height,
        claimed_line_indices,
    )
    _join_open_vector_equation_rows(candidates, components, available_lines, median_height)

    padding = min(1.5, 0.1 * median_height)
    blocks: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item.bbox[1], item.bbox[0])):
        padded_bbox = _clip_bbox(
            (
                candidate.bbox[0] - padding,
                candidate.bbox[1] - padding,
                candidate.bbox[2] + padding,
                candidate.bbox[3] + padding,
            ),
            source.page_size,
        )
        if padded_bbox is None:
            continue
        paths = [path for path in source.path_infos if path.source_index in candidate.path_source_indices]
        math_evidence = candidate.has_number or _vector_has_math_evidence(paths, padded_bbox, available_lines, median_height)
        decoration_shape = _vector_has_decoration_shape(paths)
        edge = (
            _bbox_center_y(padded_bbox) < 0.085 * source.page_size[1]
            or _bbox_center_y(padded_bbox) > 0.90 * source.page_size[1]
        )
        publication_regions = source.publication_bboxes + [line.bbox for line in available_lines if publication_text(line.text)]
        publication = any(
            _bbox_distance(bounds, padded_bbox) <= 8 * median_height
            or _bbox_axis_overlap_ratio(bounds, padded_bbox, axis="y") >= 0.5
            for bounds in publication_regions
        )
        blocks.append(
            {
                "type": "image" if edge and decoration_shape and publication and not math_evidence else "equation",
                "bbox": padded_bbox,
                "angle": 0,
                "content": "",
                "_vector_shape": _vector_shape_signature(paths, padded_bbox),
                "_vector_edge_decoration": edge and decoration_shape and not math_evidence,
            }
        )
        if not blocks[-1]["_vector_edge_decoration"]:
            blocks[-1].pop("_vector_shape")
            blocks[-1].pop("_vector_edge_decoration")
        hosts = {
            index
            for component in components
            if set(path.source_index for path in component.path_infos) <= candidate.path_source_indices
            for index in inline_sources[id(component)]
        }
        if hosts:
            blocks[-1]["_inline_formula_prose_sources"] = hosts
    return blocks, claimed_number_indices


def _inline_vector_formula_prose_sources(component, lane, em, containers) -> set[int]:
    """原生正文两侧留给复杂矢量字形的槽位可保留公式裁图；单字装饰和容器内标签排除。"""
    box, paths = component.bbox, component.path_infos
    width, height = box[2] - box[0], box[3] - box[1]
    if (
        not 0.45 * em <= height <= 1.6 * em
        or width > 0.45 * (lane.right - lane.left)
        or any(_bbox_overlap_in_smaller(box, bounds) >= 0.1 for bounds in containers)
        or any(path.fill_rgba is not None and max(path.fill_rgba[:3]) - min(path.fill_rgba[:3]) > 20 for path in paths)
    ):
        return set()
    complex_paths = [path for path in paths if path.segment_count >= 20]
    scripted = any(
        (
            large.bbox[3] - large.bbox[1] >= 1.4 * (small.bbox[3] - small.bbox[1])
            and abs(_bbox_center_y(large.bbox) - _bbox_center_y(small.bbox)) >= 0.1 * em
            and -0.2 * em <= small.bbox[0] - large.bbox[2] <= 0.5 * em
        )
        or (
            0.3 * em <= min(large.bbox[3] - large.bbox[1], small.bbox[3] - small.bbox[1])
            and max(large.bbox[3] - large.bbox[1], small.bbox[3] - small.bbox[1]) <= 0.6 * em
            and abs(large.bbox[0] - small.bbox[0]) <= 0.15 * em
            and abs(_bbox_center_y(large.bbox) - _bbox_center_y(small.bbox)) >= 0.5 * em
        )
        for large in complex_paths
        for small in complex_paths
        if large is not small
    )
    if not (len(complex_paths) >= 8 and width >= 3 * em or len(complex_paths) >= 2 and width <= 2 * em and scripted):
        return set()
    neighbors = [
        line
        for line, bounds in lane.lines
        if line.angle == 0
        and line.semantic_type is None
        and not line.caption_start
        and line.font_signature is not None
        and 0.85 <= _line_effective_height(line, bounds) / em <= 1.15
        and _bbox_axis_overlap_ratio(bounds, box, axis="y") >= 0.5
        and _bbox_overlap_in_smaller(bounds, box) < 0.1
    ]
    left = [line for line in neighbors if 0 <= box[0] - line.bbox[2] <= 0.8 * em]
    right = [line for line in neighbors if 0 <= line.bbox[0] - box[2] <= 0.8 * em]
    hosts = left + right
    if (
        not hosts
        or len(left) > 1
        or len(right) > 1
        or len(re.findall(r"\b[A-Za-z]{2,}\b", " ".join(line.text for line in hosts))) < 4
    ):
        return set()
    if not left and (
        not right
        or not right[0].text.lstrip().startswith((",", ".", ";", ":", ")"))
        or any(line.bbox[2] < box[0] for line in neighbors)
    ):
        return set()
    if not right and (
        not left or left[0].bbox[2] - left[0].bbox[0] < 0.65 * (lane.right - lane.left) or box[2] < lane.right - em
    ):
        return set()
    return {line.source_index for line in hosts}


def _join_open_vector_equation_rows(candidates, components, text_lines, em) -> None:
    """短左式末尾居中扁平运算字形加紧接长右式确认跨行等式，正文和独立编号阻断合并。"""
    for first in list(candidates):
        if first not in candidates or first.has_number:
            continue
        box = first.bbox
        glyphs = [
            path for component in components for path in component.path_infos if path.source_index in first.path_source_indices
        ]
        if not glyphs or not 8 * em <= box[2] - box[0] <= 15 * em:
            continue
        last = max(glyphs, key=lambda path: path.bbox[2]).bbox
        if (
            not 0.65 * em <= last[2] - last[0] <= 1.2 * em
            or not 0.12 * em <= last[3] - last[1] <= 0.4 * em
            or last[2] - last[0] < 2.5 * (last[3] - last[1])
            or abs(_bbox_center_y(last) - _bbox_center_y(box)) > 0.2 * em
        ):
            continue
        following = [
            second
            for second in candidates
            if second is not first
            and not second.has_number
            and second.lane_index == first.lane_index
            and abs(second.bbox[0] - box[0]) <= 0.5 * em
            and 0 <= second.bbox[1] - box[3] <= 1.3 * em
            and second.bbox[2] - second.bbox[0] >= 3 * (box[2] - box[0])
            and not any(
                box[3] < _bbox_center_y(line.bbox) < second.bbox[1]
                and _bbox_axis_overlap_ratio(line.bbox, second.bbox, axis="x") >= 0.2
                for line in text_lines
            )
        ]
        if len(following) == 1:
            second = following[0]
            first.bbox = _bbox_union(box, second.bbox)
            first.path_source_indices.update(second.path_source_indices)
            candidates.remove(second)


def _vector_shape_signature(paths: list[PDFPathInfo], bbox: BBox) -> tuple:
    """保存与颜色及绝对位置无关的路径结构，避免同位置的不同图形被当成重复装饰。"""
    width, height = max(0.1, bbox[2] - bbox[0]), max(0.1, bbox[3] - bbox[1])
    return tuple(
        sorted(
            (
                path.segment_count,
                path.fill_visible,
                path.stroke_visible,
                round((path.bbox[0] - bbox[0]) / width, 2),
                round((path.bbox[1] - bbox[1]) / height, 2),
                round((path.bbox[2] - bbox[0]) / width, 2),
                round((path.bbox[3] - bbox[1]) / height, 2),
            )
            for path in paths
        )
    )


def _vector_has_decoration_shape(paths: list[PDFPathInfo]) -> bool:
    """混合图案字标或多行等高字标提供装饰形证据；仍须出版上下文或跨页重复且无数学证据。"""
    heights = [p.bbox[3] - p.bbox[1] for p in paths if p.bbox[3] > p.bbox[1]]
    if len(heights) < 5:
        return False
    typical = statistics.median(heights)
    rows: list[list[PDFPathInfo]] = []
    for path in sorted(paths, key=lambda item: _bbox_center_y(item.bbox)):
        if (
            rows
            and abs(_bbox_center_y(path.bbox) - statistics.median(_bbox_center_y(p.bbox) for p in rows[-1])) <= 0.2 * typical
        ):
            rows[-1].append(path)
        else:
            rows.append([path])
    wordmark = 2 <= len(rows) <= 3 and all(len(row) >= 4 for row in rows) and max(heights) <= 1.5 * min(heights)
    small = sorted(heights)[len(heights) // 4]
    large = sorted(heights)[3 * len(heights) // 4]
    overlaps = sum(_bbox_overlap_in_smaller(a.bbox, b.bbox) >= 0.25 for i, a in enumerate(paths) for b in paths[i + 1 :])
    large_paths = [p for p in paths if p.bbox[3] - p.bbox[1] >= large]
    irregular = (
        len(large_paths) >= 3 and max(p.bbox[3] for p in large_paths) - min(p.bbox[3] for p in large_paths) >= 0.6 * large
    )
    return wordmark or small > 0 and large >= 2 * small and (overlaps >= 2 or irregular)


def _vector_has_math_evidence(paths: list[PDFPathInfo], bbox: BBox, lines: list[_LineItem], em: float) -> bool:
    """公式编号、可读数学内容和上下居中的分数结构优先于任何装饰判断。"""
    if any(
        (
            _standalone_formula_number_marker(line.text)
            or _formula_line_has_math_operator(line.text)
            and not _has_sentence_words(line.text)
        )
        and _bbox_distance(line.bbox, bbox) <= 2 * em
        for line in lines
    ):
        return True
    rules = [p.bbox for p in paths if p.bbox[3] - p.bbox[1] <= 0.2 * em and p.bbox[2] - p.bbox[0] >= 0.4 * em]
    for rule in rules:
        neighbors = [p.bbox for p in paths if abs(_bbox_center_x(p.bbox) - _bbox_center_x(rule)) <= 0.25 * (rule[2] - rule[0])]
        if any(0 <= rule[1] - b[3] <= em for b in neighbors) and any(0 <= b[1] - rule[3] <= em for b in neighbors):
            return True
    return False


def classify_repeated_vector_decorations(pages) -> None:
    """跨页同时核对结构签名和归一化几何，只重标缺少数学证据的装饰形候选。"""
    groups = {}
    for index, page in enumerate(pages):
        for block in page.fixed_blocks:
            if not block.get("_vector_edge_decoration") or not block.get("_vector_shape"):
                continue
            bounds = tuple(round(v / page.page_size[i % 2], 2) for i, v in enumerate(block["bbox"]))
            groups.setdefault((block["_vector_shape"], bounds), []).append((index, block))
    for members in groups.values():
        if len({index for index, _ in members}) >= 2:
            for _, block in members:
                block["type"] = "image"


def _build_vector_path_components(
    path_infos: list[PDFPathInfo],
    lanes: list[_TextLane],
    median_height: float,
) -> list[_VectorPathComponent]:
    """按文本栏带筛选矢量字形，并用空间网格生成局部连通组件。"""

    members_by_lane: dict[int, list[PDFPathInfo]] = {}
    for path_info in path_infos:
        if path_info.form_depth != 0 or not path_info.fill_visible or path_info.stroke_visible:
            continue
        lane_index = _assign_vector_path_lane(path_info.bbox, lanes, median_height)
        if lane_index is None:
            continue
        if not _is_vector_formula_path_member(
            path_info.bbox,
            lanes[lane_index],
            median_height,
        ):
            continue
        members_by_lane.setdefault(lane_index, []).append(path_info)

    components: list[_VectorPathComponent] = []
    for lane_index, members in members_by_lane.items():
        components.extend(
            _connect_vector_path_members(
                members,
                lane_index,
                median_height,
            )
        )
    return sorted(
        components,
        key=lambda item: (item.bbox[1], item.bbox[0], item.path_infos[0].source_index),
    )


def _assign_vector_path_lane(
    bbox: BBox,
    lanes: list[_TextLane],
    median_height: float,
) -> int | None:
    """按中心点和水平覆盖率把 Path 唯一分配给一个正文栏带。"""

    center_x = _bbox_center_x(bbox)
    path_width = max(0.1, bbox[2] - bbox[0])
    tolerance = 0.75 * median_height
    matches: list[tuple[float, float, int]] = []
    for lane_index, lane in enumerate(lanes):
        if not lane.left - tolerance <= center_x <= lane.right + tolerance:
            continue
        overlap = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left))
        coverage = overlap / path_width
        lane_center = (lane.left + lane.right) / 2.0
        matches.append((-coverage, abs(center_x - lane_center), lane_index))
    return min(matches)[2] if matches else None


def _is_vector_formula_path_member(
    bbox: BBox,
    lane: _TextLane,
    median_height: float,
) -> bool:
    """保留小字形轮廓和细横线，过滤跨栏或过大的普通矢量对象。"""

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    lane_width = max(0.1, lane.right - lane.left)
    is_glyph = width <= 3.0 * median_height and height <= 3.0 * median_height
    is_formula_rule = height <= 0.2 * median_height and width <= lane_width + median_height
    return is_glyph or is_formula_rule


def _connect_vector_path_members(
    members: list[PDFPathInfo],
    lane_index: int,
    median_height: float,
) -> list[_VectorPathComponent]:
    """用扩张 bbox 的网格邻接和并查集连接同栏 Path，避免全量两两比较。"""

    if not members:
        return []
    ordered = sorted(members, key=lambda item: item.source_index)
    parents = list(range(len(ordered)))

    def find(index: int) -> int:
        """查找并压缩一个 Path 的并查集根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first: int, second: int) -> None:
        """合并两个相交扩张框所属的连通分量。"""

        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    margin = 0.5 * median_height
    cell_size = max(1.0, median_height)
    expanded_bboxes = [_expand_bbox(item.bbox, margin) for item in ordered]
    grid: dict[tuple[int, int], list[int]] = {}
    seen_pairs: set[tuple[int, int]] = set()
    for index, bbox in enumerate(expanded_bboxes):
        start_x = int(bbox[0] // cell_size)
        end_x = int(bbox[2] // cell_size)
        start_y = int(bbox[1] // cell_size)
        end_y = int(bbox[3] // cell_size)
        for cell_x in range(start_x, end_x + 1):
            for cell_y in range(start_y, end_y + 1):
                cell = (cell_x, cell_y)
                for other_index in grid.get(cell, []):
                    pair = (other_index, index)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if _bbox_intersects(bbox, expanded_bboxes[other_index]):
                        merge(index, other_index)
                grid.setdefault(cell, []).append(index)

    grouped: dict[int, list[PDFPathInfo]] = {}
    for index, path_info in enumerate(ordered):
        grouped.setdefault(find(index), []).append(path_info)
    return [
        _VectorPathComponent(
            lane_index=lane_index,
            path_infos=group,
            bbox=_bbox_union_many([item.bbox for item in group]),
        )
        for group in grouped.values()
    ]


def _is_vector_formula_core(
    component: _VectorPathComponent,
    lane: _TextLane,
    median_height: float,
    page_size: tuple[float, float],
    container_bboxes: list[BBox],
) -> bool:
    """按复杂度、尺寸、正文碰撞和容器优先级校验公式主体组件。"""

    path_count = len(component.path_infos)
    complex_count = sum(item.segment_count >= _VECTOR_FORMULA_COMPLEX_SEGMENTS for item in component.path_infos)
    if (
        path_count < _VECTOR_FORMULA_MIN_PATHS
        or complex_count < _VECTOR_FORMULA_MIN_COMPLEX_PATHS
        and not (complex_count >= 4 and _vector_has_math_evidence(component.path_infos, component.bbox, [], median_height))
        or complex_count / path_count < _VECTOR_FORMULA_MIN_COMPLEX_RATIO
    ):
        return False

    bbox = component.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if not (width >= 2.5 * median_height and 0.6 * median_height <= height <= 8.0 * median_height and width >= 1.4 * height):
        return False
    if _is_formula_component_in_page_margin(bbox, page_size[1]):
        return False
    if any(_bbox_overlap_in_smaller(bbox, container_bbox) >= 0.5 for container_bbox in container_bboxes):
        return False
    return not any(_vector_formula_collides_with_text(bbox, line, line_bbox, median_height) for line, line_bbox in lane.lines)


def _is_formula_component_in_page_margin(bbox: BBox, page_height: float) -> bool:
    """仅当公式组件完全落在页面顶部或底部边缘带时排除。"""

    margin = _FORMULA_PAGE_MARGIN_RATIO * page_height
    return bbox[3] <= margin or bbox[1] >= page_height - margin


def _vector_formula_collides_with_text(
    formula_bbox: BBox,
    line: _LineItem,
    line_bbox: BBox,
    median_height: float,
) -> bool:
    """排除覆盖正文或紧贴正文同行的 Path 组件，独立公式编号除外。"""

    if _standalone_formula_number_marker(line.text) is not None:
        return False
    if _bbox_overlap_in_first(formula_bbox, line_bbox) >= 0.2:
        return True
    horizontal_gap = max(
        formula_bbox[0] - line_bbox[2],
        line_bbox[0] - formula_bbox[2],
        0.0,
    )
    return _bbox_axis_overlap_ratio(formula_bbox, line_bbox, axis="y") >= 0.5 and horizontal_gap <= median_height


def _attach_vector_formula_rules(
    candidates: list[_VectorFormulaCandidate],
    components: list[_VectorPathComponent],
    median_height: float,
) -> None:
    """把靠近公式主体且横向覆盖充分的孤立细横线唯一并入主体。"""

    used_sources = {source_index for candidate in candidates for source_index in candidate.path_source_indices}
    for component in components:
        component_sources = {item.source_index for item in component.path_infos}
        if component_sources & used_sources or not all(
            item.bbox[3] - item.bbox[1] <= 0.2 * median_height for item in component.path_infos
        ):
            continue
        matches = [
            (
                _bbox_distance(candidate.bbox, component.bbox),
                abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(component.bbox)),
                candidate_index,
            )
            for candidate_index, candidate in enumerate(candidates)
            if candidate.lane_index == component.lane_index
            and _bbox_distance(candidate.bbox, component.bbox) <= 0.5 * median_height
            and _bbox_axis_overlap_ratio(candidate.bbox, component.bbox, axis="x") >= 0.5
        ]
        if not matches:
            continue
        candidate = candidates[min(matches)[2]]
        candidate.bbox = _bbox_union(candidate.bbox, component.bbox)
        candidate.path_source_indices.update(component_sources)
        used_sources.update(component_sources)


def _attach_vector_formula_path_numbers(
    candidates: list[_VectorFormulaCandidate],
    components: list[_VectorPathComponent],
    lanes: list[_TextLane],
    median_height: float,
) -> None:
    """把栏右缘的小型复杂 Path 组件作为公式编号并入唯一主体。"""

    used_sources = {source_index for candidate in candidates for source_index in candidate.path_source_indices}
    for component in components:
        component_sources = {item.source_index for item in component.path_infos}
        local_number = _is_local_vector_number(component, median_height)
        if component_sources & used_sources or not (
            local_number
            or _is_vector_formula_number_component(
                component,
                lanes[component.lane_index],
                median_height,
            )
        ):
            continue
        matches = _vector_formula_number_matches(
            component.bbox,
            component.lane_index,
            candidates,
            median_height,
        )
        if not matches:
            continue
        if local_number and not _is_vector_formula_number_component(component, lanes[component.lane_index], median_height):
            matches = [match for match in matches if match[2] <= 4]
            if not matches:
                continue
        candidate = candidates[min(matches)[3]]
        candidate.bbox = _bbox_union(candidate.bbox, component.bbox)
        candidate.path_source_indices.update(component_sources)
        candidate.has_number = True
        used_sources.update(component_sources)


def _is_local_vector_number(component: _VectorPathComponent, em: float) -> bool:
    """独立小组件两端同尺度的狭长轮廓提供括号编号证据，允许编号紧随短公式而不在栏右缘。"""
    paths = sorted(component.path_infos, key=lambda path: path.bbox[0])
    if not 3 <= len(paths) <= 6 or not all(path.segment_count >= _VECTOR_FORMULA_COMPLEX_SEGMENTS for path in paths):
        return False
    first, last = paths[0].bbox, paths[-1].bbox
    return (
        0.5 * em <= component.bbox[2] - component.bbox[0] <= 2 * em
        and 0.6 * em <= component.bbox[3] - component.bbox[1] <= 1.5 * em
        and all(bounds[3] - bounds[1] >= 2.5 * (bounds[2] - bounds[0]) for bounds in (first, last))
        and abs(first[1] - last[1]) <= 0.15 * em
        and abs(first[3] - last[3]) <= 0.15 * em
        and abs(first[2] - first[0] - (last[2] - last[0])) <= 0.15 * em
    )


def _is_vector_formula_number_component(
    component: _VectorPathComponent,
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别位于栏右缘、尺寸接近正文行高的全复杂路径编号组件。"""

    path_count = len(component.path_infos)
    bbox = component.bbox
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (
        _VECTOR_FORMULA_NUMBER_MIN_PATHS <= path_count <= _VECTOR_FORMULA_NUMBER_MAX_PATHS
        and all(item.segment_count >= _VECTOR_FORMULA_COMPLEX_SEGMENTS for item in component.path_infos)
        and 0.5 * median_height <= width <= 2.0 * median_height
        and 0.6 * median_height <= height <= 1.4 * median_height
        and abs(lane.right - bbox[2]) <= 1.5 * median_height
    )


def _vector_formula_number_matches(
    number_bbox: BBox,
    lane_index: int,
    candidates: list[_VectorFormulaCandidate],
    median_height: float,
    *,
    allow_left: bool = False,
) -> list[tuple[float, float, float, int]]:
    """返回编号可关联的公式主体及稳定排序分值。"""

    matches: list[tuple[float, float, float, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        if candidate.has_number or candidate.lane_index != lane_index:
            continue
        vertical_overlap = _bbox_axis_overlap_ratio(candidate.bbox, number_bbox, axis="y")
        left_number = (
            allow_left and number_bbox[2] <= candidate.bbox[0] and candidate.bbox[0] - number_bbox[2] <= 3 * median_height
        )
        if (number_bbox[0] < candidate.bbox[2] and not left_number) or vertical_overlap < 0.6:
            continue
        center_distance = abs(_bbox_center_y(candidate.bbox) - _bbox_center_y(number_bbox))
        horizontal_gap = candidate.bbox[0] - number_bbox[2] if left_number else max(0.0, number_bbox[0] - candidate.bbox[2])
        matches.append((-vertical_overlap, center_distance, horizontal_gap / max(0.1, median_height), candidate_index))
    return matches


def _attach_vector_formula_text_numbers(
    candidates: list[_VectorFormulaCandidate],
    lanes: list[_TextLane],
    median_height: float,
    claimed_line_indices: set[int],
) -> set[int]:
    """关联可提取的独立公式编号并认领其文本身份，防止重复输出。"""

    claimed: set[int] = set()
    for lane_index, lane in enumerate(lanes):
        for line, bbox in sorted(lane.lines, key=lambda item: (item[1][1], item[1][0])):
            if line.source_index in claimed_line_indices or not _FORMULA_NUMBER_MARKER_RE.fullmatch(line.text.strip()):
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            at_right = (
                0.5 * median_height <= width <= 2.0 * median_height
                and 0.6 * median_height <= height <= 1.4 * median_height
                and abs(lane.right - bbox[2]) <= 1.5 * median_height
            )
            at_left = (
                0.5 * median_height <= width <= 4.0 * median_height
                and 0.6 * median_height <= height <= 1.4 * median_height
                and abs(lane.left - bbox[0]) <= 1.5 * median_height
                and line.semantic_type is None
            )
            if not (at_right or at_left):
                continue
            matches = _vector_formula_number_matches(
                bbox,
                lane_index,
                candidates,
                median_height,
                allow_left=at_left,
            )
            if at_left and not at_right:
                # 左侧近编号只能有一个同栏、同高度的数学主体，防止普通条目误绑相邻公式。
                matches = [match for match in matches if candidates[match[3]].bbox[0] >= bbox[2]]
                if len(matches) != 1:
                    continue
            if not matches:
                continue
            candidate = candidates[min(matches)[3]]
            candidate.bbox = _bbox_union(candidate.bbox, bbox)
            candidate.has_number = True
            claimed.add(line.source_index)
    return claimed


@_formula_text_feature
def _standalone_formula_number_marker(text: str) -> str | None:
    """仅接受整行由圆括号公式编号构成的文本，不接纳带正文前缀的后缀。"""

    parts = _split_trailing_formula_number(text)
    if parts is None:
        return None
    prefix, marker = parts
    return marker if not prefix else None


@_formula_page_scope
def _build_formula_like_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    *,
    drawing_lines: list[_AxisLine] | None = None,
) -> tuple[list[dict[str, Any]], list[_LineItem]]:
    """仅依据栏带、右侧短锚点和空间连通关系聚合公式状区域。"""

    blocks, claimed_source_indices = _build_split_visual_row_formula_blocks(
        lines,
        table_bboxes,
        page_size,
    )
    paragraph_lines: list[_LineItem] = []
    for angle in sorted({line.angle for line in lines}):
        angle_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.source_index not in claimed_source_indices
        ]
        if len(angle_geometry) < 2:
            continue
        effective_heights = [_line_effective_height(line, bbox) for line, bbox in angle_geometry]
        median_height = statistics.median(effective_heights) if effective_heights else 1.0
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_horizontal_rules = [
            rule.bbox
            for rule in _transform_axis_lines(
                drawing_lines or [],
                page_size,
                angle,
            )
            if rule.orientation == "horizontal"
        ]
        lanes = _infer_text_lanes(angle_geometry, local_page_width, median_height)
        for lane in lanes:
            if lane.is_span:
                continue
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            dominant_body_font = _infer_formula_body_font(
                lane,
                median_height,
            )
            for line, bbox in list(lane.lines):
                if not _is_single_line_numbered_formula(
                    (line, bbox),
                    lane,
                    median_height,
                ):
                    continue
                members = _expand_single_line_numbered_formula_members(
                    (line, bbox),
                    lane,
                    claimed_source_indices,
                    table_bboxes,
                    dominant_body_font,
                    median_height,
                )
                block = _formula_members_to_block(
                    members,
                    page_size,
                    angle,
                    include_member_ids=True,
                    anchor_source_index=line.source_index,
                )
                if block is None:
                    continue
                blocks.append(block)
                claimed_source_indices.update(member_line.source_index for member_line, _member_bbox in members)
            lane.lines = [item for item in lane.lines if item[0].source_index not in claimed_source_indices]
            for line, bbox in list(lane.lines):
                if (
                    (
                        line.compact_formula_cluster
                        and not _compact_cluster_has_nearby_number_anchor(
                            (line, bbox),
                            lane,
                            median_height,
                        )
                        and _is_isolated_compact_formula_cluster(
                            (line, bbox),
                            lane,
                            median_height,
                        )
                    )
                    or _is_isolated_unnumbered_formula_line(
                        (line, bbox),
                        lane,
                        median_height,
                        dominant_body_font,
                    )
                ) and not _is_formula_component_in_page_margin(
                    bbox,
                    local_page_height,
                ):
                    content = _sanitize_pdf_control_text(
                        line.text,
                        preserve_newlines=False,
                    ).strip()
                    if not content:
                        continue
                    block = {
                        "type": "equation",
                        "bbox": line.bbox,
                        "angle": angle,
                        "content": content,
                        "_formula_members": [line.source_index],
                    }
                    tight_output_bbox = _line_tight_output_bbox(
                        line,
                        page_size,
                    )
                    if tight_output_bbox is not None:
                        block["_tight_output_bbox"] = tight_output_bbox
                    blocks.append(block)
                    claimed_source_indices.add(line.source_index)
            lane.lines = [item for item in lane.lines if item[0].source_index not in claimed_source_indices]
            if len(lane.lines) < 2:
                continue
            anchors = _find_formula_spatial_anchors(
                lane,
                median_height,
                dominant_body_font,
            )
            if not anchors:
                continue
            anchor_centers = [_bbox_center_y(anchor.bbox) for anchor in anchors]
            lane_top = min(bbox[1] for _line, bbox in lane.lines)
            lane_bottom = max(bbox[3] for _line, bbox in lane.lines)
            for anchor_index, anchor in enumerate(anchors):
                anchor_line = anchor.line
                if anchor_line.source_index in claimed_source_indices:
                    continue
                band_top = lane_top
                band_bottom = lane_bottom
                if anchor_index > 0:
                    band_top = max(
                        band_top,
                        (anchor_centers[anchor_index - 1] + anchor_centers[anchor_index]) / 2.0,
                    )
                if anchor_index + 1 < len(anchors):
                    band_bottom = min(
                        band_bottom,
                        (anchor_centers[anchor_index] + anchor_centers[anchor_index + 1]) / 2.0,
                    )
                members = _grow_formula_spatial_component(
                    lane,
                    anchor,
                    band_top,
                    band_bottom,
                    claimed_source_indices,
                    table_bboxes,
                    dominant_body_font,
                    median_height,
                )
                has_isolated_numbered_fraction = _formula_component_has_isolated_numbered_fraction(
                    members,
                    lane,
                    median_height,
                    local_horizontal_rules,
                )
                if (
                    _formula_component_has_left_prose(
                        members,
                        lane,
                        median_height,
                    )
                    and not has_isolated_numbered_fraction
                ):
                    for member_line, _member_bbox in members:
                        member_line.paragraph_formula_context = True
                    if _fragmented_left_prose(members, lane, median_height):
                        paragraph_lines.append(_merge_paragraph_formula_members(members, page_size, median_height))
                        claimed_source_indices.update(line.source_index for line, _bbox in members)
                    continue
                if len(members) < 2:
                    continue
                if (
                    len(members) == 2
                    and _bbox_axis_overlap_ratio(
                        members[0][1],
                        members[1][1],
                        axis="y",
                    )
                    < 0.2
                    and not any(
                        _is_wide_tagged_formula_member(
                            anchor_line,
                            member_line,
                            member_bbox,
                            max(0.1, lane.right - lane.left),
                        )
                        for member_line, member_bbox in members
                        if member_line is not anchor_line
                    )
                ):
                    continue
                component_bbox = _bbox_union_many([member_bbox for _member_line, member_bbox in members])
                if _is_formula_component_in_page_margin(
                    component_bbox,
                    local_page_height,
                ):
                    continue
                block = _formula_members_to_block(
                    members,
                    page_size,
                    angle,
                    include_member_ids=True,
                    anchor_source_index=anchor_line.source_index,
                )
                if block is None:
                    continue
                blocks.append(block)
                claimed_source_indices.update(line.source_index for line, _bbox in members)

    recovered, recovered_indices = _build_mixed_body_display_formulas(
        [line for line in lines if line.source_index not in claimed_source_indices],
        table_bboxes,
        page_size,
    )
    blocks.extend(recovered)
    claimed_source_indices.update(recovered_indices)
    # 两次紧邻的连通重放不改写字符、字体或原生变量证据，同次页面输入只求值一次。
    detached_prose_flags = {id(line): _detached_math_line_has_prose(line) for line in lines}
    detached_blocks, detached_sources = _recover_detached_display_components(
        [line for line in lines if line.source_index not in claimed_source_indices],
        table_bboxes,
        page_size,
        _prose_flags=detached_prose_flags,
    )
    blocks.extend(detached_blocks)
    claimed_source_indices.update(detached_sources)
    # 完整二维带只替换缺失成员或重复切割；已有完整公式保留原编号和成员序列。
    whole_blocks, _whole_sources = _recover_detached_display_components(
        lines,
        table_bboxes,
        page_size,
        drawing_lines=drawing_lines,
        _prose_flags=detached_prose_flags,
    )
    for whole in whole_blocks:
        members = set(whole.get("_formula_members", []))
        intersecting = [block for block in blocks if members.intersection(block.get("_formula_members", []))]
        previous_members = {index for block in intersecting for index in block.get("_formula_members", [])}
        if len(intersecting) == 1 and members <= previous_members:
            continue
        if any(not set(block.get("_formula_members", [])) <= members for block in intersecting):
            continue
        blocks = [block for block in blocks if block not in intersecting]
        blocks.append(whole)
        claimed_source_indices.update(members)
    retained = []
    for block in blocks:
        member_ids = set(block.get("_formula_members", []))
        bbox = block.get("_tight_output_bbox", block["bbox"])
        hosts = [
            line
            for line in lines
            if line.source_index not in member_ids
            and _has_sentence_words(line.text)
            and line.source_index not in claimed_source_indices
            and line.ink_bbox is not None
            and _bbox_axis_overlap_ratio(bbox, line.ink_bbox or line.bbox, axis="y") >= 0.25
            and max(0.0, bbox[0] - line.bbox[2], line.bbox[0] - bbox[2]) <= 3 * _line_effective_height(line, line.bbox)
        ]
        if member_ids and hosts and "\\tag{" not in block["content"] and not block.get("_spatial_display_band"):
            claimed_source_indices.difference_update(member_ids)
            for line in lines:
                if line.source_index in member_ids:
                    line.paragraph_formula_context = True
                    line.formula_candidate_only = False
                    line.compact_formula_cluster = False
        else:
            retained.append(block)
    blocks = retained
    anchored_blocks, anchored_sources = _recover_number_anchored_inline_display(
        [line for line in lines if line.source_index not in claimed_source_indices],
        table_bboxes,
        page_size,
    )
    blocks.extend(anchored_blocks)
    claimed_source_indices.update(anchored_sources)
    em = statistics.median(_line_effective_height(line, line.bbox) for line in lines) if lines else 10.0
    for block in blocks:
        core = block.get("_tight_output_bbox", block["bbox"])
        for line in lines:
            if line.source_index in claimed_source_indices or line.ink_bbox is None:
                continue
            text = line.text.strip()
            ink = line.ink_bbox
            if (
                len(text) > 20
                or _has_sentence_words(text)
                or not (len(text) <= 2 or _formula_line_has_math_operator(text))
                or not core[0] <= _bbox_center_x(ink) <= core[2]
                or _bbox_distance(core, ink) > 0.7 * em
            ):
                continue
            if any(
                other.source_index != line.source_index
                and _has_sentence_words(other.text)
                and _bbox_axis_overlap_ratio(ink, other.ink_bbox or other.bbox, axis="y") >= 0.4
                and _bbox_distance(ink, other.ink_bbox or other.bbox) <= em
                for other in lines
            ):
                continue
            block["bbox"] = _bbox_union(block["bbox"], line.bbox)
            block["_tight_output_bbox"] = _bbox_union(core, ink)
            block["content"] += "\n" + text
            claimed_source_indices.add(line.source_index)
    bands, _ = _build_spatial_numbered_bands(lines, table_bboxes, page_size, drawing_lines or [])
    recovered_band_ids = set()
    for band in bands:
        bbox = band.get("_tight_output_bbox", band["bbox"])
        overlaps = [
            block for block in blocks if _bbox_overlap_in_smaller(bbox, block.get("_tight_output_bbox", block["bbox"])) >= 0.5
        ]
        if any(
            all(old[i] <= bbox[i] + 1 if i < 2 else old[i] >= bbox[i] - 1 for i in range(4))
            for old in [block.get("_tight_output_bbox", block["bbox"]) for block in overlaps]
        ):
            continue
        # 只补齐缺失成员，既有完整公式的内容、裁图和顺序保持原结果。
        ids = set(band["_formula_members"]).union(*(block.get("_formula_members", []) for block in overlaps))
        members = [(line, line.ink_bbox or line.bbox) for line in lines if line.source_index in ids]
        if not members:
            continue
        merged = _formula_members_to_block(
            members, page_size, 0, anchor_source_index=band["_formula_members"][-1], include_member_ids=True
        )
        if merged:
            blocks = [block for block in blocks if not any(block is old for old in overlaps)]
            blocks.append(merged)
            claimed_source_indices.update(ids)
            recovered_band_ids.update(ids)
    for block in blocks:
        block.pop("_formula_members", None)
    remaining_lines = [
        line
        for line in lines
        if line.source_index not in claimed_source_indices
        and (not line.formula_candidate_only or line.paragraph_formula_context)
    ]
    return blocks, remaining_lines + [line for line in paragraph_lines if line.source_index not in recovered_band_ids]


def _build_spatial_numbered_bands(lines, table_bboxes, page_size, rules):
    """在成员被拆散认领前，用正文栏右缘的短编号与独立数学带恢复完整公式。"""
    prose = [line for line in lines if line.angle == 0 and len(line.text) >= 25 and _has_sentence_words(line.text)]
    layout = build_layout_evidence(prose, page_size, barriers=table_bboxes)
    blocks, claimed = [], set()
    for lane in layout.lanes:
        em = statistics.median(_line_effective_height(line, bounds) for line, bounds in lane.lines)

        def is_body(line):
            """短数学簇的拉丁变量不等同于正文词行，实际正文宽度及词组提供屏障。"""
            if not _display_math_has_prose(line.text):
                return False
            words = re.findall(r"\b[A-Za-z]{3,}\b", line.text)
            if len(words) >= 2 and line.font_coverage >= 0.75 and not _formula_line_has_math_operator(line.text):
                # 短说明句不足半栏宽仍是正文；不得借由邻近数学带吸收进公式裁图。
                return True
            return _has_sentence_words(line.text) and (
                line.bbox[2] - line.bbox[0] >= 0.6 * (lane.right - lane.left)
                or len(re.findall(r"\b[A-Za-z]{3,}\b", line.text)) >= 3
                or len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 3
                or re.search(r"\b(?:where|with|when|which|the|this|these|is|are|from|that|then)\b", line.text, re.I)
            )

        local = [
            line
            for line in lines
            if line.angle == 0
            and line.paragraph_group is None
            and line.semantic_type is None
            and lane.left - em <= line.bbox[0]
            and line.bbox[2] <= lane.right + em
        ]
        markers = [
            line
            for line in local
            if 2 <= len(line.text.strip()) <= 8
            and not re.search(r"\s", line.text.strip())
            and re.search(r"\d", line.text)
            and not line.text.strip()[0].isdigit()
            and not line.text.strip()[-1].isdigit()
            and (
                _standalone_formula_number_marker(line.text) is not None
                or re.fullmatch(r"[^A-Za-z0-9][A-Za-z]?[.]?\d+(?:[.\-]\d+)*[^A-Za-z0-9]", line.text.strip())
            )
            and abs(line.bbox[2] - lane.right) < 1.5 * em
            and line.bbox[2] - line.bbox[0] <= 4 * em
        ]
        for marker in markers:
            if marker.source_index in claimed:
                continue
            mb = marker.ink_bbox or marker.bbox
            nearby = [
                line
                for line in local
                if line is not marker
                and line.source_index not in claimed
                and line not in markers
                and not is_body(line)
                and (line.ink_bbox or line.bbox)[3] >= mb[1] - 4.5 * em
                and (line.ink_bbox or line.bbox)[1] <= mb[3] + 1.1 * em
            ]
            if not nearby:
                continue
            # 先形成二维数学主体，再关联隔着水平空白或位于右下角的编号。
            pending = set(range(len(nearby)))
            components = []
            while pending:
                seed = pending.pop()
                component, frontier = [seed], [seed]
                while frontier:
                    current = nearby[frontier.pop()]
                    bounds = current.ink_bbox or current.bbox
                    connected = []
                    for index in pending:
                        other = nearby[index].ink_bbox or nearby[index].bbox
                        xgap = max(0.0, bounds[0] - other[2], other[0] - bounds[2])
                        ygap = max(0.0, bounds[1] - other[3], other[1] - bounds[3])
                        if (
                            ygap <= 0.8 * em
                            and _bbox_axis_overlap_ratio(bounds, other, axis="x") > 0.1
                            or xgap <= 3 * em
                            and _bbox_axis_overlap_ratio(bounds, other, axis="y") > 0.15
                        ):
                            connected.append(index)
                    pending.difference_update(connected)
                    frontier.extend(connected)
                    component.extend(connected)
                components.append([nearby[index] for index in component])
            candidates = []
            for members in components:
                bounds = _bbox_union_many([line.ink_bbox or line.bbox for line in members])
                if (bounds[2] >= mb[0] + em and bounds[3] > mb[1]) or bounds[2] - bounds[0] < 3 * em:
                    continue
                gap = max(0.0, mb[1] - bounds[3], bounds[1] - mb[3])
                if gap > 2 * em or bounds[3] - bounds[1] > 6 * em:
                    continue
                if any(_bbox_intersects(bounds, table) for table in table_bboxes):
                    continue
                if any(
                    _bbox_axis_overlap_ratio(bounds, line.ink_bbox or line.bbox, axis="y") > 0.1
                    for line in local
                    if is_body(line)
                ):
                    continue
                math = any(_formula_line_has_math_operator(line.text) for line in members)
                numbered_math_typography = any(
                    line.font_signature is not None
                    and (line.font_signature[1] & (1 << 6) or line.font_coverage < 0.75)
                    and not _display_math_has_prose(line.text)
                    and len(re.findall(r"\b[A-Za-z]{1,2}\b", line.text)) >= 3
                    for line in members
                )
                fraction = any(
                    rule.orientation == "horizontal"
                    and bounds[0] <= rule.bbox[0] < rule.bbox[2] <= bounds[2]
                    and bounds[1] < rule.bbox[1] < bounds[3]
                    for rule in rules
                )
                if (
                    not math
                    and not fraction
                    and not numbered_math_typography
                    and not (
                        (len(members) >= 3 or any(line.compact_formula_cluster for line in members))
                        and bounds[3] - bounds[1] >= 1.3 * em
                    )
                ):
                    continue
                candidates.append((gap, members))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 0.25 * em:
                continue
            members = candidates[0][1] + [marker]
            block = _formula_members_to_block(
                [(line, line.ink_bbox or line.bbox) for line in members],
                page_size,
                0,
                anchor_source_index=marker.source_index,
                include_member_ids=True,
            )
            if block:
                block["_spatial_display_band"] = True
                blocks.append(block)
                claimed.update(line.source_index for line in members)
    return blocks, claimed


def _recover_detached_display_components(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    *,
    drawing_lines: list[_AxisLine] | None = None,
    _prose_flags: dict[int, bool] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """在编号和字体分类前聚合独立二维数学带，正文同行连通时保留为行内内容。"""
    blocks, claimed = [], set()
    for angle in {line.angle for line in lines}:
        geometry = [
            (line, _rotate_bbox_to_upright(line.ink_bbox or line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle
        ]
        if not geometry:
            continue
        em = statistics.median(_line_effective_height(line, bbox) for line, bbox in geometry)
        # 特征绑定本次输入状态，不跨文本修改或后续认领阶段缓存。
        prose_flags = (
            _prose_flags
            if _prose_flags is not None
            else {id(line): _detached_math_line_has_prose(line) for line, _ in geometry}
        )
        prose = [(line, bbox) for line, bbox in geometry if prose_flags[id(line)]]
        prose_layout = (
            build_layout_evidence([line for line, _ in prose], page_size, barriers=table_bboxes) if angle == 0 else None
        )
        prose_ids = {line.source_index for line, _ in prose}
        candidates = [
            (line, bbox)
            for line, bbox in geometry
            if line.source_index not in prose_ids
            and line.semantic_type is None
            and not any(_bbox_intersects(line.bbox, table) for table in table_bboxes)
        ]
        pending = set(range(len(candidates)))
        operator_flags = [_formula_line_has_math_operator(line.text) for line, _ in candidates]
        neighbors = _detached_formula_neighbors_native(candidates, operator_flags, em)
        while pending:
            seed = min(pending)
            pending.remove(seed)
            component, frontier = [seed], [seed]
            while frontier:
                current = frontier.pop()
                _, bbox = candidates[current]
                if neighbors is not None:
                    for index in neighbors[current]:
                        if index in pending:
                            component.append(index)
                            frontier.append(index)
                    pending.difference_update(component)
                    continue
                for index in sorted(pending):
                    other = candidates[index][1]
                    # 完整长公式各占一条基线；松行框重叠不能把相邻等式合成一个裁图。
                    if (
                        bbox[2] - bbox[0] > 3 * em
                        and other[2] - other[0] > 3 * em
                        and operator_flags[current]
                        and operator_flags[index]
                        and abs(_bbox_center_y(bbox) - _bbox_center_y(other)) > 0.75 * em
                    ):
                        continue
                    xgap = max(0.0, bbox[0] - other[2], other[0] - bbox[2])
                    ygap = max(0.0, bbox[1] - other[3], other[1] - bbox[3])
                    if (
                        ygap <= 0.85 * em
                        and _bbox_axis_overlap_ratio(bbox, other, axis="x") > 0.1
                        or xgap <= 2 * em
                        and _bbox_axis_overlap_ratio(bbox, other, axis="y") >= 0.2
                    ):
                        component.append(index)
                        frontier.append(index)
                pending.difference_update(component)
            members = [candidates[index] for index in component]
            if any(prose_flags[id(line)] for line, _bbox in members):
                continue
            bbox = _bbox_union_many([item[1] for item in members])
            calculus = any(any(char in line.text for char in "∂∫∑") for line, _ in members)
            if not any(operator_flags[index] for index in component):
                continue
            numbered = any(_standalone_formula_number_marker(line.text) for line, _ in members)
            strong_math = any(sum(unicodedata.category(char) == "Sm" for char in line.text) >= 2 for line, _ in members)
            typed_fraction = any(
                bounds[3] - bounds[1] >= 1.35 * _line_effective_height(line, line.bbox) and line.font_coverage < 0.75
                for line, bounds in members
            )
            if not numbered and bbox[3] - bbox[1] < 1.35 * em and not (calculus or strong_math):
                continue
            prose_rows = [
                pb
                for _, pb in prose
                if pb[2] - pb[0] >= 0.35 * page_size[0]
                and abs(_bbox_center_y(pb) - _bbox_center_y(bbox)) <= 8 * em
                and _bbox_axis_overlap_ratio(pb, bbox, axis="x") > 0.5
            ]
            if (
                not numbered
                and not calculus
                and len(members) == 1
                and bbox[3] - bbox[1] < 1.35 * em
                and prose_rows
                and bbox[0] <= statistics.median(pb[0] for pb in prose_rows) + 0.5 * em
            ):
                continue
            if len(component) == 1 and any(
                _bbox_axis_overlap_ratio(bbox, pb, axis="x") >= 0.2
                and _bbox_axis_overlap_ratio(bbox, pb, axis="y") >= 0.2
                and abs(_bbox_center_y(bbox) - _bbox_center_y(pb)) < 0.7 * em
                for _, pb in prose
            ):
                continue
            # 健康原生行不保留修复专用 ink 框；编号与二维结构共同提供等价的保守证据。
            fraction_rule = any(
                rule.orientation == "horizontal"
                and rule.bbox[2] - rule.bbox[0] >= 0.7 * em
                and bbox[0] <= rule.bbox[0] < rule.bbox[2] <= bbox[2] + 0.2 * em
                and bbox[1] + 0.3 * em < rule.bbox[1] < bbox[3] - 0.3 * em
                and any(box[3] <= rule.bbox[1] + 0.35 * em for _, box in members)
                and any(box[1] >= rule.bbox[1] - 0.35 * em for _, box in members)
                for rule in _transform_axis_lines(drawing_lines or [], page_size, angle)
            )
            stacked_fraction = (
                len(members) >= 3
                and any("=" in line.text for line, _ in members)
                and any(
                    above[3] - below[1] <= 0.35 * em
                    and 0.6 * em <= _bbox_center_y(below) - _bbox_center_y(above) <= 2 * em
                    and _bbox_axis_overlap_ratio(above, below, axis="x") >= 0.7
                    and max(above[2] - above[0], below[2] - below[0]) <= 3 * em
                    for _, above in members
                    for _, below in members
                    if above is not below
                )
            )
            if not any(line.ink_bbox is not None for line, _ in members) and not (
                numbered or strong_math or typed_fraction or fraction_rule or stacked_fraction
            ):
                continue
            if bbox[3] - bbox[1] > 8 * em or bbox[2] - bbox[0] < 3 * em:
                continue
            # 原生外框可能很松，使用 ink 判断正文是否与公式同处一行；栏间正文不构成宿主。
            corridor = prose_layout.corridor(bbox) if prose_layout is not None else None
            if any(
                _bbox_axis_overlap_ratio(bbox, pb, axis="y") >= 0.15
                and abs(_bbox_center_y(bbox) - _bbox_center_y(pb)) <= 0.7 * em
                and (
                    max(0.0, bbox[0] - pb[2], pb[0] - bbox[2]) <= 3 * em
                    or corridor is not None
                    and corridor[0] <= _bbox_center_x(pb) <= corridor[1]
                )
                for _, pb in prose
            ):
                continue
            local_height = page_size[0] if angle in {90, 270} else page_size[1]
            if _is_formula_component_in_page_margin(bbox, local_height):
                continue
            # 右缘编号可以与公式主体隔开很远；同栏同行且没有正文屏障时唯一绑定。
            marker_peers = [
                (line, box)
                for line, box in candidates
                if line.source_index not in {member.source_index for member, _ in members}
                and _standalone_formula_number_marker(line.text)
                and box[0] > bbox[2]
                and abs(_bbox_center_y(box) - _bbox_center_y(bbox)) <= 0.6 * em
                and prose_rows
                and abs(box[2] - statistics.median(pb[2] for pb in prose_rows)) <= 2 * em
                and not any(
                    pb[0] >= bbox[2] and pb[2] <= box[0] and _bbox_axis_overlap_ratio(pb, box, axis="y") > 0.1
                    for _, pb in prose
                )
            ]
            if len(marker_peers) == 1:
                members.extend(marker_peers)
                claimed.update(line.source_index for line, _ in marker_peers)
                pending.difference_update(index for index, (line, _) in enumerate(candidates) if line.source_index in claimed)
            block = _formula_members_to_block(
                members,
                page_size,
                angle,
                include_member_ids=True,
                anchor_source_index=next(
                    (line.source_index for line, _ in members if _standalone_formula_number_marker(line.text)),
                    members[0][0].source_index,
                ),
            )
            if block is not None:
                blocks.append(block)
                claimed.update(line.source_index for line, _ in members)
    return blocks, claimed


def _detached_formula_neighbors_native(candidates, operators, em):
    """批量计算独立公式带邻接，特殊数值和过大邻接表完整使用原有扫描。"""
    native = get_native()
    if native is None or type(em) not in (int, float) or not math.isfinite(em):
        return None
    boxes = [bbox for _, bbox in candidates]
    if any(type(v) not in (int, float) or abs(v) > 2**50 or not math.isfinite(v) for box in boxes for v in box):
        return None
    return native.detached_formula_neighbors(boxes, operators, em)


_MATH_WORD_REFERENCES = (_LineItem.__getattribute__, _LineItem.chars, _LineItem.native_math_words, re.finditer, re._compile)


def _native_math_word_fragments(line: _LineItem) -> frozenset[str]:
    """普通原始字形整行批量读取，定制对象或正则实现完整回退到原参考路径。"""
    from ...._compute_backend import get_native

    if (
        _LineItem.__getattribute__,
        getattr(_LineItem, "chars", None),
        getattr(_LineItem, "native_math_words", None),
        re.finditer,
        re._compile,
    ) == _MATH_WORD_REFERENCES:
        # 字符已在当前行阶段释放后，只剩冻结的不可变词集合；无需再次构造集合或运行空正则。
        if (
            type(line) is _LineItem
            and type(line.chars) is list
            and not line.chars
            and type(line.native_math_words) is frozenset
        ):
            return line.native_math_words
        native = get_native()
        if native is not None:
            words = native.math_words_owned(line, _LineItem, re.finditer)
            if words is not None:
                return words
    return _native_math_word_fragments_python(line)


def _native_math_word_fragments_python(line: _LineItem) -> frozenset[str]:
    """短连写变量只有全部斜体或包含独立小号角标时才消除自然语言屏障。"""
    chars = [char for char in line.chars if str(char.get("char", "")).isprintable()]
    text = "".join(str(char.get("char", "")) for char in chars)
    words = set(line.native_math_words)
    for match in re.finditer(r"\b[a-z]{3,4}\b", text):
        members = chars[match.start() : match.end()]
        fonts = [char.get("font") or {} for char in members]
        sizes = [float(font.get("size", 0) or 0) for font in fonts]
        if fonts and (
            all(int(font.get("flags", 0) or 0) & 64 for font in fonts) or min(sizes) > 0 and min(sizes) <= 0.85 * max(sizes)
        ):
            words.add(match.group())
    return frozenset(words)


def _detached_math_line_has_prose(line: _LineItem) -> bool:
    """保留普通词和正文屏障，仅移除有原生字形证据的短变量乘积。"""
    words = _native_math_word_fragments(line)
    text = re.sub(r"\b[a-z]{3,4}\b", lambda match: "x" if match.group() in words else match.group(), line.text)
    return _display_math_has_prose(text)


def _has_sentence_words(text: str) -> bool:
    """用独立自然语言词排除正文，数学函数名和变量内部字母不计作句子。"""
    return has_prose(text)


def _build_mixed_body_display_formulas(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """在混合字体正文的独立留白带内恢复无编号公式，避免把碎片窄栏当作正文栏。"""

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for angle in sorted({line.angle for line in lines}):
        geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None and not line.paragraph_formula_context
        ]
        width = page_size[1] if angle in {90, 270} else page_size[0]
        body = [
            (line, bbox) for line, bbox in geometry if bbox[2] - bbox[0] >= 0.5 * width and _formula_prefix_has_prose(line.text)
        ]
        # 该补充路径只处理既有 dominant-font 路径缺少稳定正文覆盖的混排页面。
        if sum(line.font_coverage < 0.75 for line, _bbox in body) < 2:
            continue
        height = statistics.median(_line_effective_height(line, bbox) for line, bbox in body)
        body_left = statistics.median(bbox[0] for _line, bbox in body)
        body_right = statistics.median(bbox[2] for _line, bbox in body)
        body_width = body_right - body_left
        barriers = sorted(
            [
                (line, bbox)
                for line, bbox in geometry
                if _formula_prefix_has_prose(line.text)
                and (bbox[2] - bbox[0] >= 0.5 * body_width or len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 2)
            ],
            key=lambda item: item[1][1],
        )
        for previous, following in zip(barriers, barriers[1:]):
            top, bottom = previous[1][3], following[1][1]
            if not 0.6 * height <= bottom - top <= 8.0 * height:
                continue
            members = [
                (line, bbox)
                for line, bbox in geometry
                if line.source_index not in {previous[0].source_index, following[0].source_index}
                and line.source_index not in claimed
                and bbox[1] >= top - 0.2 * height
                and bbox[3] <= bottom + 0.2 * height
                and bbox[0] >= body_left
                and bbox[2] <= body_right
            ]
            if not members:
                continue
            bbox = _bbox_union_many([box for _line, box in members])
            text = " ".join(line.text for line, _box in members)
            if (
                not any(char in "=∑∫√≤≥<>" for char in text)
                or len(re.findall(r"[\u3400-\u9fff]", text)) >= 2
                or len(re.findall(r"\b[A-Za-z]{4,}\b", text)) >= 2
                or not 0.15 * body_width <= bbox[2] - bbox[0] <= 0.8 * body_width
                or abs(_bbox_center_x(bbox) - 0.5 * (body_left + body_right)) > 0.1 * body_width
                or any(
                    _bbox_overlap_in_smaller(line.bbox, table_bbox) > 0 for line, _box in members for table_bbox in table_bboxes
                )
            ):
                continue
            block = _formula_members_to_block(
                members, page_size, angle, include_member_ids=True, anchor_source_index=members[0][0].source_index
            )
            if block is not None:
                blocks.append(block)
                claimed.update(line.source_index for line, _bbox in members)
    return blocks, claimed


_UNMAPPED_INK_COERCE = _coerce_bbox
_UNMAPPED_INK_FLAGS = _unmapped_glyph_flags
_UNMAPPED_INK_ISFINITE = math.isfinite


def _unmapped_formula_ink_bboxes(chars: list[Char]) -> list[BBox]:
    """保留实际绘制、但映射为空白或控制码的高字形几何，不改变公开文本。"""

    from ._native_geometry import glyph_flags

    native = get_native()
    if (
        native is not None
        and hasattr(native, "unmapped_formula_ink")
        and _coerce_bbox is _UNMAPPED_INK_COERCE
        and glyph_flags is _UNMAPPED_INK_FLAGS
        and math.isfinite is _UNMAPPED_INK_ISFINITE
    ):
        values = native.unmapped_formula_ink(chars, glyph_flags)
        if values is not None:
            return values
    return _unmapped_formula_ink_bboxes_python(chars)


def _unmapped_formula_ink_bboxes_python(chars: list[Char]) -> list[BBox]:
    """完整参考筛选保留陌生转换、读取顺序与异常，供批量不支持时直接回退。"""

    output: list[BBox] = []
    for char in chars:
        value = str(char.get("char", ""))
        if value.isprintable() and not value.isspace():
            continue
        bbox = _coerce_bbox(char.get("tight_bbox"))
        font = char.get("font") or {}
        size = float(font.get("size", 0) or 0)
        if (
            bbox is not None
            and size > 0
            and char.get("text_object_id") is not None
            and char.get("text_render_mode") not in {3, 7}
            and bbox[2] - bbox[0] > 0.05 * size
            and bbox[3] - bbox[1] > 1.5 * size
        ):
            output.append(bbox)
    return output


def _attach_unmapped_formula_ink(blocks: list[dict[str, Any]], bboxes: list[BBox]) -> None:
    """将紧贴公式且纵向相容的未映射字形唯一认领到公式框。"""

    for bbox in bboxes:
        matches = [
            block
            for block in blocks
            if block.get("type") == "equation"
            and int(block.get("angle", 0)) in {0, 180}
            and _bbox_axis_overlap_ratio(bbox, block["bbox"], axis="y") >= 0.8
            and _bbox_distance(bbox, block["bbox"]) <= 0.25 * (bbox[3] - bbox[1])
            and bbox[1] >= block["bbox"][1] - 2.0
            and bbox[3] <= block["bbox"][3] + 2.0
        ]
        if len(matches) == 1:
            block = matches[0]
            block["bbox"] = _bbox_union(block["bbox"], bbox)
            if block.get("_tight_output_bbox") is not None:
                block["_tight_output_bbox"] = _bbox_union(block["_tight_output_bbox"], bbox)


@_formula_text_feature
def _formula_line_has_math_operator(text: str) -> bool:
    """检查文本行是否具有独立公式常见的数学运算符。"""

    return any(
        character in _FORMULA_OPERATOR_CHARS
        or unicodedata.category(character) == "Sm"
        or "VULGAR FRACTION" in unicodedata.name(character, "")
        for character in text
    )


@_formula_text_feature
def _display_math_has_prose(text: str) -> bool:
    """变量、函数调用及常见数学连接词不构成说明正文，其余完整词和汉字建立屏障。"""
    if len(re.findall(r"[\u3400-\u9fff]", text)) >= 2:
        return True
    # 带数字后缀的完整自然语言词仍是术语，不能因连字符被数学词法器整体消去。
    if any(len(re.findall(r"[a-z]", word)) >= 3 for word in re.findall(r"\b([A-Za-z]{4,})-\d+\b", text)):
        return True
    mathematical_words = {"lim", "sin", "cos", "tan", "log", "exp", "min", "max", "with", "and", "for"}
    words = re.findall(r"\b([A-Za-z]{3,})\b(?!\s*[(\[])", prose_residue(text))
    return any(word.lower() not in mathematical_words and not re.fullmatch(r"[a-z]{0,2}[A-Z]{2,}", word) for word in words)


@_formula_text_feature
def _formula_prefix_has_prose(prefix: str) -> bool:
    """用通用文字数量识别公式前的正文片段，不依赖特定引导词或标点。"""

    prose_prefix = re.sub(
        r"[({\[（［【｛][^)}\]）］】｝]*[)}\]）］】｝]",
        " ",
        prose_residue(prefix),
    )
    if len(re.findall(r"[\u3400-\u9fff]", prose_prefix)) >= 2:
        return True
    latin_word_count = sum(re.search(r"[A-Za-z]{2,}", token) is not None for token in prose_prefix.split())
    return latin_word_count >= 2


def _formula_component_has_left_prose(
    members: list[tuple[_LineItem, BBox]],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别贴栏左缘且在首个运算符前带同行正文的伪行间公式。"""

    for line, bbox in members:
        normalized = unicodedata.normalize("NFKC", line.text).strip()
        operator_positions = [index for index, character in enumerate(normalized) if character in _FORMULA_OPERATOR_CHARS]
        if not operator_positions:
            continue
        prefix = normalized[: min(operator_positions)]
        if not _formula_prefix_has_prose(prefix):
            continue
        if abs(bbox[0] - lane.left) <= 0.75 * median_height:
            return True
    return _fragmented_left_prose(members, lane, median_height)


def _fragmented_left_prose(members: list[tuple[_LineItem, BBox]], lane: _TextLane, median_height: float) -> bool:
    """运算符前的正文被拆成多个 run 时，沿同行邻接找到栏左缘的正文证据。"""
    for line, bbox in members:
        normalized = unicodedata.normalize("NFKC", line.text).strip()
        positions = [index for index, character in enumerate(normalized) if character in _FORMULA_OPERATOR_CHARS]
        if not positions or not _formula_prefix_has_prose(normalized[: min(positions)]):
            continue
        if any(
            other is not line
            and abs(other_bbox[0] - lane.left) <= 0.75 * median_height
            and 0 <= bbox[0] - other_bbox[2] <= 0.75 * median_height
            and _bbox_axis_overlap_ratio(bbox, other_bbox, axis="y") >= 0.7
            for other, other_bbox in members
        ):
            return True
    return False


def restore_leading_inline_fractions(lines: list[_LineItem], rules: list, page_size: tuple[float, float]) -> list[_LineItem]:
    """正文行首的短分子、分母与原生横线共同确认分式，恢复关系并保留字符成员。"""
    remaining = list(lines)
    for numerator in list(lines):
        if (
            numerator not in remaining
            or numerator.angle != 0
            or numerator.semantic_type is not None
            or not re.fullmatch(r"[\dα-ωΑ-Ω+−-]{1,8}", numerator.text.strip())
            or not re.search(r"[α-ωΑ-Ω]", numerator.text)
        ):
            continue
        em = _line_effective_height(numerator, numerator.bbox)
        for denominator in list(remaining):
            if (
                denominator is numerator
                or denominator.angle != 0
                or denominator.semantic_type is not None
                or len(re.findall(r"[\u4e00-\u9fff]", denominator.text)) < 4
                or not re.match(r"^\s*[α-ωΑ-Ω]", denominator.text)
                or not denominator.chars
            ):
                continue
            first = next((char for char in denominator.chars if str(char.get("char", "")).strip()), None)
            ink = _coerce_bbox(first.get("tight_bbox") or first.get("bbox")) if first else None
            if ink is None:
                continue
            separators = [
                rule.bbox
                for rule in rules
                if rule.orientation == "horizontal"
                and 0.5 * (numerator.bbox[2] - numerator.bbox[0]) <= rule.bbox[2] - rule.bbox[0] <= 1.5 * em
                and rule.bbox[0] <= _bbox_center_x(ink) <= rule.bbox[2]
                and numerator.bbox[1] < rule.bbox[1] < ink[3]
                and _bbox_axis_overlap_ratio(rule.bbox, numerator.bbox, axis="x") >= 0.7
                and 0 < ink[3] - numerator.bbox[1] <= 2.5 * em
            ]
            if not separators:
                continue
            preceding = [
                line
                for line in remaining
                if line is not numerator
                and line is not denominator
                and len(line.text) >= 20
                and line.bbox[3] <= numerator.bbox[1] + 0.2 * em
                and numerator.bbox[1] - line.bbox[3] <= em
                and abs(line.bbox[0] - denominator.bbox[0]) <= em
            ]
            if not preceding:
                continue
            merged = _merge_paragraph_formula_members(
                [(numerator, numerator.bbox), (denominator, denominator.bbox)], page_size, em
            )
            merged.text = numerator.text.strip() + "/" + denominator.text.lstrip()
            merged.semantic_type = "text"
            merged.title_suppressed = True
            remaining = [line for line in remaining if line is not numerator and line is not denominator] + [merged]
            break
    return sorted(remaining, key=lambda line: (line.bbox[1], line.bbox[0], line.source_index))


def _merge_paragraph_formula_members(
    members: list[tuple[_LineItem, BBox]], page_size: tuple[float, float], median_height: float
) -> _LineItem:
    """把已确认的行内分式按重叠视觉行恢复顺序，避免字形高度差把正文前缀排到分子之后。"""
    rows: list[list[tuple[_LineItem, BBox]]] = []
    for item in sorted(members, key=lambda item: (item[1][1], item[1][0])):
        match = next(
            (row for row in rows if any(_bbox_axis_overlap_ratio(item[1], other[1], axis="y") >= 0.55 for other in row)),
            None,
        )
        if match is None:
            rows.append([item])
        else:
            match.append(item)
    merged = _merge_overlapping_inline_cluster(members, page_size, median_height, compact_formula_cluster=False)
    merged.text = " ".join(_join_formula_visual_row(row, page_size) for row in rows)
    merged.paragraph_formula_context = True
    merged.formula_candidate_only = False
    return merged


def _formula_component_has_isolated_numbered_fraction(
    members: list[tuple[_LineItem, BBox]],
    lane: _TextLane,
    median_height: float,
    horizontal_rules: list[BBox],
) -> bool:
    """用右侧编号、内部分数线和上下留白确认独立多层公式。"""

    if len(members) < 3 or not horizontal_rules:
        return False
    markers = [(line, bbox) for line, bbox in members if _standalone_formula_number_marker(line.text) is not None]
    if len(markers) != 1:
        return False
    marker_line, marker_bbox = markers[0]
    lane_width = max(0.1, lane.right - lane.left)
    if marker_bbox[2] < lane.right - max(3.0, 0.08 * lane_width) or marker_bbox[2] - marker_bbox[0] > 0.12 * lane_width:
        return False
    body_members = [(line, bbox) for line, bbox in members if line is not marker_line]
    if len(body_members) < 2:
        return False
    body_bbox = _bbox_union_many(
        [bbox for _line, bbox in body_members],
    )
    member_sources = {line.source_index for line, _bbox in members}
    rows_above = [bbox for line, bbox in lane.lines if line.source_index not in member_sources and bbox[3] <= body_bbox[1]]
    rows_below = [bbox for line, bbox in lane.lines if line.source_index not in member_sources and bbox[1] >= body_bbox[3]]
    if not rows_above or not rows_below:
        return False
    gap_above = body_bbox[1] - max(bbox[3] for bbox in rows_above)
    gap_below = min(bbox[1] for bbox in rows_below) - body_bbox[3]
    if min(gap_above, gap_below) < 0.75 * median_height:
        return False

    body_centers = [_bbox_center_y(bbox) for _line, bbox in body_members]
    for rule_bbox in horizontal_rules:
        rule_width = rule_bbox[2] - rule_bbox[0]
        if not (3.0 * median_height <= rule_width <= 0.75 * lane_width):
            continue
        horizontal_overlap = max(
            0.0,
            min(rule_bbox[2], body_bbox[2]) - max(rule_bbox[0], body_bbox[0]),
        )
        if horizontal_overlap < 0.6 * rule_width:
            continue
        rule_center = _bbox_center_y(rule_bbox)
        if any(center <= rule_center - 0.1 * median_height for center in body_centers) and any(
            center >= rule_center + 0.1 * median_height for center in body_centers
        ):
            return True
    return False


def _is_wide_tagged_formula_member(
    anchor_line: _LineItem,
    member_line: _LineItem,
    member_bbox: BBox,
    lane_width: float,
) -> bool:
    """判断独立编号左侧是否为接近满栏的单行公式主体。"""

    member_width = member_bbox[2] - member_bbox[0]
    marker = _standalone_formula_number_marker(anchor_line.text)
    return (
        anchor_line.style_scale_repaired
        and marker is not None
        and _FORMULA_NUMBER_MARKER_RE.fullmatch(marker)
        and 0.75 * lane_width < member_width <= 0.95 * lane_width
        and _formula_line_has_math_operator(member_line.text)
    )


def order_equation_markers(blocks):
    """只将同行独立公式编号后置于其主体，正文中的公式引用不参与重排。"""
    result = list(blocks)
    for equation in blocks:
        if equation.get("type") != "equation":
            continue
        box = equation["bbox"]
        em = max(box[3] - box[1], 1)
        for marker in blocks:
            parts = _split_trailing_formula_number(marker.get("content", ""))
            if (
                marker.get("type") != "text"
                or parts is None
                or re.search(r"[^\s.…·•—-]", parts[0])
                or marker["bbox"][0] <= box[2]
                or marker["bbox"][0] - box[2] > 8 * em
                or abs(_bbox_center_y(marker["bbox"]) - _bbox_center_y(box)) > 0.75 * em
            ):
                continue
            if result.index(marker) < result.index(equation):
                result.remove(marker)
                result.insert(result.index(equation) + 1, marker)
    return result


def _recover_number_anchored_inline_display(lines, table_bboxes, page_size):
    """右侧编号、纯数学主体和独立居中带共同确认单行公式，保留编号独立归属。"""
    blocks, claimed = [], set()
    for marker in lines:
        parts = _split_trailing_formula_number(marker.text)
        if parts is None or re.search(r"[^\s.…·•—-]", parts[0]) or marker.angle != 0:
            continue
        em = max(marker.effective_height, 1)
        members = [
            line
            for line in lines
            if line is not marker
            and line.angle == marker.angle
            and line.source_index not in claimed
            and line.semantic_type is None
            and abs(_bbox_center_y(line.bbox) - _bbox_center_y(marker.bbox)) <= 0.65 * em
            and line.bbox[2] < marker.bbox[0]
            and marker.bbox[0] - line.bbox[2] <= 16 * em
        ]
        if not members:
            continue
        text = " ".join(line.text for line in sorted(members, key=lambda line: line.bbox[0]))
        box = _bbox_union_many([line.bbox for line in members])
        if (
            "=" not in text
            or not re.search(r"[+−*/×÷-]", text)
            or re.search(r"[\u3400-\u9fff]|[A-Za-z]{4,}", text)
            or not 0.08 * page_size[0] <= box[2] - box[0] <= 0.6 * page_size[0]
            or abs(_bbox_center_x(box) - 0.5 * page_size[0]) > 0.2 * page_size[0]
            or box[3] - box[1] > 2 * em
            or any(_bbox_intersects(box, table) for table in table_bboxes)
        ):
            continue
        block = _formula_members_to_block(
            [(line, line.bbox) for line in members],
            page_size,
            0,
            anchor_source_index=members[0].source_index,
            include_member_ids=True,
        )
        if block is not None:
            blocks.append(block)
            claimed.update(line.source_index for line in members)
    return blocks, claimed


def _is_single_line_numbered_formula(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别公式主体与右侧编号已落在同一原生文本行的情形。"""

    line, bbox = candidate
    if not line.style_scale_repaired:
        return False
    parts = _split_trailing_formula_number(line.text)
    if parts is None:
        return False
    prefix, marker = parts
    if not prefix or not _FORMULA_NUMBER_MARKER_RE.fullmatch(marker) or not _formula_line_has_math_operator(prefix):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    width_ratio = (bbox[2] - bbox[0]) / lane_width
    if not 0.12 <= width_ratio <= 0.98:
        return False
    if abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) > 0.2 * lane_width:
        return False
    if bbox[3] - bbox[1] > 4.0 * median_height:
        return False
    return not _is_hanging_indent_tail_line(
        candidate,
        lane,
        median_height,
    )


def _expand_single_line_numbered_formula_members(
    core: tuple[_LineItem, BBox],
    lane: _TextLane,
    claimed_source_indices: set[int],
    table_bboxes: list[BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """为已带编号的公式核心吸收同栏连通的等号前缀和窄分式碎片。"""

    core_line, core_bbox = core
    lane_width = max(0.1, lane.right - lane.left)
    candidates = []
    for candidate_line, candidate_bbox in lane.lines:
        if (
            candidate_line.source_index == core_line.source_index
            or candidate_line.source_index in claimed_source_indices
            or candidate_bbox[2] - candidate_bbox[0] > 0.35 * lane_width
            or _is_formula_body_barrier(
                (candidate_line, candidate_bbox),
                lane,
                dominant_body_font,
                median_height,
            )
            or _is_formula_title_barrier(
                (candidate_line, candidate_bbox),
                lane,
                dominant_body_font,
                median_height,
            )
        ):
            continue
        narrow_fragment = (
            candidate_bbox[2] - candidate_bbox[0] <= 1.5 * median_height
            and core_bbox[0] - median_height <= _bbox_center_x(candidate_bbox) <= core_bbox[2] + median_height
        )
        if not (
            _formula_line_has_math_operator(candidate_line.text)
            or candidate_line.compact_formula_cluster
            or candidate_line.formula_candidate_only
            or narrow_fragment
        ):
            continue
        if not _formula_lines_are_connected(
            core_line,
            core_bbox,
            candidate_line,
            candidate_bbox,
            table_bboxes,
        ):
            continue
        candidates.append(
            (candidate_line, candidate_bbox),
        )

    members = [core, *candidates]
    member_sources = {line.source_index for line, _bbox in members}
    changed = True
    while changed:
        changed = False
        for candidate in lane.lines:
            candidate_line, candidate_bbox = candidate
            if (
                candidate_line.source_index in member_sources
                or candidate_line.source_index in claimed_source_indices
                or candidate_bbox[2] - candidate_bbox[0] > 0.35 * lane_width
                or _is_formula_body_barrier(
                    candidate,
                    lane,
                    dominant_body_font,
                    median_height,
                )
                or _is_formula_title_barrier(
                    candidate,
                    lane,
                    dominant_body_font,
                    median_height,
                )
            ):
                continue
            if any(
                _formula_lines_are_connected(
                    member_line,
                    member_bbox,
                    candidate_line,
                    candidate_bbox,
                    table_bboxes,
                )
                for member_line, member_bbox in members
            ) and (
                _formula_line_has_math_operator(candidate_line.text)
                or candidate_line.compact_formula_cluster
                or candidate_line.formula_candidate_only
                or candidate_bbox[2] - candidate_bbox[0] <= 1.5 * median_height
            ):
                members.append(candidate)
                member_sources.add(candidate_line.source_index)
                changed = True
    return members


def _build_split_visual_row_formula_blocks(
    lines: list[_LineItem],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
) -> tuple[list[dict[str, Any]], set[int]]:
    """在栏带推断前恢复同一视觉行中带右侧编号的多字体公式。"""

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in lines:
        if line.visual_row_id is None or not line.split_from_row:
            continue
        row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    for (angle, _row_id), members in row_groups.items():
        if len(members) < 3:
            continue
        markers = [member for member in members if _standalone_formula_number_marker(member.text) is not None]
        if len(markers) != 1:
            continue
        marker = markers[0]
        if any(_bbox_intersects(member.bbox, table_bbox) for member in members for table_bbox in table_bboxes):
            continue
        local_members = [
            (
                member,
                _rotate_bbox_to_upright(
                    member.bbox,
                    page_size,
                    angle,
                ),
            )
            for member in members
        ]
        marker_bbox = next(bbox for member, bbox in local_members if member is marker)
        body_members = [(member, bbox) for member, bbox in local_members if member is not marker]
        if not body_members or marker_bbox[0] <= max(_bbox_center_x(bbox) for _member, bbox in body_members):
            continue
        median_height = statistics.median(_line_effective_height(member, bbox) for member, bbox in local_members)
        row_center = statistics.median(_bbox_center_y(bbox) for _member, bbox in local_members)
        if any(abs(_bbox_center_y(bbox) - row_center) > 0.75 * median_height for _member, bbox in local_members):
            continue
        body_fonts = {member.font_signature for member, _bbox in body_members if member.font_signature is not None}
        has_math_typography = len(body_fonts) >= 2 or any(
            member.compact_formula_cluster or member.font_coverage < 0.8 for member, _bbox in body_members
        )
        if not has_math_typography:
            continue
        body_bbox = _bbox_union_many([bbox for _member, bbox in body_members])
        body_width = max(0.1, body_bbox[2] - body_bbox[0])
        # 同行成员可能只是分式尾部；窄尾部不能压低外部公式片段的宽度容差，
        # 否则会提前认领分母、右括号和编号，使左侧公式主体落回普通文本。
        nearby_fragment_width_limit = max(
            0.65 * body_width,
            3.0 * median_height,
        )
        member_ids = {id(member) for member in members}
        has_nearby_formula_fragment = False
        for other in lines:
            if id(other) in member_ids or other.angle != angle:
                continue
            other_bbox = _rotate_bbox_to_upright(
                other.bbox,
                page_size,
                angle,
            )
            vertical_gap = max(
                0.0,
                max(other_bbox[1], body_bbox[1]) - min(other_bbox[3], body_bbox[3]),
            )
            if (
                vertical_gap <= 0.75 * median_height
                and other_bbox[2] - other_bbox[0] <= nearby_fragment_width_limit
                and max(
                    0.0,
                    max(other_bbox[0], body_bbox[0]) - min(other_bbox[2], body_bbox[2]),
                )
                <= median_height
            ):
                has_nearby_formula_fragment = True
                break
        if has_nearby_formula_fragment:
            continue
        block = _formula_members_to_block(
            local_members,
            page_size,
            angle,
            include_member_ids=True,
            anchor_source_index=marker.source_index,
        )
        if block is None:
            continue
        local_bbox = _bbox_union_many([bbox for _member, bbox in local_members])
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        if _is_formula_component_in_page_margin(
            local_bbox,
            local_page_height,
        ):
            continue
        blocks.append(block)
        claimed.update(member.source_index for member in members)
    return blocks, claimed


def _is_isolated_compact_formula_cluster(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """用上下正文邻行确认紧凑二维文本簇是独立行间公式。"""

    line, bbox = candidate
    if not line.compact_formula_cluster:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    if bbox[2] - bbox[0] > 0.6 * lane_width:
        return False
    if bbox[3] - bbox[1] > 3.0 * median_height:
        return False
    center_delta_ratio = abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) / lane_width
    left_indent_ratio = (bbox[0] - lane.left) / lane_width
    right_blank_ratio = (lane.right - bbox[2]) / lane_width
    # 部分期刊把独立公式按固定左缩进排版；同时要求右侧大留白，排除贴栏正文。
    deliberately_left_indented = 0.03 <= left_indent_ratio <= 0.25 and right_blank_ratio >= 0.35
    if center_delta_ratio > 0.2 and not deliberately_left_indented:
        return False

    candidate_center = _bbox_center_y(bbox)
    body_rows = [
        item
        for item in lane.lines
        if item[0].source_index != line.source_index
        and item[1][2] - item[1][0] >= 0.45 * lane_width
        and 0.8 * median_height <= _line_effective_height(*item) <= 1.25 * median_height
    ]
    rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < candidate_center]
    rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > candidate_center]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    return (
        candidate_center - _bbox_center_y(previous[1]) <= 8.0 * median_height
        and _bbox_center_y(following[1]) - candidate_center <= 8.0 * median_height
    )


def _compact_cluster_has_nearby_number_anchor(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """检测紧凑公式右侧的独立编号，保留给既有空间锚点统一扩张。"""

    line, bbox = candidate
    return any(
        other_line.source_index != line.source_index
        and _standalone_formula_number_marker(other_line.text) is not None
        and other_bbox[0] > _bbox_center_x(bbox)
        and abs(_bbox_center_y(other_bbox) - _bbox_center_y(bbox)) <= 2.5 * median_height
        for other_line, other_bbox in lane.lines
    )


def _is_isolated_unnumbered_formula_line(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None,
) -> bool:
    """用低正文覆盖的数学排版和上下正文邻接识别无编号行间公式。"""

    line, bbox = candidate
    if (
        line.compact_formula_cluster
        or not _formula_line_has_math_operator(line.text)
        or dominant_body_font is None
        or line.font_signature is None
        or line.font_signature == dominant_body_font
        or line.font_coverage >= 0.75
    ):
        return False
    lane_width = max(0.1, lane.right - lane.left)
    line_width = bbox[2] - bbox[0]
    if not 0.15 * lane_width <= line_width <= 0.8 * lane_width:
        return False
    if abs(_bbox_center_x(bbox) - 0.5 * (lane.left + lane.right)) > 0.08 * lane_width:
        return False
    if bbox[3] - bbox[1] > 1.8 * median_height:
        return False
    if _is_hanging_indent_tail_line(candidate, lane, median_height):
        return False
    candidate_center = _bbox_center_y(bbox)
    if any(
        other_line.source_index != line.source_index
        and _standalone_formula_number_marker(other_line.text) is not None
        and abs(_bbox_center_y(other_bbox) - candidate_center) <= 4.0 * median_height
        for other_line, other_bbox in lane.lines
    ) or _has_nearby_punctuated_formula_number_anchor(
        candidate,
        lane,
        median_height,
    ):
        return False
    body_rows = [
        item
        for item in lane.lines
        if item[0].source_index != line.source_index
        and item[0].font_signature == dominant_body_font
        and item[0].font_coverage >= 0.75
        and item[1][2] - item[1][0] >= 0.45 * lane_width
        and 0.8 * median_height <= _line_effective_height(*item) <= 1.25 * median_height
    ]
    rows_above = [item for item in body_rows if _bbox_center_y(item[1]) < candidate_center]
    rows_below = [item for item in body_rows if _bbox_center_y(item[1]) > candidate_center]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    return (
        candidate_center - _bbox_center_y(previous[1]) <= 4.0 * median_height
        and _bbox_center_y(following[1]) - candidate_center <= 4.0 * median_height
    )


def _is_hanging_indent_tail_line(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """用相邻行缩进、字体和节奏识别参考条目的悬挂缩进尾行。"""

    line, bbox = candidate
    if line.font_signature is None:
        return False
    candidate_center = _bbox_center_y(bbox)
    rows_above = [
        item for item in lane.lines if item[0].source_index != line.source_index and _bbox_center_y(item[1]) < candidate_center
    ]
    rows_below = [
        item for item in lane.lines if item[0].source_index != line.source_index and _bbox_center_y(item[1]) > candidate_center
    ]
    if not rows_above or not rows_below:
        return False
    previous = max(rows_above, key=lambda item: _bbox_center_y(item[1]))
    following = min(rows_below, key=lambda item: _bbox_center_y(item[1]))
    previous_line, previous_bbox = previous
    _following_line, following_bbox = following
    previous_pitch = candidate_center - _bbox_center_y(previous_bbox)
    following_pitch = _bbox_center_y(following_bbox) - candidate_center
    lane_width = max(0.1, lane.right - lane.left)
    candidate_width = bbox[2] - bbox[0]
    previous_width = previous_bbox[2] - previous_bbox[0]
    return (
        previous_line.font_signature == line.font_signature
        and abs(previous_bbox[0] - bbox[0]) <= 0.5 * median_height
        and candidate_width <= 0.9 * previous_width
        and 0.65 * median_height <= previous_pitch <= 1.6 * median_height
        and 0.65 * median_height <= following_pitch <= 1.6 * median_height
        and following_bbox[0] <= bbox[0] - 1.5 * median_height
        and following_bbox[2] - following_bbox[0] >= 0.75 * lane_width
    )


def _has_nearby_punctuated_formula_number_anchor(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    median_height: float,
) -> bool:
    """识别同一公式带右侧仅带标点前缀的编号，避免分式上下行被提前认领。"""

    line, bbox = candidate
    for other_line, other_bbox in lane.lines:
        if other_line.source_index == line.source_index:
            continue
        parts = _split_trailing_formula_number(other_line.text)
        if parts is None:
            continue
        prefix, _marker = parts
        compact_prefix = prefix.strip()
        if not compact_prefix or len(compact_prefix) > 3 or any(character.isalnum() for character in compact_prefix):
            continue
        vertical_gap = max(
            0.0,
            max(other_bbox[1], bbox[1]) - min(other_bbox[3], bbox[3]),
        )
        horizontal_gap = max(0.0, other_bbox[0] - bbox[2])
        if (
            other_bbox[0] >= bbox[2] - 0.5 * median_height
            and vertical_gap <= 0.75 * median_height
            and horizontal_gap <= 4.0 * median_height
        ):
            return True
    return False


def _find_repeated_formula_number_anchors(
    lane: _TextLane,
    median_height: float,
    body_interval: tuple[float, float] | None,
) -> list[_FormulaAnchor]:
    """用栏右缘重复编号恢复正文区间之外的行间公式锚点。"""
    lane_width = max(0.1, lane.right - lane.left)
    markers = [
        (line, bbox)
        for line, bbox in lane.lines
        if (parts := _split_trailing_formula_number(line.text)) is not None
        and not parts[0]
        and abs(lane.right - bbox[2]) <= max(3.0, 0.02 * lane_width)
    ]
    output: list[_FormulaAnchor] = []
    for line, bbox in markers:
        if not any(
            other_line.source_index != line.source_index
            and abs(_bbox_center_y(other_bbox) - _bbox_center_y(bbox)) <= 6.0 * median_height
            for other_line, other_bbox in markers
        ):
            continue
        line_height = _line_effective_height(line, bbox)
        left_peers = [
            (other_line, other_bbox)
            for other_line, other_bbox in lane.lines
            if other_line.source_index != line.source_index
            and _bbox_center_x(other_bbox) < bbox[0]
            and _formula_detached_seed_vertical_match(
                bbox,
                line_height,
                other_bbox,
                _line_effective_height(other_line, other_bbox),
            )
        ]
        if len(left_peers) < 2 or not any(
            peer.formula_candidate_only or peer.compact_formula_cluster or peer.font_coverage < 0.75
            for peer, _peer_bbox in left_peers
        ):
            continue
        center_y = _bbox_center_y(bbox)
        detached_above = body_interval is None or center_y < body_interval[0]
        detached_below = body_interval is not None and center_y > body_interval[1]
        output.append(
            _FormulaAnchor(
                line=line,
                bbox=bbox,
                detached_below_body=detached_below,
                detached_above_body=detached_above,
                repeated_number_band=True,
            )
        )
    return output


def _find_formula_spatial_anchors(
    lane: _TextLane,
    median_height: float,
    dominant_body_font: tuple[str, int] | None = None,
) -> list[_FormulaAnchor]:
    """查找栏带右缘短块或带编号后缀的非正文字体公式锚点。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_interval = _formula_lane_body_interval(lane, median_height)
    repeated_anchors = _find_repeated_formula_number_anchors(
        lane,
        median_height,
        body_interval,
    )
    if body_interval is None:
        return _deduplicate_formula_anchors(repeated_anchors, median_height)
    body_top, body_bottom = body_interval
    anchors: list[_FormulaAnchor] = list(repeated_anchors)
    repeated_sources = {anchor.line.source_index for anchor in repeated_anchors}
    for line, bbox in lane.lines:
        if line.source_index in repeated_sources:
            continue
        line_height = _line_effective_height(line, bbox)
        line_width = bbox[2] - bbox[0]
        is_short_right_anchor = line_width <= max(4.0 * line_height, 0.12 * lane_width)
        has_formula_number_suffix = _split_trailing_formula_number(line.text) is not None
        is_wide_numbered_anchor = (
            has_formula_number_suffix
            and line_width <= 0.75 * lane_width
            and dominant_body_font is not None
            and (line.font_signature != dominant_body_font or line.font_coverage < 0.75)
        )
        if not is_short_right_anchor and not is_wide_numbered_anchor:
            continue
        same_row_fragments = [
            other_line
            for other_line, _other_bbox in lane.lines
            if line.visual_row_id is not None
            and other_line.visual_row_id == line.visual_row_id
            and (line.split_from_row or other_line.split_from_row)
        ]
        if len(same_row_fragments) >= 3 and not any(
            other_line.font_coverage < 0.75
            or (dominant_body_font is not None and other_line.font_signature != dominant_body_font)
            for other_line in same_row_fragments
            if other_line.source_index != line.source_index
        ):
            # 一条粗行被多个大空格拆成密集词组时更像普通排版行，不能把末词当作公式编号锚点。
            continue
        if _split_visual_row_has_prose_continuation(
            lane,
            line,
            same_row_fragments,
            median_height,
        ):
            continue
        if abs(lane.right - bbox[2]) > max(3.0, 0.02 * lane_width):
            continue
        center_y = _bbox_center_y(bbox)
        detached_below_body = body_bottom < center_y <= body_bottom + 6.0 * median_height
        detached_above_body = body_top - 6.0 * median_height <= center_y < body_top
        if not body_top <= center_y <= body_bottom and not detached_below_body and not detached_above_body:
            continue
        left_peers = [
            (other_line, other_bbox)
            for other_line, other_bbox in lane.lines
            if other_line.source_index != line.source_index
            and (
                other_bbox[2] - other_bbox[0] <= 0.75 * lane_width
                or _is_wide_tagged_formula_member(
                    line,
                    other_line,
                    other_bbox,
                    lane_width,
                )
            )
            and _bbox_center_x(other_bbox) < bbox[0]
            and (
                _formula_detached_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
                if detached_below_body
                or detached_above_body
                or _is_wide_tagged_formula_member(
                    line,
                    other_line,
                    other_bbox,
                    lane_width,
                )
                else _formula_seed_vertical_match(
                    bbox,
                    line_height,
                    other_bbox,
                    _line_effective_height(other_line, other_bbox),
                )
            )
        ]
        if is_short_right_anchor and not has_formula_number_suffix:
            if (
                re.fullmatch(r"[A-Za-z]{3,}[.,]?", line.text.strip())
                and line.font_coverage >= 0.75
                and left_peers
                and all(
                    re.fullmatch(r"\d+[.,]?", peer.text.strip())
                    and peer.font_signature == line.font_signature
                    or _display_math_has_prose(peer.text)
                    and not _formula_line_has_math_operator(peer.text)
                    for peer, _bounds in left_peers
                )
            ):
                # 页号与同字体普通词语构成边缘排版行，空间分离不提供数学证据。
                continue
            # 非编号短锚点必须与左侧主体真正分离；分母字符与正文横向重叠时不能扩张成公式。
            if any(_bbox_axis_overlap_ratio(bbox, other_bbox, axis="x") >= 0.5 for _other_line, other_bbox in left_peers):
                continue
            minimum_gap = max(0.5, 0.1 * line_height)
            if not any(bbox[0] - other_bbox[2] >= minimum_gap for _other_line, other_bbox in left_peers):
                continue
        if left_peers:
            anchors.append(
                _FormulaAnchor(
                    line=line,
                    bbox=bbox,
                    detached_below_body=detached_below_body,
                    detached_above_body=detached_above_body,
                )
            )
    return _deduplicate_formula_anchors(anchors, median_height)


def _split_visual_row_has_prose_continuation(
    lane: _TextLane,
    anchor_line: _LineItem,
    same_row_fragments: list[_LineItem],
    median_height: float,
) -> bool:
    """识别覆盖大部分栏宽且紧接下一正文行的同行拆分文本。"""

    if len(same_row_fragments) < 3 or anchor_line.visual_row_id is None:
        return False
    fragment_sources = {line.source_index for line in same_row_fragments}
    fragment_geometry = [(line, bbox) for line, bbox in lane.lines if line.source_index in fragment_sources]
    if len(fragment_geometry) < 3:
        return False
    lane_width = max(0.1, lane.right - lane.left)
    row_bbox = _bbox_union_many([bbox for _line, bbox in fragment_geometry])
    if row_bbox[2] - row_bbox[0] < 0.75 * lane_width:
        return False
    row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in fragment_geometry)
    if any(
        abs(_bbox_center_y(bbox) - row_center) > 0.25 * median_height
        or not 0.7 * median_height <= _line_effective_height(line, bbox) <= 1.3 * median_height
        for line, bbox in fragment_geometry
    ):
        return False
    following_rows = [
        (line, bbox)
        for line, bbox in lane.lines
        if line.source_index not in fragment_sources and _bbox_center_y(bbox) > row_center + 0.5 * median_height
    ]
    if not following_rows:
        return False
    following_line, following_bbox = min(
        following_rows,
        key=lambda item: (_bbox_center_y(item[1]), item[1][0]),
    )
    following_height = _line_effective_height(
        following_line,
        following_bbox,
    )
    return (
        following_bbox[2] - following_bbox[0] >= 0.6 * lane_width
        and abs(following_bbox[0] - lane.left) <= 0.75 * median_height
        and 0.75 * median_height <= following_height <= 1.3 * median_height
        and following_bbox[1] - row_bbox[3] <= 1.25 * median_height
    )


def _infer_formula_body_font(
    lane: _TextLane,
    median_height: float,
) -> tuple[str, int] | None:
    """从栏内常规宽正文行推断 dominant font，供公式扩张排除正文前缀。"""

    lane_width = max(0.1, lane.right - lane.left)
    font_counts: dict[tuple[str, int], int] = {}
    for line, bbox in lane.lines:
        line_height = _line_effective_height(line, bbox)
        if bbox[2] - bbox[0] < 0.35 * lane_width:
            continue
        if not 0.8 * median_height <= line_height <= 1.25 * median_height:
            continue
        if line.font_signature is None or line.font_coverage < 0.75:
            continue
        font_counts[line.font_signature] = font_counts.get(line.font_signature, 0) + 1
    if not font_counts:
        return None
    return max(font_counts.items(), key=lambda item: (item[1], item[0]))[0]


def _formula_lane_body_interval(
    lane: _TextLane,
    median_height: float,
) -> tuple[float, float] | None:
    """用连续出现的常规宽行确定栏带正文纵向范围，排除孤立页眉。"""

    lane_width = max(0.1, lane.right - lane.left)
    body_lines = sorted(
        (item for item in lane.lines if item[1][2] - item[1][0] >= max(4.0 * _line_effective_height(*item), 0.35 * lane_width)),
        key=lambda item: (item[1][1], item[1][0]),
    )
    if len(body_lines) < 3:
        return None
    dense_lines: list[tuple[_LineItem, BBox]] = []
    for index, item in enumerate(body_lines):
        has_close_previous = index > 0 and item[1][1] - body_lines[index - 1][1][3] <= 1.5 * median_height
        has_close_next = index + 1 < len(body_lines) and body_lines[index + 1][1][1] - item[1][3] <= 1.5 * median_height
        if has_close_previous or has_close_next:
            dense_lines.append(item)
    if len(dense_lines) < 3:
        return None
    return (
        min(bbox[1] for _line, bbox in dense_lines),
        max(bbox[3] for _line, bbox in dense_lines),
    )


def _deduplicate_formula_anchors(
    anchors: list[_FormulaAnchor],
    median_height: float,
) -> list[_FormulaAnchor]:
    """同一高度出现多个右缘短块时只保留最靠右的空间锚点。"""

    if not anchors:
        return []
    output: list[_FormulaAnchor] = []
    tolerance = max(1.5, 0.35 * median_height)
    for anchor in sorted(anchors, key=lambda item: (_bbox_center_y(item.bbox), -item.bbox[2])):
        if output and abs(_bbox_center_y(anchor.bbox) - _bbox_center_y(output[-1].bbox)) <= tolerance:
            if (anchor.bbox[2], -anchor.bbox[0]) > (output[-1].bbox[2], -output[-1].bbox[0]):
                output[-1] = anchor
            continue
        output.append(anchor)
    return output


def _grow_formula_spatial_component(
    lane: _TextLane,
    anchor: _FormulaAnchor,
    band_top: float,
    band_bottom: float,
    claimed_source_indices: set[int],
    table_bboxes: list[BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """从右缘锚点的左侧首批成员出发，按二维邻接扩展公式分量。"""

    anchor_line, anchor_bbox = anchor.line, anchor.bbox
    anchor_geometry = (anchor_line, anchor_bbox)
    lane_width = max(0.1, lane.right - lane.left)
    candidates = [
        item
        for item in lane.lines
        if item[0].source_index not in claimed_source_indices
        and (
            item[1][2] - item[1][0] <= 0.8 * lane_width
            or _is_wide_tagged_formula_member(
                anchor_line,
                item[0],
                item[1],
                lane_width,
            )
        )
        and band_top <= _bbox_center_y(item[1]) <= band_bottom
        and not _is_formula_body_barrier(
            item,
            lane,
            dominant_body_font,
            median_height,
        )
        and not _is_formula_title_barrier(
            item,
            lane,
            dominant_body_font,
            median_height,
        )
        and not _is_formula_body_prefix(
            item,
            lane,
            anchor_geometry,
            dominant_body_font,
            median_height,
            minimum_font_coverage=0.5 if anchor.repeated_number_band else 0.75,
        )
    ]
    seeds = [
        item
        for item in candidates
        if item[0].source_index != anchor_line.source_index
        and _bbox_center_x(item[1]) < anchor_bbox[0]
        and (
            _formula_detached_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
            if anchor.detached_below_body
            or anchor.detached_above_body
            or _is_wide_tagged_formula_member(
                anchor_line,
                item[0],
                item[1],
                lane_width,
            )
            else _formula_seed_vertical_match(
                anchor_bbox,
                _line_effective_height(anchor_line, anchor_bbox),
                item[1],
                _line_effective_height(*item),
            )
        )
        and not _connection_crosses_table(anchor_line.bbox, item[0].bbox, table_bboxes)
    ]
    if not seeds:
        return []

    members = [anchor_geometry, *seeds]
    additions = _grow_formula_component_native(members, candidates, table_bboxes)
    if additions is not None:
        return [*members, *additions]
    member_sources = {line.source_index for line, _bbox in members}
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            candidate_line, candidate_bbox = candidate
            if candidate_line.source_index in member_sources:
                continue
            if any(
                _formula_lines_are_connected(
                    member_line,
                    member_bbox,
                    candidate_line,
                    candidate_bbox,
                    table_bboxes,
                )
                for member_line, member_bbox in members
            ):
                members.append(candidate)
                member_sources.add(candidate_line.source_index)
                changed = True
    return members


def _grow_formula_component_native(members, candidates, table_bboxes):
    """只批量执行数值连通；语义屏障、种子和最终公式重放维持原有 Python 路径。"""
    native = get_native()
    if native is None:
        return None
    items = [*members, *candidates]
    if any(
        type(line.source_index) is not int or not -(2**63) <= line.source_index < 2**63 or type(line.angle) is not int
        for line, _ in items
    ):
        return None
    records = [(bbox, line.bbox, _line_effective_height(line, bbox), line.angle, line.source_index) for line, bbox in items]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2**50
        for box, raw_box, height, _, _ in records
        for v in (*box, *raw_box, height)
    ):
        return None
    if any(
        any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2**50 for v in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
        for box in table_bboxes
    ):
        return None
    return [items[i] for i in native.grow_formula_component(records, len(members), table_bboxes)]


def _is_formula_body_barrier(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """识别具有稳定正文排版的行，阻止公式分量吸收正文尾行。"""

    line, bbox = candidate
    if not line.style_scale_repaired:
        if dominant_body_font is None:
            return False
        line_height = _line_effective_height(line, bbox)
        lane_width = max(0.1, lane.right - lane.left)
        return (
            line.font_signature == dominant_body_font
            and line.font_coverage >= 0.75
            and bbox[2] - bbox[0] >= 0.3 * lane_width
            and 0.8 * median_height <= line_height <= 1.25 * median_height
        )
    lane_width = max(0.1, lane.right - lane.left)
    body_style_scales = [
        _line_style_scale(other_line, other_bbox)
        for other_line, other_bbox in lane.lines
        if other_bbox[2] - other_bbox[0] >= 0.35 * lane_width and not _formula_line_has_math_operator(other_line.text)
    ]
    body_scale = statistics.median(body_style_scales) if body_style_scales else median_height
    line_scale = _line_style_scale(line, bbox)
    line_width = bbox[2] - bbox[0]
    left_aligned = abs(bbox[0] - lane.left) <= max(3.0, 0.75 * body_scale)
    return (
        not _formula_line_has_math_operator(line.text)
        and (line_width >= 0.3 * lane_width or (left_aligned and line_width >= 0.08 * lane_width))
        and 0.75 * body_scale <= line_scale <= 1.35 * body_scale
        and (
            (dominant_body_font is not None and line.font_signature == dominant_body_font and line.font_coverage >= 0.75)
            or left_aligned
        )
    )


def _is_formula_title_barrier(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
) -> bool:
    """用左对齐、字号突变和字体变化隔离公式下方的章节标题。"""

    if dominant_body_font is None:
        return False
    line, bbox = candidate
    lane_width = max(0.1, lane.right - lane.left)
    line_height = _line_effective_height(line, bbox)
    return (
        line.font_signature is not None
        and line.font_signature != dominant_body_font
        and line.font_coverage >= 0.75
        and 1.1 * median_height <= line_height <= 1.6 * median_height
        and bbox[2] - bbox[0] >= 0.25 * lane_width
        and abs(bbox[0] - lane.left) <= median_height
    )


def _is_formula_body_prefix(
    candidate: tuple[_LineItem, BBox],
    lane: _TextLane,
    anchor: tuple[_LineItem, BBox],
    dominant_body_font: tuple[str, int] | None,
    median_height: float,
    *,
    minimum_font_coverage: float = 0.75,
) -> bool:
    """识别锚点上方左对齐的常规正文行，防止公式空间扩张越界认领。"""

    line, bbox = candidate
    anchor_line, anchor_bbox = anchor
    if line.formula_candidate_only:
        return False
    line_height = _line_effective_height(line, bbox)
    anchor_height = _line_effective_height(anchor_line, anchor_bbox)
    if _bbox_center_y(bbox) > _bbox_center_y(anchor_bbox) - 0.2 * max(line_height, anchor_height):
        return False
    if abs(bbox[0] - lane.left) > max(3.0, 0.75 * median_height):
        return False
    if not line.style_scale_repaired and not anchor_line.style_scale_repaired:
        return (
            dominant_body_font is not None
            and line.font_signature == dominant_body_font
            and line.font_coverage >= minimum_font_coverage
            and 0.8 * median_height <= line_height <= 1.25 * median_height
        )
    lane_width = max(0.1, lane.right - lane.left)
    if bbox[2] - bbox[0] < 0.08 * lane_width:
        return False
    if _formula_line_has_math_operator(line.text):
        return False
    body_style_scales = [
        _line_style_scale(other_line, other_bbox)
        for other_line, other_bbox in lane.lines
        if other_bbox[2] - other_bbox[0] >= 0.35 * lane_width
        and abs(other_bbox[0] - lane.left) <= max(3.0, 0.75 * median_height)
        and not _formula_line_has_math_operator(other_line.text)
    ]
    body_scale = statistics.median(body_style_scales) if body_style_scales else median_height
    line_scale = _line_style_scale(line, bbox)
    return 0.75 * body_scale <= line_scale <= 1.35 * body_scale and (
        dominant_body_font is None or line.font_signature == dominant_body_font or line.font_coverage <= minimum_font_coverage
    )


def _formula_detached_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """放宽正文密集区下方锚点的同高匹配，以接纳多行分段公式底部。"""

    has_vertical_overlap = min(anchor_bbox[3], candidate_bbox[3]) > max(anchor_bbox[1], candidate_bbox[1])
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return has_vertical_overlap or center_difference <= max(anchor_height, candidate_height)


def _formula_seed_vertical_match(
    anchor_bbox: BBox,
    anchor_height: float,
    candidate_bbox: BBox,
    candidate_height: float,
) -> bool:
    """判断左侧短行是否与右缘锚点处在同一公式高度带。"""

    overlap_ratio = _bbox_axis_overlap_ratio(anchor_bbox, candidate_bbox, axis="y")
    center_difference = abs(_bbox_center_y(anchor_bbox) - _bbox_center_y(candidate_bbox))
    return overlap_ratio >= 0.3 or center_difference <= 0.6 * max(anchor_height, candidate_height)


def _formula_lines_are_connected(
    first_line: _LineItem,
    first_bbox: BBox,
    second_line: _LineItem,
    second_bbox: BBox,
    table_bboxes: list[BBox],
) -> bool:
    """按垂直接近和水平覆盖判断两个公式成员是否空间连通。"""

    if first_line.angle != second_line.angle:
        return False
    if _connection_crosses_table(first_line.bbox, second_line.bbox, table_bboxes):
        return False
    first_height = _line_effective_height(first_line, first_bbox)
    second_height = _line_effective_height(second_line, second_bbox)
    pair_height = max(first_height, second_height)
    vertical_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="y")
    vertical_gap = max(first_bbox[1] - second_bbox[3], second_bbox[1] - first_bbox[3], 0.0)
    if vertical_overlap < 0.2 and vertical_gap > 0.6 * pair_height:
        return False
    horizontal_overlap = _bbox_axis_overlap_ratio(first_bbox, second_bbox, axis="x")
    horizontal_gap = max(first_bbox[0] - second_bbox[2], second_bbox[0] - first_bbox[2], 0.0)
    return horizontal_overlap > 0.0 or horizontal_gap <= 1.5 * pair_height


def _is_detached_formula_sidecar(
    anchor: tuple[_LineItem, BBox],
    members: list[tuple[_LineItem, BBox]],
    median_height: float,
) -> bool:
    """仅依据 bbox 判断右侧锚点是否为与公式主体分离的窄幅 sidecar。"""

    anchor_line, anchor_bbox = anchor
    body_bboxes = [bbox for line, bbox in members if line.source_index != anchor_line.source_index]
    if not body_bboxes:
        return False

    body_bbox = _bbox_union_many(body_bboxes)
    component_bbox = _bbox_union(body_bbox, anchor_bbox)
    effective_height = max(0.1, median_height)
    anchor_width = max(0.0, anchor_bbox[2] - anchor_bbox[0])
    component_width = max(0.1, component_bbox[2] - component_bbox[0])
    horizontal_gap = anchor_bbox[0] - body_bbox[2]
    right_tolerance = max(0.5, 0.1 * effective_height)
    minimum_gap = max(2.5 * effective_height, 0.08 * component_width)

    return (
        anchor_bbox[0] >= body_bbox[2]
        and anchor_bbox[2] >= component_bbox[2] - right_tolerance
        and anchor_width <= 2.0 * effective_height
        and horizontal_gap > minimum_gap
    )


def _split_trailing_formula_number(text: str) -> tuple[str, str] | None:
    """拆出右缘文本末尾的圆括号公式序号，并保留序号前的标点或正文。"""

    match = _FORMULA_NUMBER_SUFFIX_RE.fullmatch(str(text or "").strip())
    if match is None:
        return None
    return match.group("prefix").rstrip(), match.group("marker").strip()


def _formula_members_to_block(
    members: list[tuple[_LineItem, BBox]],
    page_size: tuple[float, float],
    angle: int,
    *,
    anchor_source_index: int,
    include_member_ids: bool = False,
) -> dict[str, Any] | None:
    """把公式空间分量按视觉行聚类，将编号序列化为 tag 并后置其他 sidecar。"""

    anchor_line = next(
        (line for line, _bbox in members if line.source_index == anchor_source_index),
        None,
    )
    anchor_formula_number_parts = _split_trailing_formula_number(anchor_line.text) if anchor_line is not None else None
    heights = [_line_effective_height(line, bbox) for line, bbox in members]
    median_height = statistics.median(heights) if heights else 1.0
    row_tolerance = max(1.5, 0.35 * median_height)
    rows: list[list[tuple[_LineItem, BBox]]] = []
    for member in sorted(members, key=lambda item: (_bbox_center_y(item[1]), item[1][0], item[0].source_index)):
        if not rows:
            rows.append([member])
            continue
        row_center = statistics.median(_bbox_center_y(bbox) for _line, bbox in rows[-1])
        if abs(_bbox_center_y(member[1]) - row_center) <= row_tolerance:
            rows[-1].append(member)
        else:
            rows.append([member])

    trailing_sidecar_content: str | None = None
    # 右侧 sidecar 按视觉 y 常落在分式中部；仅在其后仍有公式行时转为逻辑末行。
    for row_index, row in enumerate(rows[:-1]):
        anchor_member = next(
            (member for member in row if member[0].source_index == anchor_source_index),
            None,
        )
        if anchor_member is None:
            continue
        formula_number_parts = _split_trailing_formula_number(anchor_member[0].text)
        if formula_number_parts is not None:
            prefix, marker = formula_number_parts
            rows[row_index] = [
                (
                    (replace(member[0], text=prefix), member[1])
                    if member[0].source_index == anchor_source_index and prefix
                    else member
                )
                for member in row
                if member[0].source_index != anchor_source_index or prefix
            ]
            trailing_sidecar_content = marker
        elif _is_detached_formula_sidecar(anchor_member, members, median_height):
            rows[row_index] = [member for member in row if member[0].source_index != anchor_source_index]
            trailing_sidecar_content = anchor_member[0].text.strip()
        break

    row_contents = [_join_formula_visual_row(row, page_size) for row in rows if row]
    if trailing_sidecar_content is not None:
        row_contents.append(trailing_sidecar_content)
    content = _sanitize_pdf_control_text("\n".join(filter(None, row_contents)), preserve_newlines=True)
    if anchor_formula_number_parts is not None:
        _anchor_prefix, tag_content = anchor_formula_number_parts
        stripped_content = content.rstrip()
        if stripped_content.endswith(tag_content):
            formula_content = stripped_content[: -len(tag_content)].rstrip()
            tagged_content = build_tagged_formula_content(formula_content, tag_content)
            if tagged_content is not None:
                content = tagged_content
    if not content.strip():
        return None
    block = {
        "type": "equation",
        "bbox": _bbox_union_many([line.bbox for line, _bbox in members]),
        "angle": angle,
        "content": content,
    }
    tight_output_bbox = _lines_tight_output_bbox(
        [line for line, _bbox in members],
        page_size,
    )
    if tight_output_bbox is not None:
        block["_tight_output_bbox"] = tight_output_bbox
    if include_member_ids:
        block["_formula_members"] = [line.source_index for line, _bbox in members]
    return block
