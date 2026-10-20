"""根据页面框架、正文栏及独立图题展开页面 Form，保守保留真实整图。"""

from __future__ import annotations

from dataclasses import replace
import statistics

from ....document.pdf.form_structure import _PDFFormInfo
from ....schema import BBox
from .geometry import _bbox_area, _bbox_overlap_in_first
from .graphics import _detect_strong_graphic_bboxes
from .layout_evidence import build_layout_evidence
from .line_layout import _line_effective_height
from .models import _LineItem, _PageSource
from .visual_annotations import _is_strong_caption_text


def form_owns_line(line: _LineItem, members: frozenset[int]) -> bool:
    """按去重后字符保留的原来源判定归属，不把矩形重叠当作对象成员关系。"""
    chars = [char for char in line.chars if str(char.get("char", "")).strip()]
    return bool(chars) and all(
        any(index in members for index in char.get("source_indices", (char.get("char_idx"),))) for char in chars
    )


def _page_frame_matches(source: _PageSource, form: _PDFFormInfo) -> bool:
    """声明框映射到 MediaBox 或 CropBox 的各边误差不超过对应页面尺寸的百分之二。"""
    if not form.structure_valid or form.declared_bbox is None:
        return False
    width, height = source.page_size
    tolerance = (0.02 * width, 0.02 * height, 0.02 * width, 0.02 * height)
    return any(
        all(abs(a - b) <= limit for a, b, limit in zip(form.declared_bbox, frame, tolerance, strict=True))
        for frame in ((0.0, 0.0, width, height), form.media_bbox)
    )


def _form_masks(source: _PageSource, form: _PDFFormInfo) -> list[BBox]:
    """保护栅格图、独立子 Form 和按候选局部深度识别的统一绘图核心。"""
    children = [
        child.bbox
        for child in source.form_infos
        if child.parent_id == form.instance_id
        and not _page_frame_matches(source, child)
        and _bbox_area(child.bbox) >= 0.01 * source.page_size[0] * source.page_size[1]
    ]
    relative_paths = [
        replace(path, form_depth=max(0, path.form_depth - len(form.occurrence)))
        for path in source.path_infos
        if path.source_index in form.path_indices
    ]
    em = statistics.median([_line_effective_height(line, line.bbox) for line in source.lines]) if source.lines else 10.0
    # 两段且横纵都有跨度的描边是斜连线，参与图形核心；矩形表格网格不满足此条件。
    graphic_paths = [
        replace(path, segment_count=6)
        if path.segment_count == 2
        and path.stroke_visible
        and min(path.bbox[2] - path.bbox[0], path.bbox[3] - path.bbox[1]) >= 1.5 * em
        else path
        for path in relative_paths
    ]
    # 页面角色需要保护满页绘图；扩大检测尺度只取消现有图形核心的页内尺寸上限。
    local = replace(
        source, path_infos=graphic_paths, page_size=tuple(2 * size for size in source.page_size), drawing_component_cache=[]
    )
    connectors = [
        path
        for path in relative_paths
        if path.segment_count == 2
        and path.stroke_visible
        and min(path.bbox[2] - path.bbox[0], path.bbox[3] - path.bbox[1]) >= 0.5 * em
    ]
    # 封闭框内同时有多个绘图元素和斜连线时保护整个绘图；纯横纵表格网格不触发。
    connected_frames = [
        path.bbox
        for path in relative_paths
        if path.segment_count in {4, 5}
        and any(_bbox_overlap_in_first(connector.bbox, path.bbox) >= 0.9 for connector in connectors)
        and sum(
            other.source_index != path.source_index and _bbox_overlap_in_first(other.bbox, path.bbox) >= 0.9
            for other in relative_paths
        )
        >= 4
    ]
    return [*form.image_bboxes, *children, *connected_frames, *_detect_strong_graphic_bboxes(local)]


def _masked_line(bbox: BBox, masks: list[BBox]) -> bool:
    """按遮罩并集计算覆盖率，多面板跨图长标签不能被误当作外部正文。"""
    rectangles = [
        (max(bbox[0], mask[0]), max(bbox[1], mask[1]), min(bbox[2], mask[2]), min(bbox[3], mask[3])) for mask in masks
    ]
    rectangles = [rect for rect in rectangles if rect[2] > rect[0] and rect[3] > rect[1]]
    edges = sorted({x for rect in rectangles for x in (rect[0], rect[2])})
    area = 0.0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((rect[1], rect[3]) for rect in rectangles if rect[0] <= left and rect[2] >= right)
        top = bottom = 0.0
        covered = 0.0
        for start, end in intervals:
            if start > bottom:
                covered += bottom - top
                top, bottom = start, end
            else:
                bottom = max(bottom, end)
        area += (right - left) * (covered + bottom - top)
    return area >= 0.8 * max(0.1, _bbox_area(bbox))


def _has_body_lane(source: _PageSource, form: _PDFFormInfo, masks: list[BBox], *, minimum_rows: int = 6) -> bool:
    """六行以上连续正文、稳定栏缘和长行共同支持页面角色，图内标签不参与。"""
    evidence_lines = source.lines
    if not evidence_lines and source.chars:
        from .native_text import _build_native_line_items_from_chars

        # 180 度页面的正文目前不进入公开组装；角色证据仍须识别，保持包装前后的原有行为一致。
        evidence_lines = _build_native_line_items_from_chars(
            source.chars, source.page_size, supported_angles=(0.0, 90.0, 180.0, 270.0)
        )
    lines = [
        line
        for line in evidence_lines
        if line.semantic_type is None
        and form_owns_line(line, form.text_indices)
        and not _is_strong_caption_text(line.text)
        and not _masked_line(line.ink_bbox or line.bbox, masks)
    ]
    for angle in {line.angle for line in lines}:
        local = [line for line in lines if line.angle == angle]
        layout = build_layout_evidence(local, source.page_size, angle=angle)
        for lane in layout.lanes:
            if lane.is_span or len(lane.lines) < minimum_rows:
                continue
            width = lane.right - lane.left
            heights = [_line_effective_height(line, bbox) for line, bbox in lane.lines]
            em = statistics.median(heights)
            long_rows = sorted(
                (
                    bbox
                    for line, bbox in lane.lines
                    if bbox[2] - bbox[0] >= max(10 * em, 0.6 * width)
                    and (abs(bbox[0] - lane.left) <= em or abs(bbox[2] - lane.right) <= em)
                ),
                key=lambda bbox: bbox[1],
            )
            consecutive = 0
            previous = None
            for bbox in long_rows:
                consecutive = consecutive + 1 if previous is not None and -0.2 * em <= bbox[1] - previous[3] <= 2 * em else 1
                if consecutive >= minimum_rows:
                    return True
                previous = bbox
    return False


def _has_independent_figures(source: _PageSource, form: _PDFFormInfo, masks: list[BBox]) -> bool:
    """每个图形须拥有不同的独立图题；共同图题下的多个面板不构成展开证据。"""
    captions = [
        line
        for line in source.lines
        if form_owns_line(line, form.text_indices)
        and _is_strong_caption_text(line.text)
        and not _masked_line(line.ink_bbox or line.bbox, masks)
    ]
    em = statistics.median([_line_effective_height(line, line.bbox) for line in source.lines]) if source.lines else 10.0
    paired = set()
    for mask in masks:
        matches = [
            caption
            for caption in captions
            if min(mask[2], caption.bbox[2]) > max(mask[0], caption.bbox[0])
            and min(abs(mask[1] - caption.bbox[3]), abs(caption.bbox[1] - mask[3])) <= 4 * em
            and _bbox_overlap_in_first(caption.bbox, mask) < 0.5
        ]
        if matches:
            paired.add(
                min(matches, key=lambda line: min(abs(mask[1] - line.bbox[3]), abs(line.bbox[1] - mask[3]))).source_index
            )
    return len(paired) >= 2


def _is_underlay(source: _PageSource, form: _PDFFormInfo) -> bool:
    """只在无内部正文/图片、仅有简单页面矩形且有外部正文时确认装饰底层。"""
    paths = [path for path in source.path_infos if path.source_index in form.path_indices]
    if (
        form.paint_order != 0
        or form.text_indices
        or form.image_bboxes
        or not paths
        or any(path.segment_count not in {4, 5} for path in paths)
    ):
        return False
    if not all(_bbox_area(path.bbox) >= 0.7 * source.page_size[0] * source.page_size[1] for path in paths):
        return False
    return any(line.semantic_type is None and not form_owns_line(line, form.text_indices) for line in source.lines)


def _has_native_table_regions(source: _PageSource, form: _PDFFormInfo, masks: list[BBox]) -> bool:
    """已确认页面家族内，充分的原生表格成员可支持纯表格页；大图核心仍优先保护。"""
    if any(_bbox_area(mask) >= 0.05 * source.page_size[0] * source.page_size[1] for mask in masks):
        return False
    from .tables import _detect_table_candidates

    lines = [line for line in source.lines if line.semantic_type is None and form_owns_line(line, form.text_indices)]
    if len(lines) < 6:
        return False
    local = replace(
        source,
        lines=lines,
        form_bboxes=[],
        path_infos=[
            replace(path, form_depth=max(0, path.form_depth - len(form.occurrence)))
            for path in source.path_infos
            if path.source_index in form.path_indices
        ],
        drawing_component_cache=[],
    )
    candidates = _detect_table_candidates(local, excluded_bboxes=masks)
    members = set().union(*(candidate.line_indices for candidate in candidates))
    return len(members) >= max(6, 0.5 * len(lines))


def _layout_framework(source: _PageSource, form: _PDFFormInfo, masks: list[BBox]) -> list[tuple[float, float, float]]:
    """复用栏位证据保存稳定栏缘，跨页辅助判断同时要求页面映射和版式对应。"""
    lines = [
        line
        for line in source.lines
        if form_owns_line(line, form.text_indices) and not _masked_line(line.ink_bbox or line.bbox, masks)
    ]
    output = []
    for angle in {line.angle for line in lines}:
        layout = build_layout_evidence([line for line in lines if line.angle == angle], source.page_size, angle=angle)
        width = source.page_size[1] if angle in {90, 270} else source.page_size[0]
        output.extend(
            (angle, lane.left / width, lane.right / width) for lane in layout.lanes if not lane.is_span and len(lane.lines) >= 3
        )
    # 表格单元格文字可能居中，重复水平格线提供比文字栏缘更稳定的同版式边界。
    rules: dict[tuple[float, float], int] = {}
    width = source.page_size[0]
    for path in source.path_infos:
        if (
            path.source_index in form.path_indices
            and path.stroke_visible
            and path.bbox[3] - path.bbox[1] <= 2
            and path.bbox[2] - path.bbox[0] >= 0.2 * width
        ):
            edges = (round(path.bbox[0] / width, 3), round(path.bbox[2] / width, 3))
            rules[edges] = rules.get(edges, 0) + 1
    output.extend((0, left, right) for (left, right), count in rules.items() if count >= 3)
    return output


def normalize_page_forms(sources: list[_PageSource]) -> None:
    """一次判定全文页面角色，再同步候选、成员归属及用于分析的相对 Form 深度。"""
    decisions: list[tuple[_PageSource, set[int], set[int]]] = []
    families: dict[tuple[float, ...], list[tuple[float, float, float]]] = {}
    for source in sources:
        expanded: set[int] = set()
        decorations: set[int] = set()
        for form in source.form_infos:
            if not _page_frame_matches(source, form):
                continue
            masks = _form_masks(source, form)
            if _is_underlay(source, form):
                decorations.add(form.instance_id)
            elif _has_body_lane(source, form, masks) or _has_independent_figures(source, form, masks):
                expanded.add(form.instance_id)
                family = tuple(round(v / source.page_size[i % 2], 2) for i, v in enumerate(form.declared_bbox))
                families.setdefault(family, []).extend(_layout_framework(source, form, masks))
        decisions.append((source, expanded, decorations))

    for source, expanded, decorations in decisions:
        if not source.form_infos:
            continue
        # 稀疏正文页可继承已确认页面框架；任何内部图形核心仍阻止仅凭跨页证据展开。
        for form in source.form_infos:
            if form.instance_id in expanded | decorations or not _page_frame_matches(source, form):
                continue
            family = tuple(round(v / source.page_size[i % 2], 2) for i, v in enumerate(form.declared_bbox))
            if family in families:
                masks = _form_masks(source, form)
                framework = _layout_framework(source, form, masks)
                compatible = any(
                    angle == other_angle and (abs(left - other_left) <= 0.03 or abs(right - other_right) <= 0.03)
                    for angle, left, right in framework
                    for other_angle, other_left, other_right in families[family]
                )
                if compatible and (
                    _has_body_lane(source, form, masks, minimum_rows=3) or _has_native_table_regions(source, form, masks)
                ):
                    expanded.add(form.instance_id)
        # 仅展开页面根或已展开页面容器的直接后代，不穿透仍保留的真实图形。
        for form in source.form_infos:
            if form.parent_id is not None and form.parent_id not in expanded | decorations:
                expanded.discard(form.instance_id)
                decorations.discard(form.instance_id)
        removed = expanded | decorations
        forms = {form.instance_id: form for form in source.form_infos}
        active = [
            form
            for form in source.form_infos
            if form.instance_id not in removed and (form.parent_id is None or form.parent_id in removed)
        ]
        if not removed:
            # 未展开时保留既有顶层候选范围；结构读取失败也不能提升未知子 Form。
            active = [form for form in source.form_infos if form.parent_id is None]
        source.form_bboxes = [form.bbox for form in active if _bbox_area(form.bbox) > 0]
        source.retained_page_forms = {form.bbox for form in active if _page_frame_matches(source, form)}
        # 同一位置重复调用也可能拥有不同成员；不能让后一次调用覆盖前一次来源。
        valid_boxes = {form.bbox for form in active if form.structure_valid} - {
            form.bbox for form in active if not form.structure_valid
        }
        source.form_member_sources = {
            bbox: frozenset().union(*(form.text_indices for form in active if form.bbox == bbox)) for bbox in valid_boxes
        }
        source.form_path_sources = {
            bbox: frozenset().union(*(form.path_indices for form in active if form.bbox == bbox)) for bbox in valid_boxes
        }
        source.path_infos = [
            replace(
                path,
                form_depth=max(0, path.form_depth - sum(path.source_index in forms[index].path_indices for index in expanded)),
            )
            for path in source.path_infos
        ]
        source.drawing_component_cache.clear()
