"""检测 Form、矢量图形和栅格图片并认领内部文本。"""

from __future__ import annotations

from .layout_evidence import build_layout_evidence
from .annotation_text import _is_strong_caption_text

import math
import re
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import replace
from typing import Any
from ...._compute_backend import get_native

from ....document.pdf._document import PDFPathInfo
from ....schema import BBox
from .geometry import (
    _bbox_area,
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
    _point_in_bbox,
    _rotate_bbox_to_upright,
)
from .line_layout import _infer_text_lanes, _line_effective_height, _font_signatures_share_family
from .line_merging import _join_formula_visual_row
from .models import _AxisLine, _DrawingComponentSummary, _GraphicCandidate, _LineItem, _PageSource, _TextLane
from .native_text import _fill_native_typography, _normalize_native_run_text, _sanitize_pdf_control_text

_MIN_RASTER_IMAGE_PAGE_AREA_RATIO = 0.0038

_NUMERIC_GRAPHIC_REFERENCES = (
    re.fullmatch,
    re._compile,
    getattr(getattr(re, "_compiler", None), "compile", None),
    _LineItem.__getattribute__,
    _LineItem.__setattr__,
)


def _numeric_graphic_lines(lines: list[_LineItem], mode: int) -> list[_LineItem] | None:
    """仅批量复用当前文字的数字语法；几何和语义条件仍由每次规则调用裁决。"""
    native = get_native()
    if (
        native is None
        or not hasattr(native, "graphic_numeric_lines")
        or (
            re.fullmatch,
            re._compile,
            getattr(getattr(re, "_compiler", None), "compile", None),
            _LineItem.__getattribute__,
            _LineItem.__setattr__,
        )
        != _NUMERIC_GRAPHIC_REFERENCES
    ):
        return None
    return native.graphic_numeric_lines(lines, mode, _LineItem, re.fullmatch)


_SIGNATURE_IMAGE_BBOX_DEDUP_TOLERANCE = 0.5


_MIN_FORM_IMAGE_PAGE_AREA_RATIO = 0.01


_MAX_FORM_IMAGE_PAGE_AREA_RATIO = 0.8


_IMAGE_CONTAINER_OVERLAP_THRESHOLD = 0.5


_FIGURE_CAPTION_LINE_RE = re.compile(
    r"^\s*(?:fig(?:ure)?\.?)[ \t]*\d+[A-Za-z]?(?:\s*[.:])?",
    re.IGNORECASE,
)


def _caption_graphic_display_heading_floor(source: _PageSource, caption_bbox: BBox, em: float) -> float:
    """照片上方独立居中大字号标题形成屏障，连同短续行阻止正文下划线和文字向图体扩框。"""
    floor = 0.0
    for image in source.image_bboxes:
        if not (
            image[3] <= caption_bbox[1] + 0.2 * em
            and 0 <= caption_bbox[1] - image[3] <= 8 * em
            and _bbox_axis_overlap_ratio(image, caption_bbox, axis="x") >= 0.35
        ):
            continue
        width = image[2] - image[0]
        for heading in source.lines:
            if not (
                heading.angle == 0
                and not heading.caption_start
                and _line_effective_height(heading, heading.bbox) >= 1.3 * em
                and heading.bbox[2] - heading.bbox[0] >= 0.5 * width
                and abs(_bbox_center_x(heading.bbox) - _bbox_center_x(image)) <= 0.08 * width
                and 0.5 * em <= image[1] - heading.bbox[3] <= 4 * em
            ):
                continue
            end = heading.bbox[3]
            # 同字体居中短尾行可以是括号日期或缩写，不凭长度把它降为图片标签。
            for tail in sorted(source.lines, key=lambda line: line.bbox[1]):
                if (
                    tail.angle == 0
                    and tail.font_signature == heading.font_signature
                    and 0 <= tail.bbox[1] - end <= 0.75 * em
                    and tail.bbox[3] < image[1]
                    and abs(_bbox_center_x(tail.bbox) - _bbox_center_x(image)) <= 0.08 * width
                    and 0.85 <= _line_effective_height(tail, tail.bbox) / _line_effective_height(heading, heading.bbox) <= 1.15
                ):
                    end = tail.bbox[3]
            floor = max(floor, end)
    return floor


def _build_caption_graphic_blocks(
    source: _PageSource, *, caption_line_indices: set[int], table_bboxes: list[BBox], code_bboxes: list[BBox]
) -> tuple[list[dict[str, Any]], set[int]]:
    """用独立图题、正文屏障和实际绘图证据恢复整图，允许标签全部是矢量字形。"""
    width, height = source.page_size
    heights = [_line_effective_height(line, line.bbox) for line in source.lines if line.angle == 0]
    em = statistics.median(heights) if heights else 10.0
    captions = [
        line
        for line in source.lines
        if line.angle == 0
        and _FIGURE_CAPTION_LINE_RE.match(line.text)
        and line.source_index in caption_line_indices
        and not any(_owned_form_member_bbox(source, line, form) is not None for form in source.retained_page_forms)
    ]
    # 即使图题候选被邻栏正文屏障拒绝，照片旁已确认的连续段仍要跨局部栏宽保持同一成员组。
    raster_prose = (
        _raster_outside_prose_sources(
            source,
            [image for image in source.image_bboxes if image[2] - image[0] >= 5 * em and image[3] - image[1] >= 5 * em],
            caption_line_indices,
            em,
        )
        if caption_line_indices
        else set()
    )
    primitives = list(source.image_bboxes) + [
        path.bbox
        for path in source.path_infos
        if path.fill_visible or path.stroke_visible
        if not (
            path.fill_rgba is not None
            and min(path.fill_rgba[:3]) >= 250
            and not path.stroke_visible
            and _bbox_area(path.bbox) > 0.02 * width * height
        )
    ]
    layout = build_layout_evidence(source.lines, source.page_size, barriers=table_bboxes + code_bboxes)
    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    code_members = {
        line.source_index
        for line in source.lines
        if any(_bbox_overlap_in_first(line.bbox, code) >= 0.8 for code in code_bboxes)
    }
    for caption in sorted(captions, key=lambda line: (line.bbox[1], line.bbox[0])):
        side_members = _framed_side_caption_members(source, caption)
        if side_members:
            blocks.append(
                {
                    "type": "caption",
                    "bbox": _bbox_union_many([line.bbox for line in side_members]),
                    "angle": 0,
                    "content": " ".join(line.text.strip() for line in side_members),
                    "_line_heights": [_line_effective_height(line, line.bbox) for line in side_members],
                    "_font_signatures": {line.font_signature for line in side_members if line.font_signature},
                }
            )
            claimed.update(line.source_index for line in side_members)
            continue
        cb = caption.bbox
        # 图题中的数学上下标会拆开同一物理行，使用整行投影决定是否跨栏。
        companions = [
            line.bbox
            for line in source.lines
            if line.angle == 0
            and line is not caption
            and line not in captions
            and _bbox_axis_overlap_ratio(cb, line.bbox, axis="y") >= 0.5
            and min(abs(line.bbox[0] - cb[2]), abs(cb[0] - line.bbox[2])) <= em
        ]
        if companions:
            cb = _bbox_union_many([cb, *companions])
        corridor = layout.corridor(cb)
        if corridor is None:
            continue
        left, right = corridor
        # 同排独立编号图题建立横向边界，不能因上方没有正文栏带而把两张图分别扩成同一大框。
        peers = [
            line for line in captions if line is not caption and abs(_bbox_center_y(line.bbox) - _bbox_center_y(cb)) <= 1.5 * em
        ]
        for peer in peers:
            if peer.bbox[2] <= cb[0] - 0.5 * em:
                left = max(left, (peer.bbox[2] + cb[0]) / 2)
            elif peer.bbox[0] >= cb[2] + 0.5 * em:
                right = min(right, (cb[2] + peer.bbox[0]) / 2)
        barriers = [
            line.bbox[3]
            for line in source.lines
            if line.angle == 0
            and line is not caption
            and line.bbox[3] <= cb[1] - 0.3 * em
            and left <= _bbox_center_x(line.bbox) <= right
            and (
                (line in captions)
                or (
                    (len(re.findall(r"[A-Za-z]{3,}", line.text)) >= 6 or len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 10)
                    and line.bbox[2] - line.bbox[0] > 0.25 * (right - left)
                    and line.source_index not in caption_line_indices
                    and not any(_bbox_overlap_in_first(line.bbox, form) >= 0.9 for form in source.form_bboxes)
                    and not any(
                        path.fill_visible
                        and path.segment_count >= 4
                        and _bbox_area(line.bbox) < _bbox_area(path.bbox) < 0.4 * width * height
                        and _bbox_overlap_in_first(line.bbox, path.bbox) >= 0.95
                        for path in source.path_infos
                    )
                )
            )
        ]
        floor = max(barriers, default=0.06 * height)
        floor = max(floor, _caption_graphic_display_heading_floor(source, cb, em))
        floor = max([floor, *(bbox[3] for bbox in table_bboxes if bbox[3] <= cb[1] and bbox[2] > left and bbox[0] < right)])
        floor = max([floor, *(bbox[3] for bbox in code_bboxes if bbox[3] <= cb[1] and bbox[2] > left and bbox[0] < right)])
        candidates = [
            bbox
            for bbox in primitives
            if bbox[1] >= floor - 0.3 * em
            and bbox[3] <= cb[1] + 0.2 * em
            and left <= _bbox_center_x(bbox) <= right
            and bbox[0] >= left - em
            and bbox[2] <= right + em
            and not any(_bbox_overlap_in_first(bbox, table) >= 0.25 for table in table_bboxes)
            and not any(_bbox_overlap_in_first(bbox, code) >= 0.25 for code in code_bboxes)
        ]
        if not candidates:
            continue
        bottom = max(bbox[3] for bbox in candidates)
        if cb[1] - bottom > 8 * em:
            continue
        # 同一图题上方直到正文、表格或前一图题的空白带属于同一绘图区域。
        selected = candidates
        bbox = _bbox_union_many(selected)
        raster_cores = [
            image
            for image in source.image_bboxes
            if _bbox_overlap_in_first(image, bbox) >= 0.95 and _bbox_area(image) >= 0.85 * _bbox_area(bbox)
        ]
        protected_prose = raster_prose if raster_cores else set()
        if any(_bbox_overlap_in_smaller(bbox, code) >= 0.8 for code in code_bboxes):
            continue
        if bbox[2] - bbox[0] < 4 * em or bbox[3] - bbox[1] < 3 * em:
            continue
        if len(selected) < 4 and not any(_bbox_overlap_in_first(image, bbox) >= 0.95 for image in source.image_bboxes):
            continue
        for _ in range(2):
            members = [
                line
                for line in source.lines
                if line.source_index not in claimed
                and line.source_index not in protected_prose
                and line.source_index not in code_members
                and line not in captions
                and line.source_index not in caption_line_indices
                and (line.semantic_type is None or line.angle in {90, 270})
                and line.bbox[1] >= floor
                and line.bbox[3] <= cb[1]
                and left <= _bbox_center_x(line.bbox) <= right
                and _bbox_distance(line.ink_bbox or line.bbox, bbox) <= 4 * em
                and _form_region_allows_line(source, line, bbox)
                and (
                    len(line.text.split()) < 8
                    or _bbox_overlap_in_first(line.bbox, bbox) >= 0.9
                    or any(
                        _bbox_overlap_in_first(line.bbox, form) >= 0.95 and _bbox_overlap_in_first(form, bbox) >= 0.75
                        for form in source.form_bboxes
                    )
                )
                and not (
                    _bbox_overlap_in_first(line.bbox, bbox) < 0.5
                    and (
                        line.text.rstrip().endswith(("。", "！", "？"))
                        or re.search(r"[.!?]$", line.text.rstrip())
                        and len(re.findall(r"[A-Za-z]{2,}", line.text)) >= 2
                    )
                )
            ]
            if members:
                bbox = _bbox_union_many([bbox, *(line.ink_bbox or line.bbox for line in members)])
        if any(_image_bboxes_are_near_equal(bbox, existing["bbox"]) for existing in blocks):
            continue
        blocks.append(
            {
                "type": "image",
                "bbox": bbox,
                "angle": 0,
                "content": _image_members_to_content(members, source.page_size),
                "_caption_graphic": True,
                "_native_numeric_grid": _graphic_members_are_numeric_grid(members),
            }
        )
        claimed.update(line.source_index for line in members)
    return blocks, claimed


def _framed_side_caption_members(source: _PageSource, caption: _LineItem) -> list[_LineItem]:
    """邻图旁的裸编号及同式多行说明有完整细线框时属于装饰图注卡，不能当作新图。"""
    if not re.fullmatch(r"\s*(?:figure|fig\.)\s*\d+\s*", caption.text, re.I) or caption.font_signature is None:
        return []
    em = _line_effective_height(caption, caption.bbox)
    left_images = [
        box
        for box in source.image_bboxes
        if box[2] - box[0] >= 8 * em
        and box[3] - box[1] >= 8 * em
        and 0 <= caption.bbox[0] - box[2] <= 6 * em
        and box[1] <= _bbox_center_y(caption.bbox) <= box[3]
    ]
    if len(left_images) != 1:
        return []
    description = sorted(
        [
            line
            for line in source.lines
            if line is not caption
            and line.angle == caption.angle
            and line.semantic_type is None
            and line.font_signature == caption.font_signature
            and 0.85 <= _line_effective_height(line, line.bbox) / em <= 1.15
            and 0.5 * em <= line.bbox[0] - caption.bbox[2] <= 6 * em
            and abs(_bbox_center_y(line.bbox) - _bbox_center_y(caption.bbox)) <= 3 * em
        ],
        key=lambda line: (line.bbox[1], line.bbox[0]),
    )
    if (
        len(description) < 3
        or len(re.findall(r"[A-Za-z]{2,}", " ".join(line.text for line in description))) < 8
        or max(line.bbox[0] for line in description) - min(line.bbox[0] for line in description) > 0.5 * em
        or any(
            not 0.7 * em <= _bbox_center_y(b.bbox) - _bbox_center_y(a.bbox) <= 1.75 * em
            for a, b in zip(description, description[1:])
        )
    ):
        return []
    bounds = _bbox_union_many([caption.bbox, *(line.bbox for line in description)])
    horizontal = [
        line.bbox
        for line in source.drawing_lines
        if line.orientation == "horizontal"
        and line.width <= 0.15 * em
        and line.bbox[0] <= bounds[0] + 0.25 * em
        and line.bbox[2] >= bounds[2] - 0.25 * em
    ]
    top = [box for box in horizontal if -0.1 * em <= bounds[1] - box[3] <= 0.5 * em]
    bottom = [box for box in horizontal if -0.1 * em <= box[1] - bounds[3] <= 0.5 * em]
    sides = [
        line
        for line in source.drawing_lines
        if line.orientation == "vertical"
        and line.width <= 0.15 * em
        and line.bbox[1] <= bounds[1] + 0.25 * em
        and line.bbox[3] >= bounds[3] - 0.25 * em
    ]
    if not (
        top
        and bottom
        and any(0 <= bounds[0] - line.bbox[2] <= em for line in sides)
        and any(0 <= line.bbox[0] - bounds[2] <= 1.5 * em for line in sides)
    ):
        return []
    return [caption, *description]


def _graphic_members_are_numeric_grid(members: list[_LineItem]) -> bool:
    """四排以上重复三列数字证明是原生数据网格，防止图题把电子表格误作折线图父对象。"""
    numeric = _numeric_graphic_lines(members, 2)
    if numeric is None:
        numeric = [line for line in members if re.fullmatch(r"\s*[-+]?\d+(?:[,.]\d+)*%?\s*", line.text)]
    if len(numeric) < 12:
        return False
    em = statistics.median(_line_effective_height(line, line.bbox) for line in numeric)
    rows: list[list[_LineItem]] = []
    for line in sorted(numeric, key=lambda line: (_bbox_center_y(line.bbox), line.bbox[0])):
        if not rows or abs(_bbox_center_y(line.bbox) - _bbox_center_y(rows[-1][0].bbox)) > 0.4 * em:
            rows.append([])
        rows[-1].append(line)
    dense = [row for row in rows if len(row) >= 3]
    if len(dense) < 4:
        return False
    reference = dense[0]
    return (
        sum(sum(any(abs(line.bbox[0] - other.bbox[0]) <= em for other in row) for line in reference) >= 3 for row in dense) >= 4
    )


def _raster_outside_prose_sources(source, raster_bboxes, caption_indices, em):
    """图片外同栏连续自然语言正文保持独立；短收句和同排强调碎片也不能被图题扩框认领。"""
    free = [
        line
        for line in source.lines
        if line.angle == 0
        and line.semantic_type is None
        and line.source_index not in caption_indices
        and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", line.text)) >= 1
        and 0.7 * em <= _line_effective_height(line, line.bbox) <= 1.3 * em
        and all(_bbox_overlap_in_first(line.ink_bbox or line.bbox, image) <= 0.1 for image in raster_bboxes)
    ]
    rows = []
    for line in sorted(free, key=lambda item: (item.bbox[1], item.bbox[0])):
        peers = [
            row
            for row in rows
            if _bbox_axis_overlap_ratio(row[0].bbox, line.bbox, axis="y") >= 0.65
            and -0.1 * em <= line.bbox[0] - max(member.bbox[2] for member in row) <= 3 * em
        ]
        if peers:
            peers[-1].append(line)
        else:
            rows.append([line])
    pending = list(rows)
    protected = set()
    while pending:
        run = [pending.pop(0)]
        while True:
            previous = _bbox_union_many([line.bbox for line in run[-1]])
            main = max(run[-1], key=lambda item: len(item.text))
            followers = [
                row
                for row in pending
                if max(row, key=lambda item: len(item.text)).font_signature == main.font_signature
                and abs(min(line.bbox[0] for line in row) - previous[0]) <= 0.4 * em
                and -0.25 * em <= min(line.bbox[1] for line in row) - previous[3] <= 0.85 * em
                and _bbox_center_y(_bbox_union_many([line.bbox for line in row])) > _bbox_center_y(previous) + 0.5 * em
            ]
            if not followers:
                break
            row = min(followers, key=lambda members: min(line.bbox[1] for line in members))
            # 明确短收句后恢复常规宽度属于新段，不能用照片旁连续排版抹掉原生段界。
            following_box = _bbox_union_many([line.bbox for line in row])
            if (
                len(run) >= 2
                and any(line.paragraph_terminal for line in run[-1])
                and re.search(r"[.!?。！？]$", " ".join(line.text for line in run[-1]).strip())
                and previous[2] - previous[0] < 0.65 * (following_box[2] - following_box[0])
            ):
                break
            pending.remove(row)
            run.append(row)
        text = " ".join(line.text for row in run for line in row)
        if (
            len(run) >= 3
            and len(re.findall(r"[A-Za-z]{2,}|[\u3400-\u9fff]", text)) >= 15
            and re.search(r"[.!?。！？][\d\s]*$", text.strip())
        ):
            protected.update(line.source_index for row in run for line in row)
            bounds = _bbox_union_many([line.bbox for row in run for line in row])
            if any(
                _bbox_axis_overlap_ratio(bounds, image, axis="x") <= 0.1
                and _bbox_axis_overlap_ratio(bounds, image, axis="y") >= 0.5
                and bounds[2] - bounds[0] <= 1.5 * (image[2] - image[0])
                and -0.3 * em <= bounds[1] - image[1] <= 2 * em
                and 0 <= bounds[3] - image[3] <= 5 * em
                and run[-1][0].paragraph_terminal
                and _bbox_union_many([line.bbox for line in run[-1]])[2] - bounds[0] < 0.7 * (bounds[2] - bounds[0])
                for image in raster_bboxes
            ):
                group = max((line.paragraph_group for line in source.lines if line.paragraph_group is not None), default=-1) + 1
                for row in run:
                    for line in row:
                        line.paragraph_group = group
    return protected


def _form_supersedes_nested_bbox(form_bbox: BBox, nested_bbox: BBox) -> bool:
    """判断 Form 是否应整体吞并其内部面积明显更小的候选容器。"""

    form_area = _bbox_area(form_bbox)
    nested_area = _bbox_area(nested_bbox)
    return form_area > 0 and nested_area < 0.5 * form_area and _bbox_overlap_in_first(nested_bbox, form_bbox) >= 0.9


def _form_member_bbox(line: _LineItem, form_bbox: BBox) -> BBox | None:
    """优先保持既有行框，仅在 Form 边缘用完整的可见字形证据补回成员。"""

    if _bbox_overlap_in_first(line.bbox, form_bbox) >= 0.9:
        return line.bbox
    ink_bbox = line.ink_bbox
    if ink_bbox is None:
        visible_chars = [char for char in line.chars if str(char.get("char", "")).strip()]
        tight = [_coerce_bbox(char.get("tight_bbox")) for char in visible_chars]
        if not tight or any(bbox is None for bbox in tight):
            return None
        ink_bbox = _bbox_union_many([bbox for bbox in tight if bbox is not None])
    return ink_bbox if _bbox_overlap_in_first(ink_bbox, form_bbox) >= 0.99 else None


def _owned_form_member_bbox(source: _PageSource, line: _LineItem, form_bbox: BBox) -> BBox | None:
    """有效结构同时要求字符来源属于该 Form；旧快照和未知结构保持原有空间判断。"""
    members = source.form_member_sources.get(form_bbox)
    if members is not None:
        from .form_roles import form_owns_line

        if not form_owns_line(line, members):
            return None
    return _form_member_bbox(line, form_bbox)


def _form_region_allows_line(source: _PageSource, line: _LineItem, region: BBox) -> bool:
    """图题合并和后续补认领也检查 Form 归属，图外图题仍可按正常空间证据处理。"""
    containing = [
        bbox
        for bbox in source.form_member_sources
        if _bbox_overlap_in_first(bbox, region) >= 0.75 and _form_member_bbox(line, bbox) is not None
    ]
    return not containing or any(_owned_form_member_bbox(source, line, bbox) is not None for bbox in containing)


def _tighten_form_image_bbox(
    source: _PageSource,
    form_bbox: BBox,
) -> BBox:
    """用充分的 Form 内部矢量与文本证据收紧空白容器，证据不足时保留原框。"""

    internal_paths = [
        path_info.bbox
        for path_info in source.path_infos
        if path_info.form_depth > 0
        and _bbox_overlap_in_first(path_info.bbox, form_bbox) >= 0.9
        and (form_bbox not in source.form_path_sources or path_info.source_index in source.form_path_sources[form_bbox])
    ]
    internal_drawing_lines = [
        drawing_line.bbox
        for drawing_line in source.drawing_lines
        if _bbox_overlap_in_first(drawing_line.bbox, form_bbox) >= 0.9
    ]
    # 至少两个嵌套 Path 和四个矢量元素，避免只凭普通边框或少量文本裁剪 Form。
    if len(internal_paths) < 2 or len(internal_paths) + len(internal_drawing_lines) < 4:
        return form_bbox
    internal_text = [bbox for line in source.lines if (bbox := _owned_form_member_bbox(source, line, form_bbox)) is not None]
    internal_text.extend(
        bbox
        for char in source.chars
        if str(char.get("char", "")).strip()
        and (bbox := _coerce_bbox(char.get("tight_bbox"))) is not None
        and _bbox_overlap_in_first(bbox, form_bbox) >= 0.99
        and (
            form_bbox not in source.form_member_sources
            or any(
                index in source.form_member_sources[form_bbox] for index in char.get("source_indices", (char.get("char_idx"),))
            )
        )
    )
    evidence_bbox = _clip_bbox(
        _bbox_union_many(internal_paths + internal_drawing_lines + internal_text),
        source.page_size,
    )
    if evidence_bbox is None:
        return form_bbox
    form_width = max(0.1, form_bbox[2] - form_bbox[0])
    form_height = max(0.1, form_bbox[3] - form_bbox[1])
    evidence_width = evidence_bbox[2] - evidence_bbox[0]
    evidence_height = evidence_bbox[3] - evidence_bbox[1]
    if (
        evidence_width < 0.5 * form_width
        or evidence_height < 0.5 * form_height
        or _bbox_area(evidence_bbox) < 0.25 * _bbox_area(form_bbox)
    ):
        return form_bbox
    return evidence_bbox


def _select_form_image_bboxes(source: _PageSource) -> list[BBox]:
    """按页面占比、行高和内部视觉证据筛选矢量 Form 图片候选。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0 or not source.form_bboxes:
        return []
    effective_heights = [_line_effective_height(line, line.bbox) for line in source.lines if line.angle == 0]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    output: list[BBox] = []
    for raw_bbox in source.form_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None:
            continue
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = _bbox_area(bbox) / page_area
        # 实际调用树已确认这是保留的 Form；满页真实图形不能被旧面积上限排除。
        retained_form = any(form.bbox == raw_bbox for form in source.form_infos)
        if not (
            _MIN_FORM_IMAGE_PAGE_AREA_RATIO <= area_ratio
            and (area_ratio <= _MAX_FORM_IMAGE_PAGE_AREA_RATIO or retained_form)
            and width >= 4.0 * median_height
            and height >= 4.0 * median_height
        ):
            continue

        member_rows = {
            line.visual_row_id if line.visual_row_id is not None else line.source_index
            for line in source.lines
            if _owned_form_member_bbox(source, line, bbox) is not None
        }
        internal_drawing_count = sum(
            _bbox_overlap_in_first(drawing_line.bbox, bbox) >= 0.9 for drawing_line in source.drawing_lines
        )
        if len(member_rows) < 2 and internal_drawing_count < 4:
            continue
        tightened = _tighten_form_image_bbox(source, bbox)
        if raw_bbox in source.form_member_sources:
            source.form_member_sources[tightened] = source.form_member_sources[raw_bbox]
        if raw_bbox in source.form_path_sources:
            source.form_path_sources[tightened] = source.form_path_sources[raw_bbox]
        output.append(tightened)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _build_form_image_blocks(
    source: _PageSource,
    form_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """把 Form 及其完整内含文本输出为 image，并保持 source_index 唯一认领。"""

    if not form_bboxes:
        return [], set()
    members_by_candidate: list[list[_LineItem]] = [[] for _ in form_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        matching_indices = [
            candidate_index
            for candidate_index, bbox in enumerate(form_bboxes)
            if _owned_form_member_bbox(source, line, bbox) is not None
        ]
        if not matching_indices:
            continue
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(form_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks = [
        {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        for bbox, members in zip(form_bboxes, members_by_candidate, strict=True)
    ]
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _build_graphic_like_blocks(
    source: _PageSource,
    table_bboxes: list[BBox],
    claimed_line_indices: set[int],
    strong_core_bboxes: list[BBox] | None = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """在表格认领后把紧凑绘图组件及其短标签聚成内部图形文本块。"""

    from .isolated_graphics import isolated_vector_components

    isolated = isolated_vector_components(source)
    isolated.extend(_detect_raster_connected_diagrams(source, _native_graphic_em(source, 10)))

    lines = [line for line in source.lines if line.source_index not in claimed_line_indices]
    if strong_core_bboxes is None:
        strong_core_bboxes = _detect_strong_graphic_bboxes(source)
    if len(lines) < 2:
        return (
            [
                {"type": "image", "content": "", "bbox": box, "angle": 0}
                for box in isolated
                if not any(_bbox_overlap_in_smaller(box, table) >= 0.5 for table in table_bboxes)
            ],
            set(),
        )
    if len(source.drawing_lines) < 4 and not strong_core_bboxes:
        return [], set()

    effective_heights = [
        max(
            0.1,
            line.effective_height
            or min(
                max(0.1, line.bbox[2] - line.bbox[0]),
                max(0.1, line.bbox[3] - line.bbox[1]),
            ),
        )
        for line in lines
    ]
    median_height = statistics.median(effective_heights)
    median_height = _native_graphic_em(source, median_height)
    # 已有复杂填充容器可容许外框角落擦过长正文行，但不认领该正文或近旁标签。
    isolated.extend(
        box
        for box in _detect_complex_path_containers(source.path_infos, source.page_size, median_height)
        if not any(_bbox_overlap_in_first(line.bbox, box) > 0.05 for line in source.lines)
        and not any(_bbox_overlap_in_smaller(box, raster) > 0.2 for raster in source.image_bboxes)
    )
    lanes = _infer_graphic_text_lanes(lines, source.page_size, median_height)
    line_candidates = _detect_graphic_candidates(
        source.drawing_lines,
        source.page_size,
        median_height,
        lanes,
        table_bboxes,
        component_summaries=_drawing_component_summaries(source, max(2.0, 0.75 * median_height)),
    )
    # Plain frames around text do not make that text an image.
    line_candidates = [candidate for candidate in line_candidates
                       if not _is_plain_text_frame(source, candidate.core_bbox, median_height)]
    # 复杂 Path 或成对坐标轴形成的强图形核心优先于普通绘图线组件，
    # 避免同一图表被拆成多个相互重叠的 image。
    raster_axis_cores = _detect_native_raster_axis_graphics(source)
    candidates = [
        candidate
        for candidate in line_candidates
        if not any(_bbox_overlap_in_smaller(candidate.core_bbox, core_bbox) >= 0.5 for core_bbox in strong_core_bboxes)
    ]
    candidates.extend(
        _GraphicCandidate(
            core_bbox=core_bbox,
            lane_index=_strong_graphic_lane_index(
                core_bbox,
                lanes,
                median_height,
            ),
            label_margin_scale=(
                2.5
                if any(
                    _bbox_overlap_in_smaller(candidate.core_bbox, core_bbox) >= 0.5
                    and _bbox_area(candidate.core_bbox) >= 0.8 * _bbox_area(core_bbox)
                    for candidate in line_candidates
                )
                else 1.0
            ),
            strict_core_members=any(
                _bbox_overlap_in_first(core_bbox, axis) >= 0.98 and _bbox_overlap_in_first(axis, core_bbox) >= 0.98
                for axis in raster_axis_cores
            ),
        )
        for core_bbox in strong_core_bboxes
        if not any(_bbox_overlap_in_smaller(core_bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes)
    )
    if not candidates:
        return [], set()

    row_groups: dict[tuple[int, int, int, int], list[_LineItem]] = {}
    for line in lines:
        lane_index = _graphic_lane_index(line.bbox, lanes)
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault(
            (line.angle, row_kind, row_identity, lane_index),
            [],
        ).append(line)

    protected_caption_indices = _graphic_caption_line_indices_to_preserve(
        lines,
        candidates,
        median_height,
    )
    protected_body_tail_indices = _graphic_body_tail_line_indices_to_preserve(
        lines,
        candidates,
        lanes,
        median_height,
    )

    for row_lines in row_groups.values():
        if any(line.source_index in protected_caption_indices | protected_body_tail_indices for line in row_lines):
            continue
        row_lane_index = _graphic_lane_index(row_lines[0].bbox, lanes)
        matches: list[tuple[int, float, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            if candidate.lane_index >= 0 and candidate.lane_index != row_lane_index:
                continue
            member_flags = [
                _bbox_overlap_in_first(line.bbox, candidate.core_bbox) >= 0.95
                if candidate.strict_core_members
                else _is_graphic_label_member(
                    line,
                    candidate.core_bbox,
                    median_height,
                    margin_scale=candidate.label_margin_scale,
                )
                for line in row_lines
            ]
            # 同一 pdftext 视觉行必须整体归属或整体保留，避免只吞掉 caption 的短碎片。
            if not all(member_flags):
                continue
            inside_count = sum(
                _point_in_bbox(
                    (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                    candidate.core_bbox,
                )
                for line in row_lines
            )
            mean_distance = statistics.fmean(_bbox_distance(line.bbox, candidate.core_bbox) for line in row_lines)
            matches.append((-inside_count, mean_distance, candidate_index))
        if not matches:
            continue
        candidate_index = min(matches)[2]
        candidates[candidate_index].line_indices.update(line.source_index for line in row_lines)

    blocks: list[dict[str, Any]] = []
    claimed: set[int] = set()
    lines_by_index = {line.source_index: line for line in lines}
    for candidate in candidates:
        if any(
            _bbox_overlap_in_first(candidate.core_bbox, box) >= 0.95
            and _bbox_overlap_in_first(box, candidate.core_bbox) >= 0.95
            for box in isolated
        ):
            blocks.append({"type": "image", "content": "", "bbox": candidate.core_bbox, "angle": 0})
            continue
        members = [
            lines_by_index[source_index] for source_index in sorted(candidate.line_indices) if source_index in lines_by_index
        ]
        if len(members) < 2:
            continue
        block = _graphic_members_to_block(candidate, members, source.page_size)
        if block is None:
            continue
        block["_native_numeric_grid"] = _graphic_members_are_numeric_grid(members)
        blocks.append(block)
        claimed.update(line.source_index for line in members)

    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _parallel_graphic_rule_pairs(
    drawing_lines: list[_AxisLine],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    median_height: float,
) -> list[tuple[BBox, BBox]]:
    """筛选分别贴近两个并排图形上沿的同高长横线。"""

    minimum_rule_width = max(8.0 * median_height, 0.18 * page_size[0])
    long_rules = [
        line.bbox
        for line in drawing_lines
        if line.orientation == "horizontal"
        and line.bbox[2] - line.bbox[0] >= minimum_rule_width
        and not any(
            _point_in_bbox(
                (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)),
                table_bbox,
            )
            for table_bbox in table_bboxes
        )
    ]
    ordered_images = sorted(image_bboxes, key=lambda bbox: (bbox[0], bbox[1]))
    rule_pairs: list[tuple[BBox, BBox]] = []
    seen_pairs: set[tuple[BBox, BBox]] = set()
    for left_index, left_image in enumerate(ordered_images):
        left_height = max(0.1, left_image[3] - left_image[1])
        for right_image in ordered_images[left_index + 1 :]:
            if left_image[2] >= right_image[0]:
                continue
            right_height = max(0.1, right_image[3] - right_image[1])
            image_overlap = max(
                0.0,
                min(left_image[3], right_image[3]) - max(left_image[1], right_image[1]),
            )
            if image_overlap < 0.7 * min(left_height, right_height):
                continue
            if any(
                _bbox_overlap_in_smaller(image_bbox, table_bbox) >= 0.5
                for image_bbox in (left_image, right_image)
                for table_bbox in table_bboxes
            ):
                continue

            left_rules = [
                rule_bbox
                for rule_bbox in long_rules
                if _bbox_axis_overlap_ratio(rule_bbox, left_image, axis="x") >= 0.8
                and -0.25 * median_height <= left_image[1] - rule_bbox[3] <= 3.0 * median_height
            ]
            right_rules = [
                rule_bbox
                for rule_bbox in long_rules
                if _bbox_axis_overlap_ratio(rule_bbox, right_image, axis="x") >= 0.8
                and -0.25 * median_height <= right_image[1] - rule_bbox[3] <= 3.0 * median_height
            ]
            for left_rule in left_rules:
                for right_rule in right_rules:
                    if left_rule[2] >= right_rule[0]:
                        continue
                    if abs(_bbox_center_y(left_rule) - _bbox_center_y(right_rule)) > 0.5 * median_height:
                        continue
                    rule_gap = right_rule[0] - left_rule[2]
                    if not 0.5 * median_height <= rule_gap <= 5.0 * median_height:
                        continue
                    pair = (left_rule, right_rule)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        rule_pairs.append(pair)
    return rule_pairs


def _parallel_graphic_row_split_boundary(
    members: list[_LineItem],
    left_rule: BBox,
    right_rule: BBox,
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    median_height: float,
) -> float | None:
    """用横线栏沟和字符投影确认并排图形上方文本的安全切分点。"""

    if not members or any(member.angle != 0 for member in members) or len({member.semantic_type for member in members}) != 1:
        return None
    row_bbox = _bbox_union_many([member.bbox for member in members])
    if any(_bbox_intersects(row_bbox, table_bbox) for table_bbox in table_bboxes):
        return None
    rule_top = min(left_rule[1], right_rule[1])
    if not -0.2 * median_height <= rule_top - row_bbox[3] <= 1.5 * median_height:
        return None

    glyph_bboxes = [
        bbox
        for member in members
        for char in member.chars
        if str(char.get("char") or "").isprintable()
        and not str(char.get("char") or "").isspace()
        and (bbox := _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)) is not None
    ]
    if not glyph_bboxes:
        return None
    boundary = 0.5 * (left_rule[2] + right_rule[0])
    if any(
        _bbox_center_x(bbox) < left_rule[0] - median_height or _bbox_center_x(bbox) > right_rule[2] + median_height
        for bbox in glyph_bboxes
    ):
        return None
    left_glyphs = [bbox for bbox in glyph_bboxes if _bbox_center_x(bbox) < boundary]
    right_glyphs = [bbox for bbox in glyph_bboxes if _bbox_center_x(bbox) > boundary]
    if len(left_glyphs) < 3 or len(right_glyphs) < 3:
        return None
    left_width = max(bbox[2] for bbox in left_glyphs) - min(bbox[0] for bbox in left_glyphs)
    right_width = max(bbox[2] for bbox in right_glyphs) - min(bbox[0] for bbox in right_glyphs)
    if min(left_width, right_width) < 4.0 * median_height:
        return None
    left_edge = max(bbox[2] for bbox in left_glyphs)
    right_edge = min(bbox[0] for bbox in right_glyphs)
    if right_edge - left_edge < 0.75 * median_height:
        return None
    if not (left_edge <= left_rule[2] <= right_edge and left_edge <= right_rule[0] <= right_edge):
        return None
    return boundary


def _split_parallel_graphic_rule_rows(
    lines: list[_LineItem],
    drawing_lines: list[_AxisLine],
    image_bboxes: list[BBox],
    table_bboxes: list[BBox],
    page_size: tuple[float, float],
    *,
    source_index_start: int | None = None,
) -> list[_LineItem]:
    """按成对图形、独立顶边横线和栏沟字符投影拆分并排图形上方文本。"""

    horizontal_lines = [line for line in lines if line.angle == 0 and line.effective_height > 0]
    if len(horizontal_lines) < 1 or len(image_bboxes) < 2:
        return list(lines)
    median_height = statistics.median(line.effective_height for line in horizontal_lines)
    rule_pairs = _parallel_graphic_rule_pairs(
        drawing_lines,
        image_bboxes,
        table_bboxes,
        page_size,
        median_height,
    )
    if not rule_pairs:
        return list(lines)

    row_groups: dict[tuple[int, int], list[_LineItem]] = {}
    for line in horizontal_lines:
        if line.visual_row_id is not None:
            row_groups.setdefault((line.angle, line.visual_row_id), []).append(line)
    boundaries_by_row: dict[tuple[int, int], list[float]] = {}
    for row_key, members in row_groups.items():
        for left_rule, right_rule in rule_pairs:
            boundary = _parallel_graphic_row_split_boundary(
                members,
                left_rule,
                right_rule,
                table_bboxes,
                page_size,
                median_height,
            )
            if boundary is None:
                continue
            row_boundaries = boundaries_by_row.setdefault(row_key, [])
            if not any(abs(boundary - existing) <= 0.5 * median_height for existing in row_boundaries):
                row_boundaries.append(boundary)
    if not boundaries_by_row:
        return list(lines)

    next_source_index = max(
        max((line.source_index for line in lines), default=-1) + 1,
        source_index_start or 0,
    )
    consumed_source_indices: set[int] = set()
    split_lines: list[_LineItem] = []
    for row_key, boundaries in boundaries_by_row.items():
        members = sorted(
            row_groups[row_key],
            key=lambda line: (line.bbox[0], line.run_index, line.source_index),
        )
        ordered_chars = [char for member in members for char in member.chars]
        split_indices: list[int] = []
        for boundary in sorted(boundaries):
            split_index = next(
                (
                    index
                    for index, char in enumerate(ordered_chars)
                    if str(char.get("char") or "").isprintable()
                    and not str(char.get("char") or "").isspace()
                    and (
                        bbox := _clip_bbox(
                            _coerce_bbox(char.get("bbox")),
                            page_size,
                        )
                    )
                    is not None
                    and _bbox_center_x(bbox) > boundary
                ),
                None,
            )
            if split_index is not None and split_index not in split_indices:
                split_indices.append(split_index)
        if not split_indices:
            continue
        ranges: list[tuple[int, int]] = []
        start = 0
        for split_index in sorted(split_indices):
            ranges.append((start, split_index))
            start = split_index
        ranges.append((start, len(ordered_chars)))

        source_indices = [member.source_index for member in members]
        rebuilt: list[_LineItem] = []
        for run_index, (start, end) in enumerate(ranges):
            run_chars = ordered_chars[start:end]
            run_bboxes = [
                bbox
                for char in run_chars
                if str(char.get("char") or "").isprintable()
                and not str(char.get("char") or "").isspace()
                and (
                    bbox := _clip_bbox(
                        _coerce_bbox(char.get("bbox")),
                        page_size,
                    )
                )
                is not None
            ]
            run_text = _normalize_native_run_text("".join(str(char.get("char") or "") for char in run_chars))
            if not run_text or not run_bboxes:
                continue
            if run_index < len(source_indices):
                source_index = source_indices[run_index]
            else:
                source_index = next_source_index
                next_source_index += 1
            template = members[min(run_index, len(members) - 1)]
            rebuilt_line = replace(
                template,
                text=run_text,
                bbox=_bbox_union_many(run_bboxes),
                source_index=source_index,
                chars=list(run_chars),
                visual_row_id=row_key[1],
                run_index=run_index,
                split_from_row=True,
                preserve_split_boundary=True,
            )
            _fill_native_typography(rebuilt_line, page_size)
            rebuilt.append(rebuilt_line)
        if len(rebuilt) < 2:
            continue
        consumed_source_indices.update(member.source_index for member in members)
        split_lines.extend(rebuilt)

    output = [line for line in lines if line.source_index not in consumed_source_indices]
    output.extend(split_lines)
    output.sort(key=lambda line: (line.angle, line.bbox[1], line.bbox[0], line.source_index))
    return output


def _graphic_caption_line_indices_to_preserve(
    lines: list[_LineItem],
    candidates: list[_GraphicCandidate],
    median_height: float,
) -> set[int]:
    """保护图形上下沿的图注及其同字体续行，避免图题末行被图片容器认领。"""

    protected: set[int] = set()
    ordered_lines = sorted(
        (line for line in lines if line.angle == 0),
        key=lambda line: (line.bbox[1], line.bbox[0], line.source_index),
    )
    for seed_index, seed in enumerate(ordered_lines):
        if not _FIGURE_CAPTION_LINE_RE.match(seed.text):
            continue
        # 多行图题应以最后一行计算到图体的距离；首行较远不能让末行落入引线标签区。
        wrapped = [seed]
        if seed.font_signature is not None:
            for tail in ordered_lines[seed_index + 1 :]:
                if tail.bbox[1] - wrapped[-1].bbox[3] > 0.75 * median_height:
                    break
                if (
                    not _font_signatures_share_family(tail.font_signature, seed.font_signature)
                    or abs((tail.dominant_font_weight or 400) - (seed.dominant_font_weight or 400)) >= 100
                    or abs(tail.bbox[0] - seed.bbox[0]) > 0.3 * median_height
                ):
                    continue
                if _bbox_center_y(tail.bbox) <= _bbox_center_y(wrapped[-1].bbox):
                    continue
                if (
                    len(re.findall(r"\b[A-Za-z]{2,}\b", tail.text)) < 2
                    or _FIGURE_CAPTION_LINE_RE.match(tail.text)
                    or tail.bbox[2] - tail.bbox[0] > 1.5 * (seed.bbox[2] - seed.bbox[0])
                ):
                    break
                wrapped.append(tail)
        wrapped_bounds = _bbox_union_many([line.bbox for line in wrapped])
        wrapped_parents = [
            candidate
            for candidate in candidates
            if len(wrapped) >= 2
            and _is_strong_caption_text(seed.text)
            and _bbox_axis_overlap_ratio(wrapped_bounds, candidate.core_bbox, axis="x") >= 0.35
            and 0 <= candidate.core_bbox[1] - wrapped_bounds[3] <= 4 * median_height
            and seed.bbox[3] <= candidate.core_bbox[1]
        ]
        if wrapped_parents:
            caption_group = max((line.paragraph_group for line in lines if line.paragraph_group is not None), default=-1) + 1
            for line in wrapped:
                line.semantic_type = "caption"
                line.title_suppressed = True
                line.paragraph_group = caption_group
                protected.add(line.source_index)
        matching_candidates = [
            candidate
            for candidate in candidates
            if _bbox_axis_overlap_ratio(
                seed.bbox,
                candidate.core_bbox,
                axis="x",
            )
            >= 0.35
            and (
                candidate.core_bbox[3] - 2.5 * median_height
                <= _bbox_center_y(seed.bbox)
                <= candidate.core_bbox[3] + 2.5 * median_height
                or -0.25 * median_height <= candidate.core_bbox[1] - seed.bbox[3] <= 4 * median_height
            )
        ]
        if not matching_candidates:
            continue
        protected.add(seed.source_index)
        previous = seed
        for candidate_line in ordered_lines[seed_index + 1 :]:
            if candidate_line.bbox[1] - previous.bbox[3] > 0.75 * median_height:
                break
            if _bbox_center_y(candidate_line.bbox) <= _bbox_center_y(previous.bbox):
                continue
            # 上方图题续行允许缩进，但必须留在种子行横向范围内，并排除紧邻的轴值和新标签。
            above = any(seed.bbox[3] <= candidate.core_bbox[1] + 0.25 * median_height for candidate in matching_candidates)
            inset_tail = (
                above
                and candidate_line.bbox[0] >= seed.bbox[0] - 0.25 * median_height
                and candidate_line.bbox[2] <= seed.bbox[2] + 0.25 * median_height
                and not re.fullmatch(r"\s*[+−-]?\d+(?:[,.]\d+)?%?\s*", candidate_line.text)
                and not _FIGURE_CAPTION_LINE_RE.match(candidate_line.text)
                and not re.match(r"^\s*(?:sources?|notes?)\s*:", candidate_line.text, re.I)
                and candidate_line.bbox[3] <= min(candidate.core_bbox[1] for candidate in matching_candidates)
            )
            if above and not inset_tail:
                break
            if (not inset_tail and abs(candidate_line.bbox[0] - seed.bbox[0]) > median_height) or (
                seed.font_signature is not None
                and candidate_line.font_signature is not None
                and seed.font_signature != candidate_line.font_signature
            ):
                continue
            protected.add(candidate_line.source_index)
            previous = candidate_line
            if candidate_line.text.rstrip().endswith((".", "!", "?")):
                break
    return protected


def _graphic_body_tail_line_indices_to_preserve(
    lines: list[_LineItem],
    candidates: list[_GraphicCandidate],
    lanes: list[_TextLane],
    median_height: float,
) -> set[int]:
    """保护贴近图形上沿但延续上方满栏正文排版的短尾行。"""

    protected: set[int] = set()
    horizontal_lines = [line for line in lines if line.angle == 0]
    for tail in horizontal_lines:
        lane_index = _graphic_lane_index(tail.bbox, lanes)
        lane = lanes[lane_index]
        lane_width = max(0.1, lane.right - lane.left)
        tail_width = tail.bbox[2] - tail.bbox[0]
        if tail_width > 0.5 * lane_width:
            continue

        matching_candidates = [
            candidate
            for candidate in candidates
            if (candidate.lane_index < 0 or candidate.lane_index == lane_index)
            and tail.bbox[3] <= candidate.core_bbox[1] + 0.25 * median_height
            and _is_graphic_label_member(
                tail,
                candidate.core_bbox,
                median_height,
                margin_scale=candidate.label_margin_scale,
            )
        ]
        if not matching_candidates:
            continue

        tail_height = _line_effective_height(tail, tail.bbox)
        for previous in horizontal_lines:
            if previous.source_index == tail.source_index:
                continue
            if _graphic_lane_index(previous.bbox, lanes) != lane_index:
                continue
            vertical_gap = tail.bbox[1] - previous.bbox[3]
            if not -0.25 * median_height <= vertical_gap <= 0.75 * median_height:
                continue
            if abs(previous.bbox[0] - tail.bbox[0]) > 0.75 * median_height:
                continue

            previous_width = previous.bbox[2] - previous.bbox[0]
            if (
                previous_width < 0.75 * lane_width
                or lane.right - previous.bbox[2] > median_height
                or previous_width < 1.5 * tail_width
            ):
                continue
            previous_height = _line_effective_height(previous, previous.bbox)
            if max(previous_height, tail_height) > 1.25 * min(previous_height, tail_height):
                continue
            if (
                previous.font_signature is not None
                and tail.font_signature is not None
                and previous.font_signature != tail.font_signature
            ):
                continue
            if any(
                _is_graphic_label_member(
                    previous,
                    candidate.core_bbox,
                    median_height,
                    margin_scale=candidate.label_margin_scale,
                )
                for candidate in matching_candidates
            ):
                continue
            protected.add(tail.source_index)
            break
    return protected


def _detect_strong_graphic_bboxes(source: _PageSource) -> list[BBox]:
    """仅按复杂 Path、容器尺度与成对坐标轴识别高置信图形核心。"""

    from .isolated_graphics import isolated_vector_components

    if not source.path_infos and not source.image_bboxes:
        return []
    effective_heights = [_line_effective_height(line, line.bbox) for line in source.lines if line.angle == 0]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0
    median_height = _native_graphic_em(source, median_height)
    candidates = [
        *isolated_vector_components(source),
        *_detect_raster_connected_diagrams(source, median_height),
        *_detect_native_diagram_components(source, median_height),
        *_detect_stacked_column_graphics(source, median_height),
        *_detect_native_raster_axis_graphics(source),
        *_detect_native_bar_graphics(source, median_height),
        *_detect_captioned_concentric_path_graphics(source, median_height),
        *_detect_complex_path_containers(
            source.path_infos,
            source.page_size,
            median_height,
        ),
        *_detect_axis_path_graphics(
            source.path_infos,
            source.page_size,
            median_height,
        ),
        *_detect_complex_drawing_components(
            source.drawing_lines,
            source.path_infos,
            source.page_size,
            median_height,
            component_summaries=_drawing_component_summaries(source, max(2.0, 0.75 * median_height)),
        ),
    ]

    output: list[BBox] = []
    from .table_rules import _native_multicolumn_grid_bboxes

    for bbox in sorted(candidates, key=_bbox_area, reverse=True):
        # 各类图形候选共享网格仲裁，不能先认领表格底色再排除表格检测。
        if _native_multicolumn_grid_bboxes(source, median_height, bbox):
            continue
        if any(_bbox_overlap_in_first(bbox, accepted) >= 0.9 for accepted in output):
            continue
        # 坐标轴候选与多系列柱核心可能只部分包含彼此；重叠的同一图体只输出一份。
        overlapping = [accepted for accepted in output if _bbox_overlap_in_smaller(bbox, accepted) >= 0.75]
        if overlapping:
            output = [accepted for accepted in output if accepted not in overlapping]
            bbox = _bbox_union_many([bbox, *overlapping])
        output.append(bbox)
    return sorted(output, key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]))


def _is_plain_text_frame(source: _PageSource, bbox: BBox, em: float) -> bool:
    summaries = _drawing_component_summaries(source, max(2.0, .75 * em))
    simple = any(len(item.lines) == 4 and item.horizontal_count == 2 and item.vertical_count == 2
                 and _bbox_overlap_in_first(bbox, item.bbox) >= .98
                 and _bbox_overlap_in_first(item.bbox, bbox) >= .98 for item in summaries)
    if not simple:
        return False
    if any(_bbox_overlap_in_first(image, bbox) >= .9 for image in source.image_bboxes):
        return False
    if any(path.segment_count > 8 and _bbox_overlap_in_first(path.bbox, bbox) >= .9 for path in source.path_infos):
        return False
    return any(_bbox_overlap_in_first(line.bbox, bbox) >= .95 for line in source.lines)


def _detect_raster_connected_diagrams(source: _PageSource, em: float) -> list[BBox]:
    """Keep connected boxes/arrows with rasterized labels as one complete image."""
    if len(source.image_bboxes) < 4 or len(source.drawing_lines) < 12:
        return []
    from .table_rules import _native_multicolumn_grid_bboxes

    output = []
    for component in _drawing_component_summaries(source, max(2.0, .75 * em)):
        box = component.bbox
        if (len(component.lines) < 12 or component.horizontal_count < 4 or component.vertical_count < 4
                or box[2] - box[0] < 8 * em or box[3] - box[1] < 5 * em
                or _bbox_area(box) > .65 * source.page_size[0] * source.page_size[1]):
            continue
        images = [image for image in source.image_bboxes if _bbox_overlap_in_first(image, box) >= .95]
        if len(images) < 4 or _native_multicolumn_grid_bboxes(source, em, box):
            continue
        # Independent internal endpoints distinguish connectors/boxes from a
        # regular full-width lattice populated with ordinary table pictures.
        interior = [line for line in component.lines if line.orientation == 'horizontal'
                    and line.bbox[0] > box[0] + em and line.bbox[2] < box[2] - em]
        if len(interior) >= 3:
            output.append(_bbox_union_many([box, *images]))
    return output


def _native_graphic_em(source: _PageSource, fallback: float) -> float:
    """汉字字宽和真实墨迹共同否决异常高的宽松字体框，仅校准图形距离尺度。"""
    chars = [char for char in source.chars if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", "")) and char.get("tight_bbox")]
    if len(chars) < 20:
        return fallback
    width = statistics.median(char["bbox"][2] - char["bbox"][0] for char in chars)
    ink = statistics.median(char["tight_bbox"][3] - char["tight_bbox"][1] for char in chars)
    return width if width > 0 and fallback >= 2.2 * width and 0.5 * width <= ink <= 1.4 * width else fallback


def _detect_native_diagram_components(source: _PageSource, em: float) -> list[BBox]:
    """连接正文区真实路径和相接栅格，复杂图体整体输出；规则表格和长正文行形成屏障。"""
    from .table_filled_grid import _detect_tiled_fill_table_candidates, _detect_filled_grid_table_candidates
    from .isolated_graphics import _isolated_path_groups
    from .table_detection import _detect_table_candidates, has_native_closed_grid
    from .table_filled_grid import has_inflated_native_cjk_metrics

    if not has_inflated_native_cjk_metrics(source.chars):
        return []

    table_bounds = [
        candidate.bbox
        for candidate in [
            *_detect_tiled_fill_table_candidates(source, []),
            *_detect_filled_grid_table_candidates(source.path_infos, source.page_size, []),
        ]
    ]
    table_bounds.extend(
        candidate.core_bbox or candidate.bbox
        for candidate in _detect_table_candidates(source)
        if has_native_closed_grid(source, candidate, allow_two_by_two=True)
    )
    area = source.page_size[0] * source.page_size[1]
    paths = [
        path
        for path in source.path_infos
        if path.form_depth == 0
        and (path.fill_visible or path.stroke_visible)
        and path.bbox[1] >= 0.08 * source.page_size[1]
        and path.bbox[3] <= 0.93 * source.page_size[1]
        and _bbox_area(path.bbox) < 0.5 * area
        and not any(_bbox_overlap_in_first(path.bbox, box) >= 0.9 for box in table_bounds)
        and not (path.segment_count == 13 and max(path.bbox[2] - path.bbox[0], path.bbox[3] - path.bbox[1]) <= 1.1 * em)
    ]
    result = []
    for group in _isolated_path_groups(source, paths, 2 * em):
        box = _bbox_union_many([path.bbox for path in group])
        if (
            len(group) < 3
            or sum(path.segment_count for path in group) < 100
            or min(box[2] - box[0], box[3] - box[1]) < 2 * em
            or not 0.001 <= _bbox_area(box) / area <= 0.55
        ):
            continue
        prose = [
            line
            for line in source.lines
            if len(line.text) >= 25
            and line.bbox[2] - line.bbox[0] > 0.6 * (box[2] - box[0])
            and _bbox_overlap_in_first(line.bbox, box) > 0.2
        ]
        if len(prose) >= 2:
            continue
        # 路径组与图中阴影/中心照片相接时补足整幅，版心装饰底图不参与扩张。
        for _ in range(2):
            photos = [
                image
                for image in source.image_bboxes
                if _bbox_area(image) < 0.15 * area
                and image[1] >= 0.08 * source.page_size[1]
                and image[3] <= 0.93 * source.page_size[1]
                and _bbox_distance(image, box) <= 3 * em
            ]
            nearby = [path.bbox for path in paths if _bbox_distance(path.bbox, box) <= 1.5 * em]
            expanded = _bbox_union_many([box, *photos, *nearby])
            if (
                sum(
                    len(line.text) >= 25
                    and line.bbox[2] - line.bbox[0] > 0.6 * (expanded[2] - expanded[0])
                    and _bbox_overlap_in_first(line.bbox, expanded) > 0.2
                    for line in source.lines
                )
                >= 2
            ):
                break
            box = expanded
        overlaps = [prior for prior in result if _bbox_overlap_in_smaller(box, prior) >= 0.2]
        result = [prior for prior in result if prior not in overlaps]
        result.append(_bbox_union_many([box, *overlaps]))
    # 多个独立图标与同一条彩色箭头轨道共同证明水平时间轴，下方窄列说明属于图体。
    icon_rows = [box for box in result if 2 * em <= box[2] - box[0] <= 5 * em and 2 * em <= box[3] - box[1] <= 5 * em]
    for anchor in icon_rows:
        peers = [box for box in icon_rows if abs(_bbox_center_y(box) - _bbox_center_y(anchor)) <= em]
        if len(peers) < 4:
            continue
        row = _bbox_union_many(peers)
        strips = [
            path.bbox
            for path in paths
            if path.fill_visible
            and path.rectangle_bboxes
            and path.fill_rgba
            and min(path.fill_rgba[:3]) < 230
            and 0 <= path.bbox[1] - row[3] <= 4 * em
            and 0.5 * em <= path.bbox[3] - path.bbox[1] <= 2 * em
            and path.bbox[2] - path.bbox[0] >= 2 * em
        ]
        if not strips or _bbox_union_many(strips)[2] - _bbox_union_many(strips)[0] < 0.8 * (row[2] - row[0]):
            continue
        bounds = _bbox_union_many([row, *strips])
        labels = [
            line.bbox
            for line in source.lines
            if len(line.text) <= 25
            and line.bbox[2] - line.bbox[0] < 0.2 * (bounds[2] - bounds[0])
            and bounds[0] <= _bbox_center_x(line.bbox) <= bounds[2]
            and row[1] <= line.bbox[1] <= bounds[3] + 20 * em
        ]
        result = [box for box in result if box not in peers]
        result.append(_bbox_union_many([bounds, *labels]))
        break
    return result


def _detect_stacked_column_graphics(source: _PageSource, em: float) -> list[BBox]:
    """复合柱系列的共同零轴及外侧数值刻度确认图体，不以柱高上限截断高柱。"""
    from .table_filled_grid import has_inflated_native_cjk_metrics

    if not has_inflated_native_cjk_metrics(source.chars):
        return []
    output = []
    for path in source.path_infos:
        bars = path.rectangle_bboxes
        if not path.fill_visible or len(bars) < 4:
            continue
        if not all(0.25 * em <= box[2] - box[0] <= 2 * em for box in bars):
            continue
        baseline = statistics.median(box[3] for box in bars)
        ordered = sorted(bars, key=lambda box: box[0])
        if (
            sum(abs(box[3] - baseline) <= 0.2 * em for box in bars) < 0.75 * len(bars)
            or not all(b[0] - a[2] >= 0.2 * em for a, b in zip(ordered, ordered[1:]))
            or max(box[3] - box[1] for box in bars) < 3 * em
        ):
            continue
        bounds = _bbox_union_many(bars)
        peers = [
            box
            for other in source.path_infos
            if other.fill_visible
            for box in other.rectangle_bboxes
            if any(
                abs(box[0] - bar[0]) <= 0.1 * em and abs(box[2] - bar[2]) <= 0.1 * em and box[3] <= baseline + 0.2 * em
                for bar in bars
            )
        ]
        bounds = _bbox_union_many([bounds, *peers])
        ticks = [
            line
            for line in source.lines
            if re.fullmatch(r"\d+(?:[.,]\d+)?", line.text.strip())
            and 0 <= bounds[0] - line.bbox[2] <= 4 * em
            and bounds[1] - 2 * em <= _bbox_center_y(line.bbox) <= bounds[3] + em
        ]
        if len(ticks) < 3:
            continue
        axes = [
            rule.bbox
            for rule in source.drawing_lines
            if rule.orientation == "horizontal"
            and _bbox_axis_overlap_ratio(rule.bbox, bounds, axis="x") >= 0.9
            and bounds[1] - 2 * em <= rule.bbox[1] <= bounds[3] + 0.2 * em
        ]
        labels = [
            item.bbox
            for item in source.path_infos
            if item.fill_visible
            and item.segment_count >= 8
            and bounds[0] - 2 * em <= item.bbox[0] <= bounds[2] + em
            and baseline <= item.bbox[1] <= baseline + 4 * em
            and item.bbox[2] - item.bbox[0] <= 3 * em
            and item.bbox[3] - item.bbox[1] <= 4 * em
        ]
        output.append(_bbox_union_many([bounds, *(line.bbox for line in ticks), *axes, *labels]))
    return output


def _raster_axis_tick_groups(numeric):
    """批量分组数值刻度；非有限或特殊数值继续使用相同 Python 参考规则。"""
    from ...._compute_backend import get_native

    native = get_native()
    records = [(line.bbox, line.effective_height, float(line.text.strip().replace("−", "-"))) for line in numeric]
    if native is not None and all(
        type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2**50
        for box, height, value in records
        for v in (*box, height, value)
    ):
        return [(vertical, [numeric[i] for i in ids], em) for vertical, ids, em in native.raster_axis_groups(records)]
    output = []
    for vertical in (True, False):
        groups = []
        for line in sorted(numeric, key=lambda item: item.bbox[2] if vertical else _bbox_center_y(item.bbox)):
            coordinate = line.bbox[2] if vertical else _bbox_center_y(line.bbox)
            if groups:
                prior = groups[-1]
                anchor = statistics.median(item.bbox[2] if vertical else _bbox_center_y(item.bbox) for item in prior)
                em = statistics.median(item.effective_height for item in [*prior, line])
            if groups and abs(coordinate - anchor) <= 0.4 * em:
                groups[-1].append(line)
            else:
                groups.append([line])
        for ticks in groups:
            if len(ticks) < 5:
                continue
            ticks.sort(key=lambda item: _bbox_center_y(item.bbox) if vertical else _bbox_center_x(item.bbox))
            em = statistics.median(item.effective_height for item in ticks)
            values = [float(item.text.strip().replace("−", "-")) for item in ticks]
            steps = [b - a for a, b in zip(values, values[1:])]
            positions = [_bbox_center_y(item.bbox) if vertical else _bbox_center_x(item.bbox) for item in ticks]
            gaps = [b - a for a, b in zip(positions, positions[1:])]
            if (
                abs(steps[0]) < 1e-8
                or any(abs(step - steps[0]) > 0.01 * abs(steps[0]) for step in steps)
                or max(gaps) - min(gaps) > 0.4 * em
                or min(gaps) < em
                or min(item.effective_height for item in ticks) < 0.8 * em
                or max(item.effective_height for item in ticks) > 1.2 * em
            ):
                continue
            output.append((vertical, ticks, em))
    return output


def _detect_native_raster_axis_graphics(source: _PageSource) -> list[BBox]:
    """等差数值刻度与贴邻栅格图共同证明坐标图；完整聚合标签和小图例，独立大标题及远距脚注不扩框。"""
    text_matches = _numeric_graphic_lines(source.lines, 0)
    if text_matches is None:
        numeric = [
            line
            for line in source.lines
            if line.angle == 0
            and line.semantic_type is None
            and line.effective_height > 0
            and re.fullmatch(r"[+−-]?\d+(?:\.\d+)?", line.text.strip())
            and len(line.text.strip()) <= 8
            and line.bbox[2] - line.bbox[0] <= 4 * line.effective_height
        ]
    else:
        numeric = [
            line
            for line in text_matches
            if line.angle == 0
            and line.semantic_type is None
            and line.effective_height > 0
            and line.bbox[2] - line.bbox[0] <= 4 * line.effective_height
        ]
    output = []
    for vertical, ticks, em in _raster_axis_tick_groups(numeric):
        axis = _bbox_union_many([item.bbox for item in ticks])
        matches = []
        for image in source.image_bboxes:
            width, height = image[2] - image[0], image[3] - image[1]
            if width < 8 * em or height < 8 * em:
                continue
            if vertical:
                compatible = (
                    0 <= image[0] - axis[2] <= 4 * em
                    and 0.7 * height <= axis[3] - axis[1] <= 1.25 * height
                    and abs(_bbox_center_y(axis) - _bbox_center_y(image)) <= em
                )
            else:
                compatible = (
                    0 <= axis[1] - image[3] <= 2 * em
                    and 0.7 * width <= axis[2] - axis[0] <= 1.25 * width
                    and abs(_bbox_center_x(axis) - _bbox_center_x(image)) <= em
                )
            if compatible:
                matches.append(image)
        if not matches:
            continue
        image = max(matches, key=_bbox_area)
        # 纵轴图下方可有两层类别和场景标签；横轴图的左侧类别与右侧图例都受刻度字号约束。
        window = (
            (axis[0] - 0.4 * em if vertical else image[0] - 10 * em),
            image[1] - 1.5 * em,
            image[2] + (2 * em if vertical else 11 * em),
            image[3] + (6.5 * em if vertical else 2.3 * em),
        )
        members = [
            line.bbox
            for line in source.lines
            if line.angle == 0
            and not line.caption_start
            and line.semantic_type is None
            and 0.6 * em <= line.effective_height <= 1.8 * em
            and _bbox_overlap_in_first(line.bbox, window) >= 0.95
        ]
        legends = [
            box
            for box in source.image_bboxes
            if not vertical
            and 0 <= box[0] - image[2] <= 4 * em
            and box[2] - box[0] <= 10 * em
            and box[3] - box[1] <= 6 * em
            and _bbox_axis_overlap_ratio(box, image, axis="y") >= 0.8
        ]
        bounds = _bbox_union_many([image, axis, *members, *legends])
        output.append((bounds[0] - 0.2 * em, bounds[1] - 0.2 * em, bounds[2] + 0.2 * em, bounds[3] + 0.2 * em))
    return output


def group_native_raster_chart_descriptions(source: _PageSource) -> None:
    """刻度证明的图上方同式说明续行成组；短尾行须紧贴同栏长行，独立标题、列表和图注不参与。"""
    cores = _detect_native_raster_axis_graphics(source)
    next_group = max((line.paragraph_group for line in source.lines if line.paragraph_group is not None), default=0) + 1
    for core in cores:
        above = sorted(
            [
                line
                for line in source.lines
                if line.angle == 0
                and line.semantic_type is None
                and line.paragraph_group is None
                and not line.caption_start
                and line.font_signature
                and 0 <= core[1] - line.bbox[3] <= 5 * line.effective_height
                and _bbox_axis_overlap_ratio(line.bbox, core, axis="x") >= 0.3
            ],
            key=lambda line: line.bbox[1],
        )
        for first, tail in zip(above, above[1:]):
            em = first.effective_height
            if (
                em <= 0
                or first.font_signature != tail.font_signature
                or not 0.9 * em <= tail.effective_height <= 1.1 * em
                or abs(first.bbox[0] - tail.bbox[0]) > 0.2 * em
                or first.bbox[2] - first.bbox[0] < 10 * em
                or tail.bbox[2] - tail.bbox[0] > 0.6 * (first.bbox[2] - first.bbox[0])
                or not 0 <= tail.bbox[1] - first.bbox[3] <= 0.75 * em
                or first.paragraph_terminal
                or re.match(r"\s*(?:\d+[.)、]|[•▪])", tail.text)
                or not 0 <= core[1] - tail.bbox[3] <= 3 * em
            ):
                continue
            first.paragraph_group = tail.paragraph_group = next_group
            next_group += 1


def _detect_captioned_concentric_path_graphics(source: _PageSource, em: float) -> list[BBox]:
    """同心二维轮廓、百分比标签及邻近编号图题共同确认环图，连接引线也属于完整图体。"""
    captions = [line for line in source.lines if line.angle == 0 and _FIGURE_CAPTION_LINE_RE.match(line.text)]
    if not captions:
        return []
    paths = [
        path
        for path in source.path_infos
        if path.form_depth == 0
        and path.stroke_visible
        and path.segment_count >= 7
        and min(path.bbox[2] - path.bbox[0], path.bbox[3] - path.bbox[1]) >= 4 * em
        and 0.7 <= (path.bbox[2] - path.bbox[0]) / max(0.1, path.bbox[3] - path.bbox[1]) <= 1.3
    ]
    output = []
    for outer in paths:
        bounds = outer.bbox
        diameter = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        if diameter > 0.4 * source.page_size[0] or diameter > 0.4 * source.page_size[1]:
            continue
        peers = [
            path
            for path in paths
            if path is not outer
            and _bbox_overlap_in_smaller(path.bbox, bounds) >= 0.9
            and abs(_bbox_center_x(path.bbox) - _bbox_center_x(bounds)) <= 0.08 * diameter
            and abs(_bbox_center_y(path.bbox) - _bbox_center_y(bounds)) <= 0.08 * diameter
        ]
        if not peers or not any(
            0 <= bounds[1] - caption.bbox[3] <= 10 * em and _bbox_axis_overlap_ratio(caption.bbox, bounds, axis="x") >= 0.5
            for caption in captions
        ):
            continue
        percentages = [
            line
            for line in source.lines
            if re.fullmatch(r"\s*\d+(?:[.,]\d+)?\s*%\s*", line.text) and _bbox_distance(line.bbox, bounds) <= 0.5 * diameter
        ]
        if len(percentages) < 2:
            continue
        leaders = [
            path.bbox
            for path in source.path_infos
            if path.form_depth == 0
            and path.stroke_visible
            and 2 <= path.segment_count <= 5
            and _bbox_overlap_in_smaller(path.bbox, bounds) >= 0.1
            and path.bbox[2] - path.bbox[0] <= 0.7 * diameter
            and path.bbox[3] - path.bbox[1] <= 0.8 * diameter
        ]
        complete = _bbox_union_many([bounds, *(path.bbox for path in peers), *leaders])
        if not any(_bbox_overlap_in_first(complete, prior) >= 0.9 for prior in output):
            output.append(complete)
    return output


def _bar_zero_axis_bboxes(source: _PageSource, bounds: BBox, horizontal: bool, baseline: float, em: float) -> list[BBox]:
    """柱组已确认后补入共同零轴两端，拒绝浮动装饰线和明显超出图宽的跨栏横线。"""
    result = []
    for line in source.drawing_lines:
        box = line.bbox
        start, end = (box[1], box[3]) if horizontal else (box[0], box[2])
        low, high = (bounds[1], bounds[3]) if horizontal else (bounds[0], bounds[2])
        cross_start, cross_end = (box[0], box[2]) if horizontal else (box[1], box[3])
        length = high - low
        if (
            line.orientation == ("vertical" if horizontal else "horizontal")
            and cross_end - cross_start <= 0.25 * em
            and abs((cross_start + cross_end) / 2 - baseline) <= 0.35 * em
            and 0.8 * length <= end - start <= 2 * length
            and max(0, min(end, high) - max(start, low)) >= 0.8 * length
        ):
            # 路径端点沿主轴再包含半个线宽，避免裁去可见线帽。
            padding = max(0.0, line.width) / 2
            result.append(
                (box[0], box[1] - padding, box[2], box[3] + padding)
                if horizontal
                else (box[0] - padding, box[1], box[2] + padding, box[3])
            )
    return result


def _detect_native_bar_graphics(source: _PageSource, em: float) -> list[BBox]:
    """重复矩形的共同基线、不同长度及图外数值提供柱图证据，排除等宽表格底色。"""

    from ._graphic_geometry import compound_baseline_flags
    from .table_filled_grid import _detect_tiled_fill_table_candidates

    baseline_flags = compound_baseline_flags(source, em)
    table_bounds = [candidate.bbox for candidate in _detect_tiled_fill_table_candidates(source, [])]

    def compound_baseline_is_shared(path, horizontal=None):
        """复合路径整体须有稳定零轴，不能只取彩色表格某一行或某一列伪造柱组。"""
        boxes = path.rectangle_bboxes
        if len(boxes) <= 1:
            return True
        flags = baseline_flags.get(id(boxes))
        if flags is not None:
            return flags[0] or flags[1] if horizontal is None else flags[0] if horizontal else flags[1]
        return any(
            sum(min(abs(box[edge] - baseline) for edge in edges) <= 0.35 * em for box in boxes) >= 0.75 * len(boxes)
            for edges in (((0, 2), (1, 3)) if horizontal is None else ((0, 2),) if horizontal else ((1, 3),))
            for box in boxes
            for baseline in (box[edges[0]], box[edges[1]])
        )

    paths = [
        replace(path, bbox=box, segment_count=4)
        for path in source.path_infos
        for box in (path.rectangle_bboxes if len(path.rectangle_bboxes) > 1 else (path.bbox,))
        if path.fill_visible
        and compound_baseline_is_shared(path)
        and (len(path.rectangle_bboxes) > 1 or 4 <= path.segment_count <= 24)
        and path.fill_rgba is not None
        and min(path.fill_rgba[:3]) < 230
        and max(box[2] - box[0], box[3] - box[1]) >= 2 * em
        and not any(_bbox_overlap_in_first(box, table) >= 0.95 for table in table_bounds)
    ]
    outputs = []
    numeric_text = None
    numeric_text_prepared = False
    for horizontal in (True, False):
        bars = [
            path
            for path in paths
            if compound_baseline_is_shared(path, horizontal)
            and 0.2 * em <= (path.bbox[3] - path.bbox[1] if horizontal else path.bbox[2] - path.bbox[0]) <= 5 * em
            and (path.bbox[2] - path.bbox[0] if horizontal else path.bbox[3] - path.bbox[1])
            >= 1.25 * (path.bbox[3] - path.bbox[1] if horizontal else path.bbox[2] - path.bbox[0])
        ]
        pending = set(range(len(bars)))
        while pending:
            seed = min(pending)
            # 正负值以同一零轴为起点；择共同边最多的一侧，避免仅按柱底拆散负柱。
            edges = (0, 2) if horizontal else (3, 1)
            baseline, _ = max(
                ((bars[seed].bbox[edge], edge) for edge in edges),
                key=lambda candidate: sum(
                    min(abs(bars[i].bbox[e] - candidate[0]) for e in edges) <= 0.35 * em for i in pending
                ),
            )
            group = [i for i in pending if min(abs(bars[i].bbox[e] - baseline) for e in edges) <= 0.35 * em]
            pending.difference_update(group)
            group.sort(key=lambda i: bars[i].bbox[1] if horizontal else bars[i].bbox[0])
            clusters = [[]]
            for i in group:
                if clusters[-1]:
                    prior = bars[clusters[-1][-1]].bbox
                    current = bars[i].bbox
                    thickness = current[3] - current[1] if horizontal else current[2] - current[0]
                    if (current[1] - prior[3] if horizontal else current[0] - prior[2]) > max(5 * em, 2 * thickness):
                        clusters.append([])
                clusters[-1].append(i)
            for cluster in clusters:
                members = [bars[i].bbox for i in cluster]
                positions = {round((box[1] if horizontal else box[0]) / em, 1) for box in members}
                lengths = [box[2] - box[0] if horizontal else box[3] - box[1] for box in members]
                if len(positions) < 2 or max(lengths) - min(lengths) < 0.5 * em:
                    continue
                axis_edges = (1, 3) if horizontal else (0, 2)
                intervals = sorted({(box[axis_edges[0]], box[axis_edges[1]]) for box in members})
                if not any(second[0] - first[1] > 0.1 * em for first, second in zip(intervals, intervals[1:])):
                    # 连续相接的色块是单元格或整片底色，柱组须在分类方向具有真实间隔。
                    continue
                bounds = _bbox_union_many(members)
                # 首个合格柱组才准备纯文字匹配，后续组各自重新检查几何。
                if not numeric_text_prepared:
                    numeric_text = _numeric_graphic_lines(source.lines, 1)
                    numeric_text_prepared = True
                numeric = [
                    line
                    for line in (source.lines if numeric_text is None else numeric_text)
                    if (numeric_text is not None or re.fullmatch(r"\s*[+−-]?\d+(?:[,.]\d+)?%?\s*", line.text))
                    and all(_bbox_overlap_in_first(line.bbox, bar) < 0.1 for bar in members)
                    and _bbox_distance(line.bbox, bounds) <= 5 * em
                ]
                if len(numeric) < 2:
                    continue
                if bounds[2] - bounds[0] < 6 * em or bounds[3] - bounds[1] < 3 * em:
                    continue
                bounds = _bbox_union_many([bounds, *_bar_zero_axis_bboxes(source, bounds, horizontal, baseline, em)])
                label_anchor = _bbox_union_many([bounds, *(line.bbox for line in numeric)])
                caption_indices = _graphic_caption_line_indices_to_preserve(
                    source.lines, [_GraphicCandidate(core_bbox=label_anchor, lane_index=-1)], em
                )
                # 同基线堆叠柱的其他色段与已确认图体相交，按真实接触关系补入整根柱。
                for _ in range(2):
                    additions = [path.bbox for path in paths if _bbox_overlap_in_smaller(path.bbox, bounds) >= 0.25]
                    # 某些PDF把斜排国家标签输出成字形轮廓；只补入已确认柱图下沿附近的短小复杂墨迹。
                    outlined_labels = [
                        path.bbox
                        for path in source.path_infos
                        if path.fill_visible
                        and path.segment_count > 24
                        and path.bbox[2] - path.bbox[0] <= 0.4 * (bounds[2] - bounds[0])
                        and path.bbox[3] - path.bbox[1] <= 6 * em
                        and _bbox_axis_overlap_ratio(path.bbox, label_anchor, axis="x") >= 0.5
                        and 0 <= path.bbox[1] - label_anchor[3] <= 2 * em
                        and not any(
                            re.match(r"^\s*(?:sources?|notes?)\s*:", line.text, re.I)
                            and label_anchor[3] <= line.bbox[1] <= path.bbox[3]
                            for line in source.lines
                        )
                    ]
                    # 混合栅格和原生柱体时，以已确认柱图的接触关系补全系列，整页背景不参与。
                    image_parts = [
                        image
                        for image in source.image_bboxes
                        if _bbox_area(image) < 0.4 * source.page_size[0] * source.page_size[1]
                        and (
                            _bbox_overlap_in_smaller(image, bounds) >= 0.8
                            or _bbox_axis_overlap_ratio(image, bounds, axis="x") >= 0.3
                            and _bbox_distance(image, bounds) <= 1.5 * em
                        )
                    ]
                    labels = [
                        line.bbox
                        for line in source.lines
                        if line.semantic_type is None
                        and line.source_index not in caption_indices
                        and not re.match(r"^\s*(?:sources?|notes?|资料来源|数据来源|来源|注)\s*[:：]", line.text, re.I)
                        and len(line.text.split()) <= 8
                        and line.bbox[2] - line.bbox[0] <= 0.45 * source.page_size[0]
                        and (
                            _bbox_axis_overlap_ratio(line.bbox, bounds, axis="y") >= 0.35
                            and _bbox_distance(line.bbox, bounds) <= 1.5 * em
                            or line.angle not in {0, 90, 180, 270}
                            and _bbox_axis_overlap_ratio(line.bbox, bounds, axis="x") >= 0.5
                            and 0 <= line.bbox[1] - bounds[3] <= 4 * em
                        )
                        and _line_effective_height(line, line.bbox) <= 1.25 * em
                        and not _FIGURE_CAPTION_LINE_RE.match(line.text)
                    ]
                    bounds = _bbox_union_many(
                        [bounds, *additions, *outlined_labels, *image_parts, *labels, *(line.bbox for line in numeric)]
                    )
                from .table_rules import _native_multicolumn_grid_bboxes

                if _native_multicolumn_grid_bboxes(source, em, bounds):
                    # 跨越多个有字单元格的底色不能成为柱图；单格内的真实柱图仍保持图形。
                    continue
                outputs.append(bounds)
    return outputs


def _detect_complex_path_containers(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
) -> list[BBox]:
    """筛选包含多个内部 Path 且至少含一个二维复杂轮廓的大容器。"""

    page_area = max(0.1, page_size[0] * page_size[1])
    output: list[BBox] = []
    for path_info in path_infos:
        bbox = path_info.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area_ratio = _bbox_area(bbox) / page_area
        if (
            path_info.form_depth != 0
            or not path_info.fill_visible
            or not 0.005 <= area_ratio <= 0.5
            or width < 4.0 * median_height
            or height < 4.0 * median_height
        ):
            continue
        inner_paths = [
            item
            for item in path_infos
            if item.source_index != path_info.source_index
            and _bbox_overlap_in_first(item.bbox, bbox) >= 0.9
            and _bbox_area(item.bbox) < 0.95 * _bbox_area(bbox)
        ]
        if len(inner_paths) < 4:
            continue
        if not any(_is_two_dimensional_complex_path(item, median_height) for item in inner_paths):
            continue
        output.append(bbox)
    return output


def _detect_axis_path_graphics(
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
) -> list[BBox]:
    """用相交的长横纵轴和内部二维复杂路径补充无外框图表。"""

    thin_limit = max(1.0, 0.5 * median_height)
    minimum_axis_length = 6.0 * median_height
    horizontal_axes = [
        item
        for item in path_infos
        if item.form_depth == 0
        and item.stroke_visible
        and item.bbox[3] - item.bbox[1] <= thin_limit
        and item.bbox[2] - item.bbox[0] >= minimum_axis_length
    ]
    vertical_axes = [
        item
        for item in path_infos
        if item.form_depth == 0
        and item.stroke_visible
        and item.bbox[2] - item.bbox[0] <= thin_limit
        and item.bbox[3] - item.bbox[1] >= minimum_axis_length
    ]
    tolerance = max(2.0, median_height)
    # 复杂路径判定只依赖路径自身与 median_height，提到轴对循环外只算一次。
    complex_candidates = [item for item in path_infos if _is_two_dimensional_complex_path(item, median_height)]
    indexed_vertical_axes = sorted((_bbox_center_x(item.bbox), index) for index, item in enumerate(vertical_axes))
    vertical_centers = [center for center, _index in indexed_vertical_axes]
    output: list[BBox] = []
    for horizontal in horizontal_axes:
        horizontal_y = _bbox_center_y(horizontal.bbox)
        # 命中横轴两个端点附近的纵轴，再恢复原输入顺序以维持候选顺序。
        candidate_indices: set[int] = set()
        for endpoint in (horizontal.bbox[0], horizontal.bbox[2]):
            # 扩一 ULP 只影响候选集合；最终相交判断仍使用原始距离条件。
            start = bisect_left(vertical_centers, math.nextafter(endpoint - tolerance, -math.inf))
            end = bisect_right(vertical_centers, math.nextafter(endpoint + tolerance, math.inf))
            candidate_indices.update(index for _center, index in indexed_vertical_axes[start:end])
        for vertical_index in sorted(candidate_indices):
            vertical = vertical_axes[vertical_index]
            vertical_x = _bbox_center_x(vertical.bbox)
            touches_x = (
                min(
                    abs(vertical_x - horizontal.bbox[0]),
                    abs(vertical_x - horizontal.bbox[2]),
                )
                <= tolerance
            )
            touches_y = (
                min(
                    abs(horizontal_y - vertical.bbox[1]),
                    abs(horizontal_y - vertical.bbox[3]),
                )
                <= tolerance
            )
            if not (touches_x and touches_y):
                continue
            plot_bbox = _bbox_union(horizontal.bbox, vertical.bbox)
            width = plot_bbox[2] - plot_bbox[0]
            height = plot_bbox[3] - plot_bbox[1]
            if width > 0.65 * page_size[0] or height > 0.5 * page_size[1]:
                continue
            complex_paths = [item for item in complex_candidates if _bbox_overlap_in_smaller(item.bbox, plot_bbox) >= 0.2]
            if not complex_paths:
                continue
            output.append(
                _bbox_union(
                    plot_bbox,
                    _bbox_union_many([item.bbox for item in complex_paths]),
                )
            )
    return output


def _is_two_dimensional_complex_path(
    path_info: PDFPathInfo,
    median_height: float,
) -> bool:
    """排除细轴线，只保留横纵均有尺寸且段数较多的图形轮廓。"""

    width = path_info.bbox[2] - path_info.bbox[0]
    height = path_info.bbox[3] - path_info.bbox[1]
    return (
        path_info.form_depth == 0
        and path_info.segment_count >= 6
        and width >= 1.5 * median_height
        and height >= 1.5 * median_height
    )


def _detect_complex_drawing_components(
    drawing_lines: list[_AxisLine],
    path_infos: list[PDFPathInfo],
    page_size: tuple[float, float],
    median_height: float,
    component_summaries: list[_DrawingComponentSummary] | None = None,
) -> list[BBox]:
    """以横纵绘图线组件和内部二维复杂 Path 识别坐标图或嵌入式图表。"""

    tolerance = max(2.0, 0.75 * median_height)
    complex_candidates = [item for item in path_infos if _is_two_dimensional_complex_path(item, median_height)]
    output: list[BBox] = []
    summaries = (
        component_summaries if component_summaries is not None else _summarize_drawing_components(drawing_lines, tolerance)
    )
    for summary in summaries:
        component = summary.lines
        horizontal_count = summary.horizontal_count
        vertical_count = summary.vertical_count
        core_bbox = summary.bbox
        width = core_bbox[2] - core_bbox[0]
        height = core_bbox[3] - core_bbox[1]
        if (
            len(component) < 4
            or horizontal_count < 2
            or vertical_count < 2
            or width < 4.0 * median_height
            or height < 3.0 * median_height
            or width > 0.65 * page_size[0]
            or height > 0.5 * page_size[1]
        ):
            continue
        complex_paths = [
            path_info for path_info in complex_candidates if _bbox_overlap_in_smaller(path_info.bbox, core_bbox) >= 0.2
        ]
        if not complex_paths:
            continue
        output.append(
            _bbox_union(
                core_bbox,
                _bbox_union_many([path_info.bbox for path_info in complex_paths]),
            )
        )
    return output


def _infer_graphic_text_lanes(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    median_height: float,
) -> list[_TextLane]:
    """用横排正文推断页内栏带，供不同角度的图形标签共享栏归属。"""

    line_geometry = [(line, line.bbox) for line in lines if line.angle == 0]
    if not line_geometry:
        return [_TextLane(left=0.0, right=page_size[0])]
    angle_heights = [_line_effective_height(line, bbox) for line, bbox in line_geometry]
    angle_median_height = statistics.median(angle_heights) if angle_heights else median_height
    lanes = [
        lane
        for lane in _infer_text_lanes(
            line_geometry,
            page_size[0],
            angle_median_height,
        )
        if not lane.is_span
    ]
    return lanes or [_TextLane(left=0.0, right=page_size[0])]


def _graphic_lane_index(bbox: BBox, lanes: list[_TextLane]) -> int:
    """按中心点、水平覆盖和距离为 bbox 选择唯一栏带。"""

    center_x = _bbox_center_x(bbox)
    best_index = 0
    best_score = (-1, -1.0, -math.inf)
    for lane_index, lane in enumerate(lanes):
        inside = int(lane.left <= center_x <= lane.right)
        overlap = max(0.0, min(bbox[2], lane.right) - max(bbox[0], lane.left))
        if inside:
            distance = 0.0
        else:
            distance = min(abs(center_x - lane.left), abs(center_x - lane.right))
        score = (inside, overlap, -distance)
        if score > best_score:
            best_score = score
            best_index = lane_index
    return best_index


def _strong_graphic_lane_index(
    core_bbox: BBox,
    lanes: list[_TextLane],
    median_height: float,
) -> int:
    """仅把几乎完整落入唯一栏带的强图形核心绑定到该栏。"""

    core_width = max(0.1, core_bbox[2] - core_bbox[0])
    tolerance = max(1.0, median_height)
    matching_indices = []
    for lane_index, lane in enumerate(lanes):
        overlap = max(
            0.0,
            min(core_bbox[2], lane.right) - max(core_bbox[0], lane.left),
        )
        if overlap / core_width >= 0.9 and core_bbox[0] >= lane.left - tolerance and core_bbox[2] <= lane.right + tolerance:
            matching_indices.append(lane_index)
    return matching_indices[0] if len(matching_indices) == 1 else -1


def _detect_graphic_candidates(
    drawing_lines: list[_AxisLine],
    page_size: tuple[float, float],
    median_height: float,
    lanes: list[_TextLane],
    table_bboxes: list[BBox],
    component_summaries: list[_DrawingComponentSummary] | None = None,
) -> list[_GraphicCandidate]:
    """从非表格绘图线连通分量中筛选尺寸受限的图形容器。"""

    tolerance = max(2.0, 0.75 * median_height)
    candidates: list[_GraphicCandidate] = []
    summaries = (
        component_summaries if component_summaries is not None else _summarize_drawing_components(drawing_lines, tolerance)
    )
    for summary in summaries:
        component = summary.lines
        horizontal_count = summary.horizontal_count
        vertical_count = summary.vertical_count
        core_bbox = summary.bbox
        width = core_bbox[2] - core_bbox[0]
        height = core_bbox[3] - core_bbox[1]
        if (
            len(component) < 4
            or horizontal_count < 2
            or vertical_count < 2
            or width < 4.0 * median_height
            or height < 3.0 * median_height
            or width > 0.5 * page_size[0]
            or height > 0.5 * page_size[1]
        ):
            continue
        if any(_bbox_overlap_in_smaller(core_bbox, table_bbox) >= 0.5 for table_bbox in table_bboxes):
            continue
        candidates.append(
            _GraphicCandidate(
                core_bbox=core_bbox,
                lane_index=_graphic_lane_index(core_bbox, lanes),
            )
        )
    return candidates


def _connected_drawing_line_components(
    drawing_lines: list[_AxisLine],
    tolerance: float,
) -> list[list[_AxisLine]]:
    """按 bbox 间距连接相邻绘图线，并返回互不重叠的连通分量。"""

    parents = list(range(len(drawing_lines)))

    def find(index: int) -> int:
        """查找绘图线连通分量的根节点。"""

        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        """合并两个距离满足条件的绘图线分量。"""

        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parents[second_root] = first_root

    # 扫描线剪枝：按左缘升序排列后，第二条线起点超过第一条线右缘加容差时，
    # 水平净空已超容差，其后所有配对必然不连通，无需再算欧氏距离。
    order = sorted(range(len(drawing_lines)), key=lambda index: drawing_lines[index].bbox[0])
    for position, first_index in enumerate(order):
        first = drawing_lines[first_index].bbox
        limit = first[2] + tolerance
        for second_position in range(position + 1, len(order)):
            second_index = order[second_position]
            second = drawing_lines[second_index].bbox
            if second[0] > limit:
                break
            if _bbox_distance(first, second) <= tolerance:
                union(first_index, second_index)

    components: dict[int, list[_AxisLine]] = {}
    for line_index, line in enumerate(drawing_lines):
        components.setdefault(find(line_index), []).append(line)
    return list(components.values())


def _summarize_drawing_components(drawing_lines: list[_AxisLine], tolerance: float) -> list[_DrawingComponentSummary]:
    """一次计算分量、包围框及横纵线数量，保持原始分量顺序。"""

    summaries = []
    for component in _connected_drawing_line_components(drawing_lines, tolerance):
        horizontal_count = sum(line.orientation == "horizontal" for line in component)
        summaries.append(
            _DrawingComponentSummary(
                lines=component,
                bbox=_bbox_union_many([line.bbox for line in component]),
                horizontal_count=horizontal_count,
                vertical_count=len(component) - horizontal_count,
            )
        )
    return summaries


def _drawing_component_summaries(source: _PageSource, tolerance: float) -> list[_DrawingComponentSummary]:
    """仅在同一页、同一绘图线对象及同一容差下复用分量统计。"""

    for lines, cached_tolerance, summaries in source.drawing_component_cache:
        if lines is source.drawing_lines and cached_tolerance == tolerance:
            return summaries
    summaries = _summarize_drawing_components(source.drawing_lines, tolerance)
    source.drawing_component_cache.append((source.drawing_lines, tolerance, summaries))
    return summaries


def _is_graphic_label_member(
    line: _LineItem,
    core_bbox: BBox,
    median_height: float,
    *,
    margin_scale: float = 2.5,
) -> bool:
    """判断短文本是否位于图形核心内部或对应轴向的邻近标签区。"""

    if re.match(r"^\s*(?:sources?|notes?|资料来源|数据来源|来源|注)\s*[:：]", line.text, re.I):
        return False
    if _bbox_overlap_in_first(line.ink_bbox or line.bbox, core_bbox) < 0.5 and re.match(
        r"^\s*(?:\d+[.、]\s*(?=[\u4e00-\u9fff])|[（(][一二三四五六七八九十]+[）)]|[一二三四五六七八九十]+、)", line.text
    ):
        return False
    center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
    if _point_in_bbox(center, core_bbox):
        return True

    line_height = max(
        0.1,
        line.effective_height
        or min(
            max(0.1, line.bbox[2] - line.bbox[0]),
            max(0.1, line.bbox[3] - line.bbox[1]),
        ),
    )
    if line.angle in {90, 270}:
        primary_length = line.bbox[3] - line.bbox[1]
        core_primary_length = core_bbox[3] - core_bbox[1]
    else:
        primary_length = line.bbox[2] - line.bbox[0]
        core_primary_length = core_bbox[2] - core_bbox[0]

    horizontal_gap = max(core_bbox[0] - line.bbox[2], line.bbox[0] - core_bbox[2], 0.0)
    vertical_gap = max(core_bbox[1] - line.bbox[3], line.bbox[1] - core_bbox[3], 0.0)
    # 横排坐标轴标题允许比刻度标签略长，但必须与图宽、行高和上下间距同时相容。
    is_horizontal_axis_title = (
        line.angle in {0, 180}
        and (
            primary_length <= 8.0 * line_height
            and primary_length <= 0.45 * core_primary_length
            or re.fullmatch(r"\s*(?:\d{4}\s*\([A-Za-z]+\)\s*){2,}\s*", line.text)
            and primary_length <= 1.15 * core_primary_length
        )
        and _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="x") >= 0.15
        and vertical_gap <= 2.5 * median_height
    )
    if is_horizontal_axis_title:
        return True
    if primary_length > min(5.0 * line_height, 0.5 * core_primary_length):
        return False

    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="x") >= 0.15:
        return vertical_gap <= margin_scale * median_height
    if _bbox_axis_overlap_ratio(line.bbox, core_bbox, axis="y") >= 0.15:
        horizontal_limit = max(
            margin_scale * median_height,
            0.2 * (core_bbox[2] - core_bbox[0]),
        )
        return horizontal_gap <= horizontal_limit
    corner_limit = min(margin_scale, 1.5) * median_height
    return (
        horizontal_gap <= corner_limit
        and vertical_gap <= corner_limit
        and math.hypot(horizontal_gap, vertical_gap) <= corner_limit
    )


def _image_members_to_content(
    members: list[_LineItem],
    page_size: tuple[float, float],
    *,
    spatial: bool = False,
) -> str:
    """按视觉行和页内位置生成图片内部文本，保留不同视觉行之间的换行。"""

    row_groups: dict[tuple[int, int, int], list[_LineItem]] = {}
    for line in members:
        if line.visual_row_id is None:
            row_kind, row_identity = 1, line.source_index
        else:
            row_kind, row_identity = 0, line.visual_row_id
        row_groups.setdefault((line.angle, row_kind, row_identity), []).append(line)

    # 原生数值图同时包含旋转日期标签；斜置文本不能按轻微字框差异改变横轴顺序。
    numeric_count = sum(re.fullmatch(r"[\d.,%+/−-]+", line.text.strip()) is not None for line in members)
    numeric_chart = numeric_count >= 6 and numeric_count >= 0.45 * len(members)
    if spatial and numeric_chart and all(line.angle == 0 for line in members):
        from .spatial_text import _SpatialTextItem, _project_spatial_items

        # 整批补认领的数值图保留横向列位置，避免原 PDF 文本对象顺序把图例插入数据。
        return _sanitize_pdf_control_text(
            _project_spatial_items([_SpatialTextItem(line.text, line.bbox) for line in members]),
            preserve_newlines=True,
        ).strip()

    rows: list[tuple[BBox, str]] = []
    rotated_rows: list[tuple[int, BBox, str, int, float]] = []
    for row_lines in row_groups.values():
        row_bbox = _bbox_union_many([line.bbox for line in row_lines])
        angle = row_lines[0].angle
        local_geometry = [(line, _rotate_bbox_to_upright(line.bbox, page_size, angle)) for line in row_lines]
        content = _join_formula_visual_row(local_geometry, page_size)
        if content:
            rows.append((row_bbox, content))
            if numeric_chart and angle in {90, 270}:
                rotated_rows.append((len(rows) - 1, row_bbox, content, angle, row_lines[0].effective_height))
    # 同一轴上的旋转短标签用共同纵向带排序，避免文本长度让年份组越过左侧年份。
    for index, bounds, _, angle, em in rotated_rows:
        peers = [
            (i, box, value)
            for i, box, value, other_angle, other_em in rotated_rows
            if other_angle == angle and min(abs(box[1] - bounds[1]), abs(box[3] - bounds[3])) <= 0.5 * min(em, other_em)
        ]
        if len(peers) >= 3:
            top = min(box[1] for _, box, _ in peers)
            for i, box, value in peers:
                rows[i] = ((box[0], top, box[2], box[3]), value)
    rows.sort(key=lambda item: (item[0][1], item[0][0]))
    return _sanitize_pdf_control_text(
        "\n".join(row_content for _row_bbox, row_content in rows),
        preserve_newlines=True,
    ).strip()


def _graphic_members_to_block(
    candidate: _GraphicCandidate,
    members: list[_LineItem],
    page_size: tuple[float, float],
) -> dict[str, Any] | None:
    """生成含内部文本的矢量图 image block，并合并绘图核心与标签 bbox。"""

    content = _image_members_to_content(members, page_size)
    if not content:
        return None
    return {
        "type": "image",
        "bbox": _bbox_union(candidate.core_bbox, _bbox_union_many([line.bbox for line in members])),
        "angle": 0,
        "content": content,
    }


def _inline_raster_gap_member(
    source: _PageSource,
    left_bbox: BBox,
    right_bbox: BBox,
    claimed_line_indices: set[int],
    median_height: float,
) -> _LineItem | None:
    """查找恰好填充两张同行图片间隙的唯一拆分文本 run。"""

    left_height = max(0.1, left_bbox[3] - left_bbox[1])
    right_height = max(0.1, right_bbox[3] - right_bbox[1])
    vertical_overlap = max(
        0.0,
        min(left_bbox[3], right_bbox[3]) - max(left_bbox[1], right_bbox[1]),
    )
    horizontal_gap = right_bbox[0] - left_bbox[2]
    if (
        max(left_height, right_height) / min(left_height, right_height) > 1.25
        or vertical_overlap / min(left_height, right_height) < 0.8
        or not 0.0 <= horizontal_gap <= 1.5 * median_height
    ):
        return None

    edge_tolerance = max(0.5, 0.25 * median_height)
    band_top = max(left_bbox[1], right_bbox[1])
    band_bottom = min(left_bbox[3], right_bbox[3])
    gap_members = [
        line
        for line in source.lines
        if line.source_index not in claimed_line_indices
        and line.angle == 0
        and line.split_from_row
        and line.visual_row_id is not None
        and left_bbox[2] - edge_tolerance <= _bbox_center_x(line.bbox) <= right_bbox[0] + edge_tolerance
        and band_top - edge_tolerance <= _bbox_center_y(line.bbox) <= band_bottom + edge_tolerance
    ]
    if len(gap_members) != 1:
        return None
    member = gap_members[0]
    if abs(member.bbox[0] - left_bbox[2]) > edge_tolerance or abs(member.bbox[2] - right_bbox[0]) > edge_tolerance:
        return None
    return member


def _inline_raster_group_has_only_expected_text(
    source: _PageSource,
    image_bboxes: list[BBox],
    gap_members: list[_LineItem],
    group_bbox: BBox,
    claimed_line_indices: set[int],
) -> bool:
    """确认复合图片框内没有图片内部文本和间隔符之外的正文。"""

    gap_member_indices = {line.source_index for line in gap_members}
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        if not _point_in_bbox(center, group_bbox):
            continue
        if line.source_index in gap_member_indices:
            continue
        if any(_point_in_bbox(center, image_bbox) for image_bbox in image_bboxes):
            continue
        return False
    return True


def _merge_inline_raster_image_candidates(
    source: _PageSource,
    candidate_bboxes: list[BBox],
    container_bboxes: list[BBox],
    claimed_line_indices: set[int],
) -> list[tuple[BBox, int | None]]:
    """把由同一视觉行间隔符连接的已准入图片合成为单一候选。"""

    if len(candidate_bboxes) < 3:
        return [(bbox, None) for bbox in candidate_bboxes]
    effective_heights = [
        _line_effective_height(line, line.bbox)
        for line in source.lines
        if line.source_index not in claimed_line_indices and line.angle == 0
    ]
    median_height = statistics.median(effective_heights) if effective_heights else 1.0

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(candidate_bboxes))}
    gap_members: dict[tuple[int, int], _LineItem] = {}
    for first_index, first_bbox in enumerate(candidate_bboxes):
        for second_index, second_bbox in enumerate(candidate_bboxes):
            if first_index == second_index or first_bbox[0] >= second_bbox[0]:
                continue
            member = _inline_raster_gap_member(
                source,
                first_bbox,
                second_bbox,
                claimed_line_indices,
                median_height,
            )
            if member is None:
                continue
            adjacency[first_index].add(second_index)
            adjacency[second_index].add(first_index)
            gap_members[(first_index, second_index)] = member

    components: list[list[int]] = []
    visited: set[int] = set()
    for start_index in range(len(candidate_bboxes)):
        if start_index in visited:
            continue
        component: list[int] = []
        pending = [start_index]
        while pending:
            current_index = pending.pop()
            if current_index in visited:
                continue
            visited.add(current_index)
            component.append(current_index)
            pending.extend(adjacency[current_index] - visited)
        components.append(component)

    merged_specs: list[tuple[BBox, int | None]] = []
    consumed_indices: set[int] = set()
    for component in components:
        if len(component) < 3:
            continue
        ordered_indices = sorted(
            component,
            key=lambda index: candidate_bboxes[index][0],
        )
        ordered_pairs = list(zip(ordered_indices, ordered_indices[1:]))
        if not all(pair in gap_members for pair in ordered_pairs):
            continue
        members = [gap_members[pair] for pair in ordered_pairs]
        visual_row_ids = {member.visual_row_id for member in members}
        if len(visual_row_ids) != 1 or None in visual_row_ids:
            continue
        image_bboxes = [candidate_bboxes[index] for index in ordered_indices]
        group_bbox = _bbox_union_many(
            [*image_bboxes, *[member.bbox for member in members]],
        )
        if any(
            _bbox_overlap_in_smaller(group_bbox, container_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if not _inline_raster_group_has_only_expected_text(
            source,
            image_bboxes,
            members,
            group_bbox,
            claimed_line_indices,
        ):
            continue
        merged_specs.append((group_bbox, next(iter(visual_row_ids))))
        consumed_indices.update(ordered_indices)

    merged_specs.extend((bbox, None) for index, bbox in enumerate(candidate_bboxes) if index not in consumed_indices)
    merged_specs.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    return merged_specs


def _image_bboxes_are_near_equal(first: BBox, second: BBox) -> bool:
    """用亚 point 边界容差识别同一图片框，避免签名与点阵来源重复输出。"""

    return all(
        abs(first_value - second_value) <= _SIGNATURE_IMAGE_BBOX_DEDUP_TOLERANCE
        for first_value, second_value in zip(first, second, strict=True)
    )


def _merge_vertical_raster_tiles(
    bboxes: list[BBox],
    page_size: tuple[float, float],
) -> list[BBox]:
    """把同宽且纵向连续的点阵切片合成一张完整图片。"""

    page_width, page_height = page_size
    page_area = max(0.0, page_width) * max(0.0, page_height)
    if len(bboxes) < 3 or page_area <= 0:
        return list(bboxes)

    endpoint_tolerance = max(0.75, 0.002 * page_width)
    endpoint_groups: list[list[tuple[int, BBox]]] = []
    for index, bbox in sorted(
        enumerate(bboxes),
        key=lambda item: (item[1][0], item[1][2], item[1][1]),
    ):
        target = next(
            (
                group
                for group in endpoint_groups
                if abs(bbox[0] - statistics.median(item[1][0] for item in group)) <= endpoint_tolerance
                and abs(bbox[2] - statistics.median(item[1][2] for item in group)) <= endpoint_tolerance
            ),
            None,
        )
        if target is None:
            endpoint_groups.append([(index, bbox)])
        else:
            target.append((index, bbox))

    merged: list[BBox] = []
    consumed: set[int] = set()
    for group in endpoint_groups:
        ordered = sorted(group, key=lambda item: (item[1][1], item[1][3]))
        segments: list[list[tuple[int, BBox]]] = []
        for item in ordered:
            if not segments:
                segments.append([item])
                continue
            previous_bbox = segments[-1][-1][1]
            current_bbox = item[1]
            previous_height = max(0.1, previous_bbox[3] - previous_bbox[1])
            current_height = max(0.1, current_bbox[3] - current_bbox[1])
            maximum_gap = max(
                1.0,
                0.5 * max(previous_height, current_height),
            )
            if current_bbox[1] - previous_bbox[3] <= maximum_gap:
                segments[-1].append(item)
            else:
                segments.append([item])

        for segment in segments:
            if len(segment) < 3:
                continue
            segment_bboxes = [bbox for _index, bbox in segment]
            union_bbox = _bbox_union_many(segment_bboxes)
            union_height = max(0.1, union_bbox[3] - union_bbox[1])
            covered_height = sum(max(0.0, bbox[3] - bbox[1]) for bbox in segment_bboxes)
            if (
                _bbox_area(union_bbox) / page_area < _MIN_RASTER_IMAGE_PAGE_AREA_RATIO
                or union_bbox[2] - union_bbox[0] < 0.12 * page_width
                or union_bbox[2] - union_bbox[0] > 0.9 * page_width
                or covered_height / union_height < 0.9
            ):
                continue
            merged.append(union_bbox)
            consumed.update(index for index, _bbox in segment)

    merged.extend(bbox for index, bbox in enumerate(bboxes) if index not in consumed)
    return sorted(
        merged,
        key=lambda bbox: (bbox[1], bbox[0], bbox[3], bbox[2]),
    )


def _build_raster_image_blocks(
    source: _PageSource,
    container_blocks: list[dict[str, Any]],
    claimed_line_indices: set[int],
) -> tuple[list[dict[str, Any]], set[int]]:
    """过滤点阵图并接纳签名框，避让高优先级容器后唯一认领内部文本。"""

    page_area = max(0.0, source.page_size[0]) * max(0.0, source.page_size[1])
    if page_area <= 0:
        return [], set()

    container_bboxes = [bbox for block in container_blocks if (bbox := _coerce_bbox(block.get("bbox"))) is not None]
    signature_bboxes: list[BBox] = []
    for raw_bbox in source.signature_bboxes:
        bbox = _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)
        if bbox is None:
            continue
        if any(
            _bbox_overlap_in_smaller(bbox, container_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if not any(_image_bboxes_are_near_equal(bbox, existing_bbox) for existing_bbox in signature_bboxes):
            # 已由注释可见性和 /AP 严格确认的签名不再套用普通点阵图面积门槛。
            signature_bboxes.append(bbox)

    clipped_raster_bboxes = [
        bbox for raw_bbox in source.image_bboxes if (bbox := _clip_bbox(_coerce_bbox(raw_bbox), source.page_size)) is not None
    ]
    raster_bboxes: list[BBox] = []
    for bbox in _merge_vertical_raster_tiles(
        _deduplicate_contained_photo_frames(source, clipped_raster_bboxes),
        source.page_size,
    ):
        from .isolated_graphics import meaningful_small_raster

        if _raster_is_native_text_background(source, bbox):
            continue

        if _bbox_area(bbox) / page_area < _MIN_RASTER_IMAGE_PAGE_AREA_RATIO and not meaningful_small_raster(source, bbox):
            continue
        if any(
            _bbox_overlap_in_smaller(bbox, container_bbox) >= _IMAGE_CONTAINER_OVERLAP_THRESHOLD
            for container_bbox in container_bboxes
        ):
            continue
        if any(_image_bboxes_are_near_equal(bbox, signature_bbox) for signature_bbox in signature_bboxes):
            continue
        raster_bboxes.append(bbox)
    if not raster_bboxes and not signature_bboxes:
        return [], set()

    candidate_specs = (
        _merge_inline_raster_image_candidates(
            source,
            raster_bboxes,
            container_bboxes,
            claimed_line_indices,
        )
        if raster_bboxes
        else []
    )
    candidate_specs.extend((bbox, None) for bbox in signature_bboxes)
    candidate_specs.sort(key=lambda item: (item[0][1], item[0][0], item[0][3], item[0][2]))
    candidate_bboxes = [bbox for bbox, _row_id in candidate_specs]

    members_by_candidate: list[list[_LineItem]] = [[] for _ in candidate_bboxes]
    claimed: set[int] = set()
    for line in source.lines:
        if line.source_index in claimed_line_indices:
            continue
        center = (_bbox_center_x(line.bbox), _bbox_center_y(line.bbox))
        matching_indices = [
            candidate_index for candidate_index, bbox in enumerate(candidate_bboxes)
            if _point_in_bbox(center, bbox)
            # Native foreground text on a page background has its own geometry.
            # Keep the background image and the text as independent content.
            and _bbox_area(bbox) < .6 * page_area
        ]
        if not matching_indices:
            continue
        # 重叠点阵图共享内部文本时归属最小容器，避免 content 重复。
        candidate_index = min(
            matching_indices,
            key=lambda index: (_bbox_area(candidate_bboxes[index]), index),
        )
        members_by_candidate[candidate_index].append(line)
        claimed.add(line.source_index)

    blocks: list[dict[str, Any]] = []
    for (bbox, visual_row_id), members in zip(
        candidate_specs,
        members_by_candidate,
        strict=True,
    ):
        block = {
            "type": "image",
            "bbox": bbox,
            "angle": 0,
            "content": _image_members_to_content(members, source.page_size),
        }
        if visual_row_id is not None:
            block["_inline_visual_row_id"] = visual_row_id
        blocks.append(block)
    blocks.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))
    return blocks, claimed


def _raster_is_native_text_background(source: _PageSource, bbox: BBox) -> bool:
    """覆盖版心的大底图前已有密集原生段落或目录时，底图不得认领这些文字。"""
    if _bbox_area(bbox) < 0.6 * source.page_size[0] * source.page_size[1]:
        return False
    members = [line for line in source.lines if line.angle == 0 and _bbox_overlap_in_first(line.bbox, bbox) >= 0.9]
    long = [line for line in members if len(line.text) >= 12]
    return len(long) >= 8 and sum(len(line.text) for line in members) >= 300


def _deduplicate_contained_photo_frames(source, bboxes):
    """近乎等大的照片和阴影边框合为一幅；独立内嵌小图及边框中的原生文字继续保留。"""
    consumed = set()
    for index, outer in enumerate(bboxes):
        width, height = outer[2] - outer[0], outer[3] - outer[1]
        if min(width, height) <= 0:
            continue
        for other, inner in enumerate(bboxes):
            if index == other or _bbox_area(inner) >= _bbox_area(outer):
                continue
            if not (
                _bbox_overlap_in_first(inner, outer) >= 0.999
                and _bbox_area(inner) / _bbox_area(outer) >= 0.85
                and 0 <= inner[0] - outer[0] <= 0.08 * width
                and 0 <= outer[2] - inner[2] <= 0.08 * width
                and 0 <= inner[1] - outer[1] <= 0.08 * height
                and 0 <= outer[3] - inner[3] <= 0.08 * height
            ):
                continue
            if any(
                _point_in_bbox((_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)), outer)
                and not _point_in_bbox((_bbox_center_x(line.bbox), _bbox_center_y(line.bbox)), inner)
                for line in source.lines
            ):
                continue
            consumed.add(other)
    return [bbox for index, bbox in enumerate(bboxes) if index not in consumed]
