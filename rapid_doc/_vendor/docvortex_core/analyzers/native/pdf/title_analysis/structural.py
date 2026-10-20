"""识别编号、排版重置及跨页一致的结构标题。"""

from __future__ import annotations

from ..layout_evidence import build_layout_evidence

import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import replace

from .....schema import BBox
from ..geometry import _bbox_axis_overlap_ratio, _bbox_center_x, _bbox_center_y, _bbox_union_many, _rotate_bbox_to_upright
from ..line_layout import (
    _effective_text_row_gap,
    _estimate_lane_gap,
    _font_signatures_share_family,
    _infer_text_lanes,
    _line_canonical_style_scale,
    _line_effective_height,
    _normalized_font_family,
    _title_fonts_compatible,
)
from ..models import _DocumentBodyProfile, _DocumentTitleProfile, _LineItem, _PreparedPage, _TextLane
from ..native_text import _fill_native_typography
from ..inline.detection import _font_styles_from_metadata
from .body_profile import _line_uses_document_regular_font

from .common import (
    _NUMBERED_SECTION_TITLE_RE,
    _SECTION_NUMBER_ONLY_RE,
    _SECTION_TITLE_TERMINAL_RE,
    _UNNUMBERED_SECTION_HEADING_RE,
    _build_physical_title_gap_map,
    _line_inside_visual_container,
)
from .page_titles import _classify_page_titles


def _line_has_bold_font(line: _LineItem) -> bool:
    """复用样式判定兼容旧 PDFium 的低字重值；只有数字字重或明确粗体元数据才提供证据。"""
    name, flags = line.font_signature or ("", 0)
    return "bold" in _font_styles_from_metadata(name, flags, line.dominant_font_weight)


def _classify_short_cjk_section_leads(lines: list[_LineItem]) -> None:
    """孤立短中文题名及其后两行同左缘正文证明章节起点，段内短尾和冒号引导不晋升标题。"""
    rows = sorted(lines, key=lambda line: (line.bbox[1], line.bbox[0]))
    for candidate in rows:
        text = candidate.text.strip()
        if (
            candidate.semantic_type is not None
            or candidate.paragraph_group is not None
            or candidate.caption_start
            or candidate.formula_candidate_only
            or candidate.angle != 0
            or candidate.font_signature is None
            or re.fullmatch(r"[\u3400-\u9fff]{4,16}", text) is None
        ):
            continue
        following = [
            line
            for line in rows
            if line is not candidate
            and line.semantic_type is None
            and line.paragraph_group is None
            and not line.caption_start
            and line.angle == 0
            and abs(line.bbox[0] - candidate.bbox[0]) <= 0.3 * candidate.effective_height
            and line.bbox[1] >= candidate.bbox[3]
        ]
        if len(following) < 2:
            continue
        first, second = following[:2]
        em = max(first.effective_height, second.effective_height)
        width = first.bbox[2] - first.bbox[0]
        if (
            em <= 0
            or width < 12 * em
            or candidate.bbox[2] - candidate.bbox[0] >= 0.5 * width
            or not 1 <= candidate.effective_height / em <= 1.65
            or not 0.9 * em <= min(first.effective_height, second.effective_height)
            or not _title_fonts_compatible(candidate, first)
            or not _title_fonts_compatible(first, second)
            or not 0 <= first.bbox[1] - candidate.bbox[3] <= 1.1 * em
            or not -0.1 * em <= second.bbox[1] - first.bbox[3] <= 0.65 * em
            or len(re.findall(r"[\u3400-\u9fff]", first.text)) < 12
            or (
                len(re.findall(r"[\u3400-\u9fff]", first.text + second.text)) < 18
                and re.search(r"》\s*[—–-]+\s*\d{4}[-./]\d{1,2}[-./]\d{1,2}\s*$", second.text) is None
            )
        ):
            continue
        preceding = [
            line
            for line in rows
            if line is not candidate
            and line.angle == 0
            and line.semantic_type not in {"paragraph_title", "doc_title"}
            and line.bbox[3] <= candidate.bbox[1]
            and _bbox_axis_overlap_ratio(line.bbox, candidate.bbox, axis="x") >= 0.5
            and 0 <= candidate.bbox[1] - line.bbox[3] < 1.2 * em
        ]
        if preceding:
            continue
        candidate.semantic_type = "paragraph_title"


def _mark_emphasized_quote_prose(lines: list[_LineItem]) -> None:
    """多行同字号强调引用保留为原生正文；开引号或完整闭引号加自然语言反证标题和公式。"""
    from ..inline.types import PDF_FONT_ITALIC_FLAG

    heights = [line.effective_height for line in lines if line.angle == 0 and line.effective_height > 0]
    if not heights:
        return
    body = statistics.median(heights)
    styled = [
        line
        for line in lines
        if line.angle == 0
        and line.semantic_type is None
        and not line.caption_start
        and line.font_signature
        and (line.font_signature[1] & PDF_FONT_ITALIC_FLAG or (line.dominant_font_weight or 400) >= 600)
        and 0.8 <= line.effective_height / body <= 1.2
    ]
    rows = []
    for line in sorted(styled, key=lambda item: (item.bbox[1], item.bbox[0])):
        peers = [
            row
            for row in rows
            if row[0].font_signature == line.font_signature
            and row[0].visual_row_id is not None
            and row[0].visual_row_id == line.visual_row_id
            and min(abs(line.bbox[0] - row[-1].bbox[2]), abs(row[-1].bbox[0] - line.bbox[2])) <= 3 * body
        ]
        if peers:
            peers[-1].append(line)
        else:
            rows.append([line])
    pending = list(rows)
    group = max((line.paragraph_group for line in lines if line.paragraph_group is not None), default=-1) + 1
    while pending:
        run = [pending.pop(0)]
        while True:
            previous = _bbox_union_many([line.bbox for line in run[-1]])
            following = [
                row
                for row in pending
                if row[0].font_signature == run[0][0].font_signature
                and -0.15 * body <= min(line.bbox[1] for line in row) - previous[3] <= 0.8 * body
                and _bbox_axis_overlap_ratio(_bbox_union_many([line.bbox for line in row]), previous, axis="x") >= 0.5
            ]
            if not following:
                break
            row = min(following, key=lambda items: min(line.bbox[1] for line in items))
            pending.remove(row)
            run.append(row)
        text = " ".join(line.text for row in run for line in row).strip()
        italic = bool(run[0][0].font_signature[1] & PDF_FONT_ITALIC_FLAG)
        opening = text.startswith(("“", '"', "‘"))
        closing = bool(re.search(r'[”"]\s*[.!?]?\s*$', text))
        if len(run) < 3 or len(re.findall(r"\b[A-Za-z]{2,}\b", text)) < 15 or re.search(r"[=<>∑∫]", text):
            continue
        if not (italic and opening or len(run) >= 6 and closing):
            continue
        for row in run:
            for line in row:
                line.semantic_type = "text"
                line.title_suppressed = True
                line.paragraph_group = group
        group += 1
        bounds = _bbox_union_many([line.bbox for row in run for line in row])
        # 同栏常规字体的完整冒号引导句仍为正文，不能因下方引用留白被提升为小节标题。
        preceding = sorted(
            [
                line
                for line in lines
                if line.semantic_type is None
                and not line.caption_start
                and line.angle == 0
                and 0.8 <= line.effective_height / body <= 1.2
                and 0 <= bounds[1] - line.bbox[3] <= 4 * body
                and _bbox_axis_overlap_ratio(line.bbox, bounds, axis="x") >= 0.5
            ],
            key=lambda item: item.bbox[1],
        )
        if preceding and preceding[-1].text.rstrip().endswith(":"):
            intro = [preceding[-1]]
            for line in reversed(preceding[:-1]):
                if (
                    line.font_signature != intro[0].font_signature
                    or not -0.1 * body <= intro[0].bbox[1] - line.bbox[3] <= 0.8 * body
                ):
                    break
                intro.insert(0, line)
            if len(re.findall(r"\b[A-Za-z]{2,}\b", " ".join(line.text for line in intro))) >= 6:
                for line in intro:
                    line.semantic_type = "text"
                    line.title_suppressed = True
                    line.paragraph_group = group
                group += 1


def _classify_native_display_resets(lines, page_size, visual_bboxes, table_bboxes, *, page_index, reference_lines):
    """由重复字体、容器净空和多行同级排版恢复标题；不用字体名称或具体文字判定。"""
    available = [
        line
        for line in lines
        if line.angle == 0
        and line.semantic_type in {None, "paragraph_title"}
        and not line.caption_start
        and not line.title_suppressed
        and line.font_signature
    ]
    regular = [
        line
        for line in reference_lines
        if line.angle == 0
        and line.effective_height > 0
        and (len(re.findall(r"\b[A-Za-z]{2,}\b", line.text)) >= 4 or len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 10)
    ]
    if len(regular) < 3:
        return
    body = statistics.median(line.effective_height for line in regular)
    short = [
        line
        for line in available
        if 3 <= len(re.findall(r"[A-Za-z\u3400-\u9fff]", line.text)) <= 70
        and len(line.text.split()) <= 12
        and not re.search(r"[=<>∑∫]|[.!。！;；]$", line.text.strip())
    ]
    # 等高轻字重标题由独立重复字体和上下正文/图像关系共同证明；侧栏图注不能仅凭字体不同晋升。
    light = [
        line
        for line in short
        if 0.85 <= line.effective_height / body <= 1.25
        and line.dominant_font_weight is not None
        and line.dominant_font_weight <= 350
        and not any(other.font_signature == line.font_signature and len(other.text.split()) >= 12 for other in regular)
    ]
    for line in light:
        peers = [
            other
            for other in light
            if other.font_signature == line.font_signature and 0.9 <= other.effective_height / line.effective_height <= 1.1
        ]
        if len(peers) < 2:
            continue
        h = line.effective_height
        followers = [
            other
            for other in regular
            if other.font_signature != line.font_signature
            and abs(other.bbox[0] - line.bbox[0]) <= 0.6 * h
            and 0 <= other.bbox[1] - line.bbox[3] <= 3 * h
            and other.bbox[2] - other.bbox[0] >= line.bbox[2] - line.bbox[0]
        ]
        above_image = any(
            0 <= box[1] - line.bbox[3] <= 2 * h
            and _bbox_axis_overlap_ratio(line.bbox, box, axis="x") >= 0.7
            and (
                abs(_bbox_center_x(line.bbox) - page_size[0] / 2) <= h
                or abs(_bbox_center_x(line.bbox) - _bbox_center_x(box)) <= h
            )
            for box in visual_bboxes
        )
        if len(followers) >= 2 or above_image:
            _set_native_title_band([line], lines)
    # 完整网格上方居中的粗体短题名以表体字号为参照，防止表格内部表头和正文尾句被误收。
    for line in short:
        h = line.effective_height
        if line.semantic_type is not None or not _line_has_bold_font(line) or h < 1.25 * body:
            continue
        if any(
            0.3 * body <= box[1] - line.bbox[3] <= 2 * body
            and 0.2 <= (line.bbox[2] - line.bbox[0]) / max(box[2] - box[0], 0.1) <= 0.7
            and abs(_bbox_center_x(line.bbox) - _bbox_center_x(box)) <= 0.05 * (box[2] - box[0])
            for box in table_bboxes
        ):
            _set_native_title_band([line], lines)
    _classify_repeated_multiline_display_bands(short, body, page_size, page_index, lines)
    # 重复冒号小标题必须比紧随的常规条目更粗，并有两行正文；冒号引导句仍保持正文。
    labels = []
    for line in short:
        if (
            line.semantic_type is not None
            or not line.text.rstrip().endswith((":", "："))
            or not 3 <= len(line.text.split()) <= 8
        ):
            continue
        h = line.effective_height
        followers = [
            other
            for other in regular
            if other.semantic_type is None
            and 0.8 <= other.effective_height / h <= 1.2
            and 0 <= other.bbox[0] - line.bbox[0] <= 2 * h
            and 0 <= other.bbox[1] - line.bbox[3] <= 3 * h
            and (line.dominant_font_weight or 400) - (other.dominant_font_weight or 400) >= 100
        ]
        if len(followers) >= 2:
            labels.append(line)
    for line in labels:
        if sum(other.font_signature == line.font_signature for other in labels) >= 3:
            _set_native_title_band([line], lines)


def _set_native_title_band(members, all_lines):
    """明确原生标题带共享身份，同行碎片与续行合并而相邻不同级标题保持分离。"""
    band = min(line.source_index for line in members)
    group = max((line.paragraph_group for line in all_lines if line.paragraph_group is not None), default=-1) + 1
    for line in members:
        line.semantic_type = "paragraph_title"
        line.structural_title = line.explicit_section_title = True
        line.title_band_id = band
        line.paragraph_group = group
        line.title_suppressed = False


def _freeze_wrapped_bold_title_evidence(line: _LineItem, page_size: tuple[float, float]) -> None:
    """释放原生字符前冻结短粗体前缀与常规正文的两段几何，不保留跨页字符字典。"""
    line.native_title_label_left = _numbered_title_label_left(line)
    prefix = []
    for char in line.chars:
        if (char.get("font") or {}).get("weight", 0) < 600:
            break
        prefix.append(char)
    tail_text = "".join(char.get("char", "") for char in prefix).strip()
    rest = line.chars[len(prefix) :]
    body_text = "".join(char.get("char", "") for char in rest).strip()
    if (
        not prefix
        or not rest
        or not 1 <= len(tail_text.split()) <= 3
        or not tail_text.endswith((".", ":", "。", "："))
        or len(body_text.split()) < 4
        or any((char.get("font") or {}).get("weight", 0) > 500 for char in rest if char.get("char", "").strip())
    ):
        return
    parts = []
    for text, chars in ((tail_text, prefix), (body_text, rest)):
        bounds = _bbox_union_many([tuple(char["bbox"]) for char in chars if char.get("char", "").strip()])
        part = replace(
            line, text=text, bbox=bounds, source_bbox=bounds, ink_bbox=None, chars=list(chars), wrapped_title_parts=None
        )
        _fill_native_typography(part, page_size)
        part.chars.clear()
        parts.append(part)
    line.wrapped_title_parts = (parts[0], parts[1])


def _restore_wrapped_bold_title_tails(lines: list[_LineItem], page_size: tuple[float, float]) -> list[_LineItem]:
    """标题下一行的短加粗前缀与标题同式时按冻结原生证据切回标题，常规正文保持完整段组。"""
    output = list(lines)
    next_source = max((line.source_index for line in lines), default=-1) + 1
    for current in lines:
        if current.angle != 0 or current.semantic_type is not None or current.caption_start:
            continue
        if current.wrapped_title_parts is None and current.chars:
            _freeze_wrapped_bold_title_evidence(current, page_size)
        if current.wrapped_title_parts is None:
            continue
        suffix, body = current.wrapped_title_parts
        prefix_box = suffix.bbox
        em = _line_effective_height(current, current.bbox)
        previous = [
            line
            for line in lines
            if line.semantic_type == "paragraph_title"
            and line.angle == 0
            and (line.dominant_font_weight or 0) >= 600
            and not line.paragraph_terminal
            and not re.search(r"[.!?。！？]$", line.text.strip())
            and abs(_numbered_title_label_left(line) - prefix_box[0]) <= 0.5 * em
            and -0.1 * em <= prefix_box[1] - line.bbox[3] <= 0.65 * em
            and line.font_signature is not None
            and line.font_signature == suffix.font_signature
        ]
        if len(previous) != 1:
            continue
        head = previous[0]
        suffix = replace(suffix, source_index=next_source)
        next_source += 1
        title_members = [
            head,
            suffix,
            *(
                line
                for line in lines
                if line is not head
                and line.semantic_type == "paragraph_title"
                and line.font_signature == head.font_signature
                and _bbox_axis_overlap_ratio(line.bbox, head.bbox, axis="y") >= 0.7
                and 0 <= head.bbox[0] - line.bbox[2] <= 3 * em
            ),
        ]
        _set_native_title_band(title_members, output)
        current.text, current.bbox, current.source_bbox = body.text, body.bbox, body.source_bbox
        current.ink_bbox, current.chars = None, []
        current.font_signature, current.dominant_font_weight = body.font_signature, body.dominant_font_weight
        current.median_glyph_width, current.leading_emphasis_width = body.median_glyph_width, None
        current.wrapped_title_parts = None
        group = max((line.paragraph_group for line in output if line.paragraph_group is not None), default=-1) + 1
        paragraph = [current]
        for line in sorted(
            (line for line in lines if line.source_index > current.source_index), key=lambda line: line.source_index
        ):
            before = paragraph[-1]
            if (
                line.semantic_type is not None
                or line.caption_start
                or line.angle != 0
                or line.font_signature != current.font_signature
                or not 0.9 <= _line_effective_height(line, line.bbox) / em <= 1.1
                or not -0.1 * em <= line.bbox[1] - before.bbox[3] <= 0.65 * em
                or abs(line.bbox[0] - prefix_box[0]) > 0.5 * em
            ):
                break
            paragraph.append(line)
        for line in paragraph:
            line.paragraph_group = group
        output.append(suffix)
    return output


def _numbered_title_label_left(line: _LineItem) -> float:
    """同排序号并入标题时用实际标签首字左缘，不能把序号的悬挂缩进当成正文续行偏移。"""
    if line.native_title_label_left is not None:
        return line.native_title_label_left
    if re.match(r"^(?:[A-Za-z]|\d{1,2})[.)]\s+", line.text) and line.chars:
        marker_ended = False
        for char in line.chars:
            text = char.get("char", "")
            if marker_ended and text.strip():
                return float(char["bbox"][0])
            if text in {".", ")"}:
                marker_ended = True
    return line.bbox[0]


def _restore_short_heading_with_image_displaced_prose(lines: list[_LineItem], image_bboxes: list[BBox]) -> None:
    """原生连续两行正文的短收句被整幅图挤开时保持段组，并保留上方有净空的独立短题名。"""
    rows = sorted(lines, key=lambda line: line.source_index)
    for index in range(len(rows) - 2):
        first, second, tail = rows[index : index + 3]
        if any(
            line.angle != 0
            or line.semantic_type is not None
            or line.paragraph_group is not None
            or line.caption_start
            or line.formula_candidate_only
            for line in (first, second, tail)
        ):
            continue
        em = _line_effective_height(first, first.bbox)
        if (
            first.font_signature is None
            or any(line.font_signature != first.font_signature for line in (second, tail))
            or any(not 0.9 <= _line_effective_height(line, line.bbox) / em <= 1.1 for line in (second, tail))
            or len(first.text.split()) < 8
            or len(second.text.split()) < 8
            or second.paragraph_terminal
            or re.search(r"[.!?;:。！？；：]$", second.text.rstrip())
            or not re.fullmatch(r"[a-z][A-Za-z-]*(?:\s+[a-z][A-Za-z-]*){0,2}[.!?]", tail.text.strip())
            or not -0.1 * em <= second.bbox[1] - first.bbox[3] <= 0.6 * em
            or abs(first.bbox[0] - second.bbox[0]) > 0.25 * em
            or abs(first.bbox[0] - tail.bbox[0]) > 0.25 * em
            or tail.bbox[2] - tail.bbox[0] > 5 * em
        ):
            continue
        images = [
            box
            for box in image_bboxes
            if box[3] - box[1] >= 8 * em
            and second.bbox[3] - 0.2 * em <= box[1] <= second.bbox[3] + 2 * em
            and tail.bbox[1] - em <= box[3] <= tail.bbox[3] + em
            and _bbox_axis_overlap_ratio(box, first.bbox, axis="x") >= 0.5
            and _bbox_axis_overlap_ratio(box, tail.bbox, axis="x") <= 0.1
        ]
        if len(images) != 1:
            continue
        group = max((line.paragraph_group for line in lines if line.paragraph_group is not None), default=-1) + 1
        for line in (first, second, tail):
            line.paragraph_group = group
        if index:
            heading = rows[index - 1]
            if (
                heading.angle == 0
                and heading.semantic_type is None
                and heading.font_signature == first.font_signature
                and 2 <= len(heading.text.split()) <= 6
                and not re.search(r"[.!?。！？:：;；]$", heading.text.rstrip())
                and abs(heading.bbox[0] - first.bbox[0]) <= 0.25 * em
                and em <= first.bbox[1] - heading.bbox[3] <= 3 * em
            ):
                _set_native_title_band([heading], lines)


def _classify_repeated_multiline_display_bands(short, body, page_size, page_index, all_lines):
    """重复大标题带或后续页中文跨行章节由同字体、同尺度与稳定对齐聚合。"""
    large = [
        line
        for line in short
        if line.effective_height >= 1.4 * body
        and (
            (line.dominant_font_weight or 400) >= 600 or page_index > 0 and len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 3
        )
    ]
    rows = []
    for line in sorted(large, key=lambda item: (item.bbox[1], item.bbox[0])):
        peers = [
            row
            for row in rows
            if row[0].font_signature == line.font_signature
            and _bbox_axis_overlap_ratio(row[0].bbox, line.bbox, axis="y") >= 0.75
            and -0.1 * body
            <= line.bbox[0] - max(member.bbox[2] for member in row)
            <= (
                6 * line.effective_height
                if all(re.search(r"[\u3400-\u9fff]", member.text) for member in row + [line])
                else 0.5 * line.effective_height
            )
        ]
        if peers:
            peers[-1].append(line)
        else:
            rows.append([line])
    bands = []
    pending = list(rows)
    while pending:
        band = [pending.pop(0)]
        while True:
            bounds = _bbox_union_many([line.bbox for line in band[-1]])
            h = band[0][0].effective_height
            following = [
                row
                for row in pending
                if row[0].font_signature == band[0][0].font_signature
                and 0.9 <= row[0].effective_height / h <= 1.1
                and -0.2 * h <= min(line.bbox[1] for line in row) - bounds[3] <= 0.6 * h
                and _bbox_center_y(_bbox_union_many([line.bbox for line in row])) > _bbox_center_y(bounds) + 0.5 * h
                and (
                    abs(_bbox_union_many([line.bbox for line in row])[0] - bounds[0]) <= 0.4 * h
                    or abs(_bbox_center_x(_bbox_union_many([line.bbox for line in row])) - _bbox_center_x(bounds)) <= 0.4 * h
                )
            ]
            if not following:
                break
            row = min(following, key=lambda members: min(line.bbox[1] for line in members))
            pending.remove(row)
            band.append(row)
        if len(band) >= 2:
            bands.append([line for row in band for line in row])
    for band in bands:
        chinese = (
            page_index > 0
            and any("：" in line.text or ":" in line.text for line in band[:1])
            and all(len(re.findall(r"[\u3400-\u9fff]", line.text)) >= 3 for line in band)
        )
        repeated = (
            all(line.text.isupper() and re.search(r"[A-Z]", line.text) for line in band)
            and sum(
                other[0].font_signature == band[0].font_signature
                and 0.9 <= other[0].effective_height / band[0].effective_height <= 1.1
                and all(line.text.isupper() for line in other)
                for other in bands
            )
            >= 2
        )
        if chinese or repeated:
            _set_native_title_band(band, all_lines)


def _classify_bold_numbered_heading_rows(
    lines: list[_LineItem], page_size: tuple[float, float], containers: list[BBox], *, appendix_only: bool = False
) -> None:
    """由粗体、编号、同基线和留白恢复数字或附录字母标题，保持成员一致。"""
    available = [line for line in lines if line.angle == 0 and line.semantic_type in {None, "paragraph_title"}]
    for line in available:
        if line.title_band_id is not None and line.structural_title:
            continue
        text = line.text.strip()
        if appendix_only and not re.match(r"^[A-Z](?:\s|$)", text):
            continue
        if not re.match(r"^(?:\d+(?:\.\d+)*\.?|[A-Z])(?:\s|$|[-–—]\s*(?=[A-Z]))", text):
            continue
        height = _line_effective_height(line, line.bbox)
        members = [line]
        if re.fullmatch(r"\d+(?:\.\d+)*\.?|[A-Z]", text):
            companions = [
                other
                for other in available
                if other is not line
                and other.bbox[0] >= line.bbox[2]
                and other.bbox[0] - line.bbox[2] <= 2 * height
                and _bbox_axis_overlap_ratio(other.bbox, line.bbox, axis="y") >= 0.6
            ]
            if not companions:
                continue
            members.append(min(companions, key=lambda other: other.bbox[0]))
        body = members[-1]
        label = " ".join(member.text.strip() for member in members)
        bounds = _bbox_union_many([member.bbox for member in members])
        if (
            not _line_has_bold_font(body)
            or body.font_coverage < 0.75
            or len(label.split()) > 15
            or re.search(r"[A-Za-z\u3400-\u9fff]", body.text) is None
            # 连续单字母图例或数学标签不构成附录题名，即使字体明确加粗也不能晋升。
            or re.fullmatch(r"[A-Z](?:\s+[A-Z])*", body.text.strip()) is not None
            or re.match(r"^\d{4,}(?:\s|$)", label)
            or re.search(r"[.!?。！？:：;；,，]$", label)
            or bounds[2] - bounds[0] > 0.8 * page_size[0]
            or any(
                _bbox_axis_overlap_ratio(bounds, box, axis="x") > 0.5 and _bbox_axis_overlap_ratio(bounds, box, axis="y") > 0.5
                for box in containers
            )
        ):
            continue
        # 粗体长标题在编号后的文字栏缩进续行；即使续行以数字开头，也继承同一段界身份。
        if len(label.split()) >= 5:
            continuations = [
                other
                for other in available
                if other not in members
                and (
                    height < other.bbox[0] - bounds[0] <= 3 * height
                    or abs(other.bbox[0] - bounds[0]) <= 0.25 * height
                    and not re.match(r"^(?:\d+(?:\.\d+)*\.?|[A-Z])(?:\s|$)", other.text.strip())
                )
                and -0.25 * height <= other.bbox[1] - bounds[3] <= 0.75 * height
                and _bbox_center_y(other.bbox) > _bbox_center_y(bounds) + 0.5 * height
                and other.bbox[2] <= bounds[2] + height
                and _line_has_bold_font(other)
                and body.font_signature == other.font_signature
                and 0.9 <= _line_effective_height(other, other.bbox) / height <= 1.1
            ]
            if continuations:
                members.append(min(continuations, key=lambda other: other.bbox[1]))
        for member in members:
            member.semantic_type = "paragraph_title"
            member.structural_title = True
            member.explicit_section_title = True
            member.title_band_id = line.source_index
            member.title_suppressed = False
    if not appendix_only:
        _classify_display_roman_heading_rows(lines, page_size, containers)
    _classify_headings_after_extreme_local_metric_repair(lines, page_size, containers)


def _classify_headings_after_extreme_local_metric_repair(lines, page_size, containers):
    """极端行框校正后，使用粗体、上下净空和正文续行确认同字号操作章节标题。"""
    repaired = [
        line
        for line in lines
        if line.style_scale_repaired
        and line.source_bbox is not None
        and line.source_bbox[3] - line.source_bbox[1] >= 4 * _line_effective_height(line, line.bbox)
    ]
    if len(repaired) < 6:
        return
    body_height = statistics.median(_line_effective_height(line, line.bbox) for line in repaired)
    available = [line for line in lines if line.angle == 0 and line.semantic_type in {None, "paragraph_title"}]
    for line in available:
        text = line.text.strip()
        height = _line_effective_height(line, line.bbox)
        if (
            (line.dominant_font_weight or 0) < 600
            or line.font_coverage < 0.4
            or not 0.85 * body_height <= height <= 1.5 * body_height
            or not 2 <= len(text.split()) <= 20
            or re.match(r"^[\d•●▪]", text)
            or re.search(r"[.!?。！？]$", text)
            or line.caption_start
            or line.formula_candidate_only
            or line.inline_math_regions
            or line.bbox[2] - line.bbox[0] > 0.85 * page_size[0]
            or _line_inside_visual_container(line.bbox, containers)
        ):
            continue
        # 同基线的独立表头、字段名和强调片段不成为独立章节。
        if any(other is not line and _bbox_axis_overlap_ratio(other.bbox, line.bbox, axis="y") >= 0.6 for other in available):
            continue
        followers = [
            other
            for other in available
            if other is not line
            and (other.dominant_font_weight or 400) < 600
            and -0.25 * body_height <= other.bbox[0] - line.bbox[0] <= 2.5 * body_height
            and 0.2 * body_height <= other.bbox[1] - line.bbox[3] <= 8 * body_height
        ]
        if len(followers) < 2:
            continue
        following = min(followers, key=lambda other: other.bbox[1])
        previous = [
            other.bbox[3]
            for other in available
            if other is not line
            and other.bbox[3] <= line.bbox[1]
            and _bbox_axis_overlap_ratio(other.bbox, line.bbox, axis="x") >= 0.2
        ]
        gap_above = line.bbox[1] - max(previous) if previous else body_height
        gap_below = following.bbox[1] - line.bbox[3]
        if not 0.3 * body_height <= gap_above <= 3 * body_height or gap_below > 2.5 * body_height:
            continue
        line.semantic_type = "paragraph_title"
        line.structural_title = line.explicit_section_title = True
        line.title_band_id = line.source_index
        line.title_suppressed = False


def _classify_display_roman_heading_rows(
    lines: list[_LineItem], page_size: tuple[float, float], containers: list[BBox]
) -> None:
    """大号粗体罗马编号与下方左对齐展示标题组成结构带；普通页码和正文数学符号不晋升。"""
    available = [line for line in lines if line.angle == 0 and line.semantic_type in {None, "paragraph_title", "doc_title"}]
    scales = [_line_canonical_style_scale(line, line.bbox) for line in available if len(line.text.split()) >= 4]
    if len(scales) < 3:
        return
    body_height = statistics.median(scales)
    for marker in available:
        height = _line_canonical_style_scale(marker, marker.bbox)
        if (
            re.fullmatch(r"[IVXLCDM]{1,8}\.", marker.text.strip()) is None
            or height < 1.8 * body_height
            or (marker.dominant_font_weight or 0) < 600
            or marker.font_coverage < 0.75
            or marker.bbox[2] - marker.bbox[0] > 5 * height
            or _line_inside_visual_container(marker.bbox, containers)
        ):
            continue
        candidates = [
            line
            for line in available
            if line is not marker
            and not line.caption_start
            and abs(line.bbox[0] - marker.bbox[0]) <= 0.25 * height
            and -0.25 * height <= line.bbox[1] - marker.bbox[3] <= 0.5 * height
            and _bbox_center_y(line.bbox) > _bbox_center_y(marker.bbox) + 0.5 * height
            and 0.6 * height <= _line_canonical_style_scale(line, line.bbox) <= 1.4 * height
            and _line_canonical_style_scale(line, line.bbox) >= 1.8 * body_height
            and (line.dominant_font_weight or 0) >= 600
            and line.font_coverage >= 0.75
            and re.search(r"[A-Za-z]{2,}|[\u3400-\u9fff]{2,}", line.text)
            and re.fullmatch(r"[IVXLCDM]+\.?", line.text.strip()) is None
            and line.bbox[2] - line.bbox[0] <= 0.6 * page_size[0]
            and not _line_inside_visual_container(line.bbox, containers)
        ]
        if not candidates:
            continue
        label = min(candidates, key=lambda line: _bbox_center_y(line.bbox))
        members = [marker, label]
        for _ in range(2):
            previous = members[-1]
            label_height = _line_canonical_style_scale(label, label.bbox)
            continuations = [
                line
                for line in available
                if line not in members
                and not line.caption_start
                and abs(line.bbox[0] - label.bbox[0]) <= 0.25 * label_height
                and -0.2 * label_height <= line.bbox[1] - previous.bbox[3] <= 0.4 * label_height
                and _bbox_center_y(line.bbox) > _bbox_center_y(previous.bbox) + 0.5 * label_height
                and line.font_signature == label.font_signature
                and 0.85 <= _line_canonical_style_scale(line, line.bbox) / label_height <= 1.15
                and not _line_inside_visual_container(line.bbox, containers)
            ]
            if not continuations:
                break
            members.append(min(continuations, key=lambda line: line.bbox[1]))
        for member in members:
            member.semantic_type = "paragraph_title"
            member.structural_title = member.explicit_section_title = True
            member.title_band_id = marker.source_index
            member.title_suppressed = False


def _demote_regular_repeated_item_titles(
    lines: list[_LineItem],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """同字号常规字体的邻近圆点或连续编号项目提供正文反证，防止列表首项被短行标题规则晋升。"""
    from ..inline.detection import _font_styles_from_metadata

    body_height = (
        document_body_profile.body_height
        if document_body_profile
        else statistics.median(
            [_line_effective_height(line, line.bbox) for line in lines if len(line.text.split()) >= 5] or [1.0]
        )
    )
    candidates = []
    for line in lines:
        match = re.match(r"^\s*(?:[•●▪]\s+|(?P<number>\d{1,2})[.)]\s+)\S", line.text)
        if (
            match is None
            or line.angle != 0
            or line.semantic_type not in {None, "paragraph_title"}
            or line.structural_title
            or line.explicit_section_title
            or line.caption_start
            or line.font_signature is None
            or line.font_coverage < 0.75
            or _line_effective_height(line, line.bbox) > 1.15 * body_height
            or "bold" in _font_styles_from_metadata(line.font_signature[0], line.font_signature[1], line.dominant_font_weight)
        ):
            continue
        candidates.append((line, int(match["number"]) if match["number"] else None))
    for line, number in candidates:
        height = _line_effective_height(line, line.bbox)
        peers = [
            other
            for other, other_number in candidates
            if other is not line
            and (
                number is None
                and other_number is None
                or number is not None
                and other_number is not None
                and abs(number - other_number) == 1
            )
            and line.font_signature == other.font_signature
            and 0.9 * height <= _line_effective_height(other, other.bbox) <= 1.1 * height
            and abs(line.bbox[0] - other.bbox[0]) <= 0.5 * height
            and 0.75 * height <= abs(_bbox_center_y(line.bbox) - _bbox_center_y(other.bbox)) <= 8 * height
        ]
        if not peers:
            continue
        line.semantic_type = None
        line.title_suppressed = True


def _normalized_section_title_text(text: str) -> str:
    """规范全半角编号和空白，仅用于结构标题规则判断。"""

    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", text),
    ).strip()


def _is_plausible_section_number(
    number: str,
    label: str = "",
) -> bool:
    """排除年代、小数值和整句正文冒充的章节编号。"""

    parts = [int(part) for part in re.findall(r"\d+", number)]
    if not parts or any(part > 99 for part in parts):
        return False
    if len(parts) > 1 and parts[0] == 0:
        return False
    if len(parts) == 1 and parts[0] > 12:
        return False
    stripped_label = label.lstrip()
    if stripped_label and not stripped_label[0].isalpha():
        return False
    if stripped_label and stripped_label[0].isascii() and stripped_label[0].isalpha() and not stripped_label[0].isupper():
        return False
    if len(re.findall(r"\d+(?:\.\d+)?", label)) >= 2:
        return False
    return not any(char in label for char in ",，;；:：。!?！？")


def _section_title_has_body_followers(
    title_bbox: BBox,
    geometry: list[tuple[_LineItem, BBox]],
    body_height: float,
    local_page_width: float,
    *,
    minimum_count: int,
) -> bool:
    """检查紧随标题的同栏常规正文行，避免把页码和数值标成标题。"""

    followers = 0
    for line, bbox in sorted(
        geometry,
        key=lambda item: (item[1][1], item[1][0], item[0].source_index),
    ):
        if bbox[1] <= title_bbox[1] + 0.4 * body_height:
            continue
        if bbox[1] - title_bbox[3] > 8.0 * body_height:
            break
        line_height = _line_effective_height(line, bbox)
        horizontally_related = (
            _bbox_axis_overlap_ratio(title_bbox, bbox, axis="x") >= 0.15 or abs(bbox[0] - title_bbox[0]) <= 2.5 * body_height
        )
        if (
            line.semantic_type is None
            and horizontally_related
            and bbox[2] - bbox[0] >= 0.25 * local_page_width
            and 0.7 <= line_height / body_height <= 1.4
        ):
            followers += 1
            if followers >= minimum_count:
                return True
    return False


def _classify_explicit_section_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """以通用编号行和紧凑结构转折补齐正文同字号章节标题。"""

    if document_body_profile is None or document_body_profile.body_height <= 0:
        return
    body_height = document_body_profile.body_height
    for angle in sorted(
        {
            line.angle
            for line in lines
            if (line.semantic_type is None or line.explicit_section_title) and not line.title_suppressed
        }
    ):
        geometry = sorted(
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
                if line.angle == angle and line.semantic_type is None and not line.title_suppressed
            ],
            key=lambda item: (
                item[1][1],
                item[1][0],
                item[0].source_index,
            ),
        )
        if not geometry:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_containers = [
            _rotate_bbox_to_upright(
                bbox,
                page_size,
                angle,
            )
            for bbox in container_bboxes
        ]
        numbered_groups: list[list[tuple[_LineItem, BBox]]] = []
        grouped_sources: set[int] = set()
        for line, bbox in geometry:
            normalized = _normalized_section_title_text(line.text)
            numbered_match = _NUMBERED_SECTION_TITLE_RE.match(
                normalized,
            )
            if numbered_match is not None and _is_plausible_section_number(
                numbered_match.group("number"),
                numbered_match.group("label"),
            ):
                numbered_groups.append([(line, bbox)])
                grouped_sources.add(line.source_index)
                continue
            if _SECTION_NUMBER_ONLY_RE.match(normalized) is None or not _is_plausible_section_number(normalized):
                continue
            marker_height = _line_effective_height(line, bbox)
            companions = [
                (candidate, candidate_bbox)
                for candidate, candidate_bbox in geometry
                if candidate is not line
                and candidate.source_index not in grouped_sources
                and candidate_bbox[0] >= bbox[2]
                and candidate_bbox[0] - bbox[2] <= 4.0 * max(body_height, marker_height)
                and _bbox_axis_overlap_ratio(
                    bbox,
                    candidate_bbox,
                    axis="y",
                )
                >= 0.5
                and candidate_bbox[2] - candidate_bbox[0] <= 0.55 * local_page_width
                and _SECTION_TITLE_TERMINAL_RE.search(
                    _normalized_section_title_text(candidate.text),
                )
                is None
            ]
            if not companions:
                continue
            companion = min(
                companions,
                key=lambda item: (
                    item[1][0] - bbox[2],
                    item[1][1],
                ),
            )
            numbered_groups.append([(line, bbox), companion])
            grouped_sources.update({line.source_index, companion[0].source_index})

        for group in numbered_groups:
            first_line, first_bbox = group[0]
            # 编号标题折行后常有短的缩进尾行；共同汉字字体及独立正文起点确认完整题名。
            first_fonts = {
                (char.get("font") or {}).get("name")
                for char in first_line.chars or []
                if re.fullmatch(r"[\u4e00-\u9fff]", char.get("char", ""))
            }
            continuations = [
                (line, bbox)
                for line, bbox in geometry
                if line is not first_line
                and 0 <= bbox[1] - first_bbox[3] <= body_height
                and 0.5 * body_height <= bbox[0] - first_bbox[0] <= 4 * body_height
                and bbox[2] - bbox[0] <= 0.5 * (first_bbox[2] - first_bbox[0])
                and re.fullmatch(r"[\u4e00-\u9fff]{2,16}", line.text.strip())
                and (
                    any((char.get("font") or {}).get("name") in first_fonts for char in line.chars or [])
                    or first_line.native_cjk_style
                    and line.native_cjk_style
                    and first_line.native_cjk_style[:3] == line.native_cjk_style[:3]
                )
            ]
            if continuations:
                group.append(min(continuations, key=lambda item: item[1][1]))
            title_bbox = _bbox_union_many(
                [bbox for _line, bbox in group],
            )
            group_line_ids = {id(line) for line, _bbox in group}
            preceding = [
                previous_bbox
                for previous_line, previous_bbox in geometry
                if id(previous_line) not in group_line_ids
                and previous_bbox[3] <= title_bbox[1]
                and (
                    _bbox_axis_overlap_ratio(
                        previous_bbox,
                        title_bbox,
                        axis="x",
                    )
                    >= 0.15
                    or abs(previous_bbox[0] - title_bbox[0]) <= 2.5 * body_height
                )
            ]
            gap_above = title_bbox[1] - max(previous_bbox[3] for previous_bbox in preceding) if preceding else body_height
            if (
                title_bbox[2] - title_bbox[0] > 0.7 * local_page_width
                or not 0.1 * local_page_height <= _bbox_center_y(title_bbox) <= 0.93 * local_page_height
                or any(
                    _bbox_axis_overlap_ratio(
                        title_bbox,
                        container_bbox,
                        axis="x",
                    )
                    >= 0.8
                    and _bbox_axis_overlap_ratio(
                        title_bbox,
                        container_bbox,
                        axis="y",
                    )
                    >= 0.8
                    for container_bbox in local_containers
                )
                or not _section_title_has_body_followers(
                    title_bbox,
                    geometry,
                    body_height,
                    local_page_width,
                    minimum_count=1,
                )
                or gap_above < (0.25 if continuations else 0.4) * body_height
            ):
                continue
            for line, _bbox in group:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                line.explicit_section_title = True
                if continuations:
                    line.title_band_id = first_line.source_index

        for line, bbox in geometry:
            if line.semantic_type is not None:
                continue
            normalized = _normalized_section_title_text(line.text)
            canonical_heading = normalized.strip(
                "[]［］【】()（）",
            ).replace(" ", "")
            if (
                _UNNUMBERED_SECTION_HEADING_RE.fullmatch(
                    canonical_heading,
                )
                is None
                or not 2 <= len(normalized) <= 24
                or _SECTION_TITLE_TERMINAL_RE.search(normalized) is not None
                or any(char in normalized for char in "[]［］")
                or bbox[2] - bbox[0] > 0.3 * local_page_width
                or _bbox_center_y(bbox) < 0.35 * local_page_height
                or not 0.75 <= _line_effective_height(line, bbox) / body_height <= 1.4
                or any(
                    _bbox_axis_overlap_ratio(
                        bbox,
                        container_bbox,
                        axis="x",
                    )
                    >= 0.8
                    and _bbox_axis_overlap_ratio(
                        bbox,
                        container_bbox,
                        axis="y",
                    )
                    >= 0.8
                    for container_bbox in local_containers
                )
                or not _section_title_has_body_followers(
                    bbox,
                    geometry,
                    body_height,
                    local_page_width,
                    minimum_count=2,
                )
            ):
                continue
            preceding = [
                previous_bbox
                for previous_line, previous_bbox in geometry
                if previous_line is not line
                and previous_bbox[3] <= bbox[1]
                and (
                    _bbox_axis_overlap_ratio(
                        previous_bbox,
                        bbox,
                        axis="x",
                    )
                    >= 0.15
                    or abs(previous_bbox[0] - bbox[0]) <= 2.5 * body_height
                )
            ]
            gap_above = bbox[1] - max(previous_bbox[3] for previous_bbox in preceding) if preceding else body_height
            if gap_above >= 0.5 * body_height:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                line.explicit_section_title = True


def _classify_document_structural_titles(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile | None,
    *,
    legacy_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> None:
    """用跨页稳定栏带和段前后转折补齐正文同字号标题。"""

    if document_body_profile is None or not document_body_profile.has_style_scale_repairs:
        return
    probe_pages = [
        replace(
            prepared,
            remaining_lines=[
                replace(
                    line,
                    style_scale_repaired=True,
                    structural_title=False,
                )
                for line in prepared.remaining_lines
            ],
        )
        for prepared in prepared_pages
    ]
    _classify_document_structural_title_candidates(
        probe_pages,
        document_body_profile,
    )
    canonical_candidate_sources = {
        (page_index, line.source_index)
        for page_index, prepared in enumerate(probe_pages)
        for line in prepared.remaining_lines
        if line.structural_title
    }
    legacy_title_sources = _collect_legacy_paragraph_title_sources(
        prepared_pages,
        legacy_body_profile,
        document_title_profile,
    )
    body_height = max(0.1, document_body_profile.body_height)
    canonical_style_candidate_pages: dict[
        tuple[str, int, float, int],
        list[int],
    ] = {}
    for page_index, prepared in enumerate(prepared_pages):
        for line in prepared.remaining_lines:
            line_key = (page_index, line.source_index)
            if line_key not in canonical_candidate_sources:
                continue
            local_bbox = _rotate_bbox_to_upright(
                line.source_bbox or line.bbox,
                prepared.page_size,
                line.angle,
            )
            layout_ratio = (local_bbox[3] - local_bbox[1]) / body_height
            if (
                line_key in legacy_title_sources
                or layout_ratio >= 1.8
                or not _line_uses_document_regular_font(
                    line,
                    document_body_profile,
                )
            ):
                line.semantic_type = "paragraph_title"
                line.structural_title = True
                if (
                    line_key not in legacy_title_sources
                    and layout_ratio >= 1.8
                    and (
                        style_key := _canonical_title_style_key(
                            line,
                        )
                    )
                    is not None
                ):
                    canonical_style_candidate_pages.setdefault(
                        style_key,
                        [],
                    ).append(page_index)
    canonical_style_prototypes = {
        style_key
        for style_key, page_indices in canonical_style_candidate_pages.items()
        if len(set(page_indices)) >= 2 or max(Counter(page_indices).values(), default=0) >= 3
    }
    if canonical_style_prototypes:
        for prepared in prepared_pages:
            prepared.canonical_formula_geometry = True
            for line in prepared.remaining_lines:
                style_key = _canonical_title_style_key(line)
                if style_key in canonical_style_prototypes:
                    line.style_scale_repaired = True


def _promote_noninitial_document_title_band(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    page_index: int,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
    title_candidate_source_indices: set[int],
) -> None:
    """把非首页中已确认且显著大于正文的最强段落标题带升为文档标题。"""

    if page_index == 0 or document_body_profile is None:
        return
    body_height = max(0.1, document_body_profile.body_height)
    page_candidates: list[
        tuple[
            tuple[float, float, int, float],
            list[tuple[_LineItem, BBox]],
        ]
    ] = []
    for angle in sorted(
        {
            line.angle
            for line in lines
            if line.source_index in title_candidate_source_indices
            and line.semantic_type in {None, "paragraph_title"}
            and not line.title_suppressed
        }
    ):
        geometry = sorted(
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
                if line.angle == angle
                and line.source_index in title_candidate_source_indices
                and line.semantic_type in {None, "paragraph_title"}
                and not line.title_suppressed
            ],
            key=lambda item: (
                item[1][1],
                item[1][0],
                item[0].source_index,
            ),
        )
        if not geometry:
            continue
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        local_containers = [
            _rotate_bbox_to_upright(
                bbox,
                page_size,
                angle,
            )
            for bbox in container_bboxes
        ]
        for index, (line, bbox) in enumerate(geometry):
            line_scale = _line_effective_height(line, bbox)
            width_ratio = (bbox[2] - bbox[0]) / max(
                0.1,
                local_page_width,
            )
            centered = abs(_bbox_center_x(bbox) - 0.5 * local_page_width) <= 0.08 * local_page_width
            if (
                line_scale < 1.25 * body_height
                or not 0.45 <= width_ratio <= 0.85
                or not centered
                or not 0.12 * local_page_height <= _bbox_center_y(bbox) <= 0.75 * local_page_height
                or _line_inside_visual_container(
                    bbox,
                    local_containers,
                )
            ):
                continue

            title_members = [(line, bbox)]
            cursor = index + 1
            while cursor < len(geometry):
                candidate_line, candidate_bbox = geometry[cursor]
                candidate_scale = _line_effective_height(
                    candidate_line,
                    candidate_bbox,
                )
                vertical_gap = max(
                    0.0,
                    candidate_bbox[1] - title_members[-1][1][3],
                )
                if (
                    candidate_scale < 1.2 * body_height
                    or vertical_gap
                    > 1.5
                    * max(
                        line_scale,
                        candidate_scale,
                    )
                    or abs(_bbox_center_x(candidate_bbox) - 0.5 * local_page_width) > 0.1 * local_page_width
                    or candidate_bbox[2] - candidate_bbox[0] > 0.85 * local_page_width
                    or _line_inside_visual_container(
                        candidate_bbox,
                        local_containers,
                    )
                    or not _title_fonts_compatible(
                        title_members[-1][0],
                        candidate_line,
                    )
                ):
                    break
                title_members.append(
                    (candidate_line, candidate_bbox),
                )
                cursor += 1

            title_scales = [_line_effective_height(title_line, title_bbox) for title_line, title_bbox in title_members]
            title_bbox = _bbox_union_many(
                [member_bbox for _member_line, member_bbox in title_members],
            )
            page_candidates.append(
                (
                    (
                        statistics.median(title_scales) / body_height,
                        sum(member_bbox[2] - member_bbox[0] for _member_line, member_bbox in title_members) / local_page_width,
                        len(title_members),
                        -_bbox_center_y(title_bbox) / local_page_height,
                    ),
                    title_members,
                )
            )

    if not page_candidates:
        return
    _score, title_members = max(
        page_candidates,
        key=lambda item: item[0],
    )
    for title_line, _title_bbox in title_members:
        title_line.semantic_type = "doc_title"


def _canonical_title_style_key(
    line: _LineItem,
) -> tuple[str, int, float, int] | None:
    """返回 canonical-only 标题向同样式正文传播时使用的稳定键。"""

    if line.font_signature is None or line.em_height <= 0:
        return None
    font_family = _normalized_font_family(line.font_signature)
    if font_family is None:
        return None
    return (
        font_family,
        line.font_signature[1],
        round(line.em_height * 4.0) / 4.0,
        line.angle,
    )


def _classify_document_structural_title_candidates(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile,
) -> None:
    """在 canonical 行副本上收集所有满足结构转折的标题候选。"""

    body_height = max(0.1, document_body_profile.body_height)
    strong_candidates: list[tuple[int, _LineItem, tuple[str, int] | None, float]] = []
    start_candidates: list[tuple[int, _LineItem, tuple[str, int] | None, float]] = []
    accepted_anchor_positions: list[tuple[int, int, float, float]] = []
    for page_index, prepared in enumerate(prepared_pages):
        container_bboxes = [
            block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
        ]
        for angle in sorted({line.angle for line in prepared.remaining_lines if line.semantic_type is None}):
            geometry = sorted(
                [
                    (
                        line,
                        _rotate_bbox_to_upright(
                            line.source_bbox or line.bbox,
                            prepared.page_size,
                            angle,
                        ),
                    )
                    for line in prepared.remaining_lines
                    if line.angle == angle and line.semantic_type is None
                ],
                key=lambda item: (
                    item[1][1],
                    item[1][0],
                    item[0].source_index,
                ),
            )
            if len(geometry) < 4:
                continue
            local_page_width = prepared.page_size[1] if angle in {90, 270} else prepared.page_size[0]
            local_page_height = prepared.page_size[0] if angle in {90, 270} else prepared.page_size[1]
            median_height = statistics.median(_line_canonical_style_scale(line, bbox) for line, bbox in geometry)
            lanes = _infer_text_lanes(
                geometry,
                local_page_width,
                median_height,
            )
            physical_gaps = _build_physical_title_gap_map(geometry)
            local_containers = [
                _rotate_bbox_to_upright(
                    bbox,
                    prepared.page_size,
                    angle,
                )
                for bbox in container_bboxes
            ]
            for line, bbox in geometry:
                if page_index == 0 and _bbox_center_y(bbox) < 0.64 * local_page_height:
                    continue
                related_lanes = [
                    lane
                    for lane in lanes
                    if not lane.is_span
                    and len(lane.lines) >= 3
                    and lane.left - body_height <= _bbox_center_x(bbox) <= lane.right + body_height
                ]
                if not related_lanes:
                    continue
                lane = max(
                    related_lanes,
                    key=lambda item: (
                        len(item.lines),
                        item.right - item.left,
                    ),
                )
                lane_width = max(0.1, lane.right - lane.left)
                width_ratio = (bbox[2] - bbox[0]) / lane_width
                style_ratio = _line_canonical_style_scale(line, bbox) / body_height
                left_offset = (bbox[0] - lane.left) / body_height
                regular_font = _line_uses_document_regular_font(
                    line,
                    document_body_profile,
                )
                if (
                    not 0.75 <= style_ratio <= 1.35
                    or width_ratio > 0.8
                    or left_offset > 0.75
                    or left_offset < (-0.75 if regular_font else -3.0)
                    or _line_inside_visual_container(
                        bbox,
                        local_containers,
                    )
                ):
                    continue
                followers = [
                    (other_line, other_bbox)
                    for other_line, other_bbox in geometry
                    if other_line is not line
                    and bbox[1] < other_bbox[1]
                    and other_bbox[1] - bbox[3] <= 3.0 * body_height
                    and lane.left - body_height <= other_bbox[0] <= lane.left + 3.5 * body_height
                    and other_bbox[2] - other_bbox[0] >= 0.4 * lane_width
                ]
                if not followers:
                    continue
                gap_above, gap_below = physical_gaps.get(
                    line.source_index,
                    (None, None),
                )
                layout_ratio = (bbox[3] - bbox[1]) / body_height
                if regular_font and layout_ratio < 1.3 and line.font_coverage < 0.75:
                    continue
                standard_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.65 * body_height
                    and gap_below >= 0.35 * body_height
                    and (not regular_font or gap_below <= 1.5 * body_height)
                )
                low_coverage_wide_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.5 * body_height
                    and gap_below >= 0.45 * body_height
                    and gap_below <= 0.7 * body_height
                    and width_ratio >= 0.75
                    and line.font_coverage <= 0.7
                    and layout_ratio >= 1.8
                )
                if low_coverage_wide_transition and any(
                    anchor_page_index == page_index
                    and anchor_angle == angle
                    and abs(anchor_left - lane.left) <= body_height
                    and 0 < _bbox_center_y(bbox) - anchor_center_y <= 12.0 * body_height
                    for (
                        anchor_page_index,
                        anchor_angle,
                        anchor_left,
                        anchor_center_y,
                    ) in accepted_anchor_positions
                ):
                    low_coverage_wide_transition = False
                compact_regular_transition = (
                    gap_above is not None
                    and gap_below is not None
                    and gap_above >= 0.65 * body_height
                    and gap_below >= 0.15 * body_height
                    and width_ratio <= 0.45
                    and line.font_coverage >= 0.75
                    and layout_ratio <= 1.3
                )
                family_key = (
                    (
                        _normalized_font_family(line.font_signature),
                        line.font_signature[1],
                    )
                    if line.font_signature is not None
                    else None
                )
                if (
                    standard_transition
                    or low_coverage_wide_transition
                    or compact_regular_transition
                    or (gap_above is None and gap_below is not None and gap_below >= 0.6 * body_height and not regular_font)
                ):
                    strong_candidates.append(
                        (
                            page_index,
                            line,
                            family_key,
                            layout_ratio,
                        ),
                    )
                    accepted_anchor_positions.append(
                        (
                            page_index,
                            angle,
                            lane.left,
                            _bbox_center_y(bbox),
                        )
                    )
                    continue
                if (
                    gap_above is None
                    and bbox[1] <= 0.18 * local_page_height
                    and width_ratio <= 0.6
                    and (line.font_coverage >= 0.75 or (not regular_font and line.font_coverage >= 0.5))
                ):
                    start_candidates.append(
                        (
                            page_index,
                            line,
                            family_key,
                            layout_ratio,
                        ),
                    )

    for _page_index, line, _family_key, _layout_ratio in strong_candidates:
        line.semantic_type = "paragraph_title"
        line.structural_title = True
    strong_families_by_page = {
        (page_index, family_key) for page_index, _line, family_key, _layout_ratio in strong_candidates if family_key is not None
    }
    for page_index, line, family_key, _layout_ratio in start_candidates:
        if not _line_uses_document_regular_font(
            line,
            document_body_profile,
        ) or (family_key is not None and (page_index, family_key) in strong_families_by_page):
            line.semantic_type = "paragraph_title"
            line.structural_title = True


def _collect_legacy_paragraph_title_sources(
    prepared_pages: list[_PreparedPage],
    document_body_profile: _DocumentBodyProfile | None,
    document_title_profile: _DocumentTitleProfile | None,
) -> set[tuple[int, int]]:
    """在行副本上使用 legacy 尺度收集原本成立的段落标题身份。"""

    if document_body_profile is None:
        return set()
    legacy_profile = replace(
        document_body_profile,
        has_style_scale_repairs=False,
    )
    output: set[tuple[int, int]] = set()
    for page_index, prepared in enumerate(prepared_pages):
        probe_lines = [
            replace(
                line,
                style_scale_repaired=False,
                structural_title=False,
            )
            for line in prepared.remaining_lines
        ]
        container_bboxes = [
            block["bbox"] for block in prepared.fixed_blocks if not isinstance(block.get("_inline_visual_row_id"), int)
        ]
        caption_container_bboxes = [block["bbox"] for block in prepared.fixed_blocks if block.get("type") in {"image", "code"}]
        _classify_page_titles(
            probe_lines,
            prepared.page_size,
            page_index=page_index,
            container_bboxes=container_bboxes,
            caption_container_bboxes=caption_container_bboxes,
            document_body_profile=legacy_profile,
            document_title_profile=document_title_profile,
        )
        output.update((page_index, line.source_index) for line in probe_lines if line.semantic_type == "paragraph_title")
    return output


def _classify_inline_typography_reset_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
) -> None:
    """用短段尾、字体切换和缩进正文识别行内结构标题。"""

    if document_body_profile is None:
        return
    for angle in sorted({line.angle for line in lines if line.semantic_type is None and not line.title_suppressed}):
        line_geometry = [
            (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
            for line in lines
            if line.angle == angle and line.semantic_type is None and not line.title_suppressed
        ]
        if len(line_geometry) < 3:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        lanes = _infer_text_lanes(
            line_geometry,
            local_page_width,
            median_height,
        )
        local_container_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
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
            for previous, current, following in zip(
                rows,
                rows[1:],
                rows[2:],
            ):
                previous_line, previous_bbox = previous
                current_line, current_bbox = current
                following_line, following_bbox = following
                if any(
                    line.semantic_type is not None
                    for line in (
                        previous_line,
                        current_line,
                        following_line,
                    )
                ):
                    continue
                if (
                    previous_line.font_signature is None
                    or current_line.font_signature is None
                    or following_line.font_signature is None
                    or previous_line.font_coverage < 0.65
                    or current_line.font_coverage < 0.65
                    or following_line.font_coverage < 0.65
                ):
                    continue
                if not _font_signatures_share_family(
                    previous_line.font_signature,
                    following_line.font_signature,
                ) or _font_signatures_share_family(
                    current_line.font_signature,
                    previous_line.font_signature,
                ):
                    continue
                previous_height = _line_effective_height(*previous)
                current_height = _line_effective_height(*current)
                following_height = _line_effective_height(*following)
                neighbor_height = statistics.median(
                    (previous_height, following_height),
                )
                pair_height = max(
                    previous_height,
                    current_height,
                    following_height,
                )
                previous_width = previous_bbox[2] - previous_bbox[0]
                current_width = current_bbox[2] - current_bbox[0]
                following_width = following_bbox[2] - following_bbox[0]
                following_indent = following_bbox[0] - current_bbox[0]
                if not (
                    previous_width <= 0.45 * lane_width
                    and 0.2 * lane_width <= current_width <= 0.65 * lane_width
                    and following_width >= 0.75 * lane_width
                    and abs(current_bbox[0] - lane.left) <= 0.75 * pair_height
                    and 0.75 * pair_height <= following_indent <= 3.0 * pair_height
                    and 0.85 <= current_height / max(0.1, neighbor_height) <= 1.15
                    and -0.25 * pair_height <= _effective_text_row_gap(previous, current) <= 0.5 * pair_height
                    and -0.25 * pair_height <= _effective_text_row_gap(current, following) <= 0.5 * pair_height
                    and not _line_inside_visual_container(
                        current_bbox,
                        local_container_bboxes,
                    )
                ):
                    continue
                current_line.semantic_type = "paragraph_title"
                current_line.structural_title = True


def _classify_body_height_section_titles(
    lines: list[_LineItem],
    page_size: tuple[float, float],
    *,
    container_bboxes: list[BBox],
    document_body_profile: _DocumentBodyProfile | None,
    page_index: int = 1,
) -> None:
    """用重复的短行加正文组结构识别与正文同字号的独立章节标题。"""

    if document_body_profile is None:
        return
    body_height = document_body_profile.body_height
    if body_height <= 0:
        return

    for angle in sorted({line.angle for line in lines if line.semantic_type is None and not line.title_suppressed}):
        line_geometry = sorted(
            [
                (line, _rotate_bbox_to_upright(line.bbox, page_size, angle))
                for line in lines
                if line.angle == angle and line.semantic_type is None and not line.title_suppressed
            ],
            key=lambda item: (item[1][1], item[1][0], item[0].source_index),
        )
        if len(line_geometry) < 8:
            continue
        median_height = statistics.median(_line_effective_height(line, bbox) for line, bbox in line_geometry)
        local_page_width = page_size[1] if angle in {90, 270} else page_size[0]
        local_page_height = page_size[0] if angle in {90, 270} else page_size[1]
        lanes = _infer_text_lanes(
            line_geometry,
            local_page_width,
            median_height,
        )
        lane_by_source: dict[int, _TextLane] = {}
        regular_gaps: list[float] = []
        for lane in lanes:
            lane.lines.sort(key=lambda item: (item[1][1], item[1][0], item[0].source_index))
            regular_gap, _gap_mad = _estimate_lane_gap(lane)
            regular_gaps.append(regular_gap)
            for line, _bbox in lane.lines:
                lane_by_source[line.source_index] = lane
        if not lane_by_source:
            continue

        page_regular_gap = statistics.median(regular_gaps) if regular_gaps else 0.2 * body_height
        physical_gaps = _build_physical_title_gap_map(line_geometry)
        local_container_bboxes = [_rotate_bbox_to_upright(bbox, page_size, angle) for bbox in container_bboxes]
        candidates: list[tuple[_LineItem, BBox, _TextLane]] = []
        for line, bbox in line_geometry:
            lane = lane_by_source.get(line.source_index)
            if lane is None:
                continue
            line_height = _line_effective_height(line, bbox)
            lane_width = max(0.1, lane.right - lane.left)
            if not 0.9 <= line_height / body_height <= 1.1:
                continue
            if bbox[2] - bbox[0] > 0.22 * lane_width:
                continue
            if abs(bbox[0] - lane.left) > 0.75 * body_height:
                continue
            previous_rows = [
                (other, bounds)
                for other, bounds in lane.lines
                if bounds[1] < bbox[1]
                and other.font_signature == line.font_signature
                and bounds[2] - bounds[0] >= 0.75 * lane_width
            ]
            if previous_rows:
                previous, bounds = max(previous_rows, key=lambda item: item[1][1])
                if bbox[1] - bounds[3] <= 0.5 * body_height and not previous.paragraph_terminal:
                    continue
            if _line_inside_visual_container(bbox, local_container_bboxes):
                continue

            followers = _body_height_section_followers(
                line,
                bbox,
                line_geometry,
                lane_by_source,
                body_height,
            )
            if len(followers) < 3:
                continue
            gap_above = physical_gaps.get(line.source_index, (None, None))[0]
            if gap_above is not None and gap_above <= 0.25 * body_height:
                continue
            starts_body = gap_above is None and bbox[1] <= 0.2 * local_page_height
            has_extra_gap = gap_above is not None and gap_above - page_regular_gap >= 0.75 * body_height
            has_full_width_follower = any(
                follower_bbox[2] - follower_bbox[0]
                >= 0.75
                * max(
                    0.1,
                    lane_by_source[follower_line.source_index].right - lane_by_source[follower_line.source_index].left,
                )
                for follower_line, follower_bbox in followers
                if follower_line.source_index in lane_by_source
            )
            if starts_body or has_extra_gap or has_full_width_follower:
                candidates.append((line, bbox, lane))

        for line, bbox, _lane in candidates:
            compatible_count = sum(
                1
                for peer_line, peer_bbox, _peer_lane in candidates
                if abs(peer_bbox[0] - bbox[0]) <= body_height
                and 0.9 <= _line_effective_height(peer_line, peer_bbox) / _line_effective_height(line, bbox) <= 1.1
                and _title_fonts_compatible(line, peer_line)
            )
            if compatible_count >= 2:
                # 只标记结构锚点本身，避免普通正文被标题邻行扩展再次吞入。
                line.semantic_type = "paragraph_title"


def _body_height_section_followers(
    candidate_line: _LineItem,
    candidate_bbox: BBox,
    line_geometry: list[tuple[_LineItem, BBox]],
    lane_by_source: dict[int, _TextLane],
    body_height: float,
) -> list[tuple[_LineItem, BBox]]:
    """返回短标题后方同锚点、同正文尺度且行距稳定的前三行。"""

    followers: list[tuple[_LineItem, BBox]] = []
    previous_top = candidate_bbox[1]
    for line, bbox in line_geometry:
        if line is candidate_line or bbox[1] <= candidate_bbox[1] + 0.4 * body_height:
            continue
        if not (candidate_bbox[0] - 0.75 * body_height <= bbox[0] <= candidate_bbox[0] + 1.5 * body_height):
            continue
        top_pitch = bbox[1] - previous_top
        if top_pitch < 0.5 * body_height:
            continue
        if top_pitch > 1.8 * body_height:
            break
        if not 0.9 <= _line_effective_height(line, bbox) / body_height <= 1.1:
            break
        if line.source_index not in lane_by_source:
            break
        followers.append((line, bbox))
        previous_top = bbox[1]
        if len(followers) == 3:
            break
    return followers


__all__ = [
    "_normalized_section_title_text",
    "_is_plausible_section_number",
    "_section_title_has_body_followers",
    "_classify_explicit_section_titles",
    "_classify_document_structural_titles",
    "_promote_noninitial_document_title_band",
    "_canonical_title_style_key",
    "_classify_document_structural_title_candidates",
    "_collect_legacy_paragraph_title_sources",
    "_classify_inline_typography_reset_titles",
    "_classify_body_height_section_titles",
    "_body_height_section_followers",
]


def _following_stable_body_bounds(rows: list[_LineItem], start: int, em: float) -> BBox | None:
    """从后继正文的重复左右缘估计局部栏宽，首行和列表缩进不决定栏中心。"""
    members: list[list[_LineItem]] = []
    for line in rows[start : start + 20]:
        if line.semantic_type is not None or line.paragraph_group is not None or line.title_suppressed:
            break
        if members and line.font_signature != members[0][0].font_signature:
            break
        if members and abs(_bbox_center_y(line.bbox) - _bbox_center_y(members[-1][0].bbox)) < 0.2 * em:
            members[-1].append(line)
            continue
        if members and (line.bbox[1] - max(item.bbox[3] for item in members[-1]) > 0.8 * em or len(members) >= 5):
            break
        members.append([line])
    if len(members) < 2:
        return None
    bounds = [_bbox_union_many([line.bbox for line in group]) for group in members]
    widths = [bbox[2] - bbox[0] for bbox in bounds]
    wide = [bbox for bbox, width in zip(bounds, widths) if width >= 0.8 * max(widths)]
    if len(wide) < 2:
        return None
    return (
        statistics.median(bbox[0] for bbox in wide),
        bounds[0][1],
        statistics.median(bbox[2] for bbox in wide),
        bounds[-1][3],
    )


def _classify_recurrent_unknown_weight_titles(pages: list[_PreparedPage]) -> None:
    """字重不可用时，以跨页独立短行组和字体切换恢复标题，普通连续正文样式不能成为种子。"""
    candidates = []
    body_styles = set()
    for page_index, page in enumerate(pages):
        if page_index == 0:
            continue
        width, height = page.page_size
        containers = [block["bbox"] for block in page.fixed_blocks]
        layout = build_layout_evidence(page.remaining_lines, page.page_size, barriers=containers)
        for lane in layout.lanes:
            left, right = layout.corridor((lane.left, 0, lane.right, 1))
            rows = sorted(
                [
                    line
                    for line in page.remaining_lines
                    if line.angle == 0
                    and left <= _bbox_center_x(line.bbox) < right
                    and line.bbox[1] >= 0.06 * height
                    and line.bbox[3] <= 0.93 * height
                    and line.semantic_type in {None, "paragraph_title"}
                    and not any(_line_inside_visual_container(line.bbox, [bbox]) for bbox in containers)
                ],
                key=lambda line: (line.bbox[1], line.bbox[0]),
            )
            if len(rows) < 4:
                continue
            em = statistics.median(_line_effective_height(line, line.bbox) for line in rows)
            i = 0
            while i < len(rows):
                first = rows[i]
                j = i + 1
                while (
                    j < len(rows)
                    and rows[j].font_signature == first.font_signature
                    and (
                        abs(rows[j].bbox[0] - first.bbox[0]) <= 0.5 * em
                        or abs(_bbox_center_x(rows[j].bbox) - _bbox_center_x(first.bbox)) <= 0.5 * em
                    )
                    and -0.25 * em <= rows[j].bbox[1] - rows[j - 1].bbox[3] <= 0.8 * em
                ):
                    j += 1
                group = rows[i:j]
                signature = first.font_signature
                if len(group) >= 4:
                    body_styles.add(signature)
                words = " ".join(line.text for line in group).split()
                heading_text = (
                    2 <= len(words) <= 30
                    or len(words) == 1
                    and first.text.strip().isalpha()
                    and 5 <= len(first.text.strip()) <= 25
                )
                body_bounds = _following_stable_body_bounds(rows, j, em)
                if (
                    signature is not None
                    and len(group) <= 3
                    and j < len(rows)
                    and heading_text
                    and all(line.dominant_font_weight is None and line.font_coverage >= 0.75 for line in group)
                    and not any(line.paragraph_terminal for line in group)
                    and rows[j].font_signature != signature
                    and rows[j].bbox[2] - rows[j].bbox[0] >= 0.25 * (right - left)
                    and len(re.findall(r"[A-Za-z]{2,}", rows[j].text)) >= 4
                    and (
                        abs(rows[j].bbox[0] - first.bbox[0]) <= em
                        or abs(_bbox_center_x(first.bbox) - (lane.left + lane.right) / 2) <= em
                        or body_bounds is not None
                        and abs(_bbox_center_x(_bbox_union_many([line.bbox for line in group])) - _bbox_center_x(body_bounds))
                        <= 0.5 * em
                    )
                    and -0.1 * em <= rows[j].bbox[1] - group[-1].bbox[3] <= 1.5 * em
                    and (i == 0 or first.bbox[1] - rows[i - 1].bbox[3] >= 0.3 * em)
                ):
                    candidates.append((page_index, signature, group))
                i = j
    by_style = {}
    for index, signature, group in candidates:
        by_style.setdefault(signature, []).append((index, group))
    for signature, groups in by_style.items():
        if signature in body_styles or len(groups) < 3 or len({index for index, _ in groups}) < 2:
            continue
        for _, group in groups:
            for line in group:
                line.semantic_type = "paragraph_title"
                line.structural_title = True
