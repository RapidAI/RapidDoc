"""把已对齐的链接、样式及上下标区间物化为 InlineSpan。"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .....content.spans import (
    append_equation_span,
    append_hyperlink_span,
    append_text_span,
    extend_inline_spans,
    normalize_span_dicts,
)
from .....schema import BBox
from .common import _canonical_styles
from .matching import (
    _ContentProjectionCache,
    _assign_lines_to_blocks,
    _assign_script_lines_to_blocks,
    _block_bbox_to_page_bbox,
    _filter_line_styles_for_block,
    _match_link_ranges,
    _match_script_line_ranges,
    _match_style_ranges,
    _merge_raw_link_intervals,
    _merge_style_ranges,
    _project_content_chars,
    _raw_link_intervals,
    _resolve_script_matches,
)
from .types import (
    _NATIVE_SCRIPT_TAG_RE,
    _PDF_INLINE_SPAN_BLOCK_TYPES,
    _PDF_LINK_INTERVALS_KEY,
    _PDF_STYLE_INTERVALS_KEY,
    PDF_NATIVE_SCRIPT_MARKUP_KEY,
    PDFTextLinkLine,
    PDFTextScriptLine,
    PDFTextScriptRange,
    PDFTextStyle,
    PDFTextStyleLine,
    PDFTextStyleRange,
    _NativeScriptMarkup,
    _MatchedStyleRange,
    _ProjectedChar,
    _RawStyleInterval,
)


def apply_pdf_text_links(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextLinkLine],
    page_size: tuple[float, float],
) -> None:
    """把页面 Link 几何证据写入自然语言 block，歧义时保持原文。"""

    assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    for block_index, block_lines in assignments.items():
        content = blocks[block_index].get("content")
        if not isinstance(content, str) or not content:
            continue
        projected = _project_content_chars(content)
        link_ranges = _match_link_ranges(projected, block_lines)
        if not link_ranges:
            continue
        intervals = _raw_link_intervals(content, projected, link_ranges)
        if intervals:
            intervals = _merge_raw_link_intervals(content, intervals)
            blocks[block_index][_PDF_LINK_INTERVALS_KEY] = intervals


def _append_raw_style_interval(
    intervals: list[_RawStyleInterval],
    start: int | None,
    end: int,
    styles: tuple[PDFTextStyle, ...],
) -> None:
    """向结果追加一个合法原字符串样式区间。"""

    if start is not None and start < end and styles:
        intervals.append(_RawStyleInterval(start, end, styles))


def _merge_raw_style_intervals(
    content: str,
    intervals: Sequence[_RawStyleInterval],
) -> list[_RawStyleInterval]:
    """合并原字符串中相邻且样式一致、仅由普通空白隔开的区间。"""

    merged: list[_RawStyleInterval] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.end, item.styles)):
        if interval.start >= interval.end or not interval.styles:
            continue
        if (
            merged
            and merged[-1].styles == interval.styles
            and (interval.start <= merged[-1].end or content[merged[-1].end : interval.start].isspace())
        ):
            merged[-1] = _RawStyleInterval(
                merged[-1].start,
                max(merged[-1].end, interval.end),
                merged[-1].styles,
            )
        else:
            merged.append(interval)
    return merged


def _raw_style_intervals(
    content: str,
    projected: Sequence[_ProjectedChar],
    ranges: Sequence[PDFTextStyleRange],
) -> list[_RawStyleInterval]:
    """把样式区间转换为不跨公式或已有行内标签的原字符串区间。"""

    intervals: list[_RawStyleInterval] = []
    for style_range in ranges:
        current_start: int | None = None
        current_end = 0
        current_styles: tuple[PDFTextStyle, ...] = ()
        for token in projected[style_range.start : style_range.end]:
            # Link 注解已提供语义，链接范围内的几何下划线不重复输出为文本样式。
            missing_styles = _canonical_styles(
                style
                for style in style_range.styles
                if style not in token.existing_styles and not (style == "underline" and token.inside_hyperlink)
            )
            if not missing_styles:
                _append_raw_style_interval(
                    intervals,
                    current_start,
                    current_end,
                    current_styles,
                )
                current_start = None
                current_styles = ()
                continue
            if current_start is None:
                current_start = token.raw_start
                current_end = token.raw_end
                current_styles = missing_styles
                continue
            gap = content[current_end : token.raw_start]
            if missing_styles == current_styles and (token.raw_start <= current_end or not gap or gap.isspace()):
                current_end = max(current_end, token.raw_end)
            else:
                _append_raw_style_interval(
                    intervals,
                    current_start,
                    current_end,
                    current_styles,
                )
                current_start = token.raw_start
                current_end = token.raw_end
                current_styles = missing_styles
        _append_raw_style_interval(
            intervals,
            current_start,
            current_end,
            current_styles,
        )
    return _merge_raw_style_intervals(content, intervals)


def apply_pdf_text_styles(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextStyleLine],
    page_size: tuple[float, float],
) -> None:
    """把页面字体和装饰线证据写入自然语言 block，歧义时保持原文。"""

    assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    for block_index, block_lines in assignments.items():
        block = blocks[block_index]
        block_lines = _filter_line_styles_for_block(
            block_lines,
            block.get("type"),
        )
        content = block.get("content")
        if not isinstance(content, str) or not content or not any(line.style_ranges for line in block_lines):
            continue
        projected = _project_content_chars(content)
        style_ranges = _match_style_ranges(projected, block_lines)
        if not style_ranges:
            continue
        intervals = _raw_style_intervals(content, projected, style_ranges)
        if intervals:
            existing = block.get(_PDF_STYLE_INTERVALS_KEY, [])
            block[_PDF_STYLE_INTERVALS_KEY] = _merge_raw_style_intervals(
                content,
                [
                    *(interval for interval in existing if isinstance(interval, _RawStyleInterval)),
                    *intervals,
                ],
            )


def _script_range_hits_late_formula_region(
    script_range: PDFTextScriptRange,
    regions: list[BBox],
) -> bool:
    """判断非公式候选是否落入后续文本块恢复出的行内数学区域。"""
    if script_range.formula_region:
        return False
    center_x = (script_range.bbox[0] + script_range.bbox[2]) / 2
    center_y = (script_range.bbox[1] + script_range.bbox[3]) / 2
    return any(region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3] for region in regions)


def _resolve_inline_styles(styles: Iterable[str], native_scripts: Iterable[str] = ()) -> tuple[PDFTextStyle, ...]:
    """原生标签优先确定脚本方向；无来源的双重脚本同时移除，保留其它样式。"""
    selected = set(styles)
    native = set(native_scripts)
    scripts = {"superscript", "subscript"}
    if native:
        selected.difference_update(scripts)
        selected.update(native)
    if scripts <= selected:
        selected.difference_update(scripts)
    return _canonical_styles(selected)


def _effective_script_intervals(
    content: str,
    intervals: Sequence[_RawStyleInterval],
    native: _NativeScriptMarkup | None,
) -> list[_RawStyleInterval]:
    """按最终 Span 的同一互斥规则计算诊断可认领的原文区间，排除标签与公式。"""
    script_intervals = native.style_intervals if native is not None else ()
    excluded = list(native.marker_ranges if native is not None else ())
    excluded.extend(match.span() for match in re.finditer(r"\\\(.*?\\\)", content, re.DOTALL))
    boundaries = {0, len(content)}
    for interval in intervals:
        boundaries.update((interval.start, interval.end))
    for start, end, _style in script_intervals:
        boundaries.update((start, end))
    for start, end in excluded:
        boundaries.update((start, end))
    ordered = sorted(value for value in boundaries if 0 <= value <= len(content))
    output: list[_RawStyleInterval] = []
    for start, end in zip(ordered, ordered[1:]):
        if any(left <= start and end <= right for left, right in excluded):
            continue
        styles = _resolve_inline_styles(
            (style for interval in intervals if interval.start <= start and end <= interval.end for style in interval.styles),
            (style for left, right, style in script_intervals if left <= start and end <= right),
        )
        scripts = tuple(style for style in styles if style in {"superscript", "subscript"})
        if scripts:
            output.append(_RawStyleInterval(start, end, scripts))
    return _merge_raw_style_intervals(content, output)


def _record_materialized_script_ranges(
    content: str,
    projected: Sequence[_ProjectedChar],
    matches: Sequence[_MatchedStyleRange],
    effective_intervals: Sequence[_RawStyleInterval],
    block_index: int,
    line: PDFTextScriptLine,
    script_ranges: Sequence[PDFTextScriptRange],
    output: list[dict[str, Any]],
) -> None:
    """仅记录胜出证据实际输出的子区间，源偏移随冲突裁剪同步更新。"""
    for script_range in script_ranges:
        for match in matches:
            if (
                match.line.source_index != line.source_index
                or match.line.bbox != line.bbox
                or match.line.text != line.text
                or match.source_range.start < script_range.start
                or match.source_range.end > script_range.end
                or script_range.style not in match.source_range.styles
            ):
                continue
            for interval in effective_intervals:
                if script_range.style not in interval.styles:
                    continue
                indices = [
                    index
                    for index in range(match.start, match.end)
                    if interval.start <= projected[index].raw_start and projected[index].raw_end <= interval.end
                ]
                if not indices:
                    continue
                raw_start, raw_end = projected[indices[0]].raw_start, projected[indices[-1]].raw_end
                source_start = match.source_range.start + indices[0] - match.start
                source_end = match.source_range.start + indices[-1] - match.start + 1
                record = {
                    "block_index": block_index,
                    "raw_start": raw_start,
                    "raw_end": raw_end,
                    "source_index": line.source_index,
                    "range_start": source_start,
                    "range_end": source_end,
                    "role": script_range.style,
                    "text": line.text[source_start:source_end],
                    "bbox": script_range.bbox,
                    "angle": line.angle,
                    "formula_region": script_range.formula_region,
                    "stable_body_count": script_range.stable_body_count,
                }
                if record not in output:
                    output.append(record)


def apply_pdf_text_scripts(
    blocks: list[dict[str, Any]],
    lines: list[PDFTextScriptLine],
    page_size: tuple[float, float],
    *,
    materialized_diagnostics: list[dict[str, Any]] | None = None,
) -> None:
    """把 Flash 上下标 sidecar 投影到最终自然语言 block，并清理公式私有区域。"""
    # 空证据仍执行原分配流程，避免为扫描页建立不会被再次消费的缓存。
    projection_cache = _ContentProjectionCache() if lines else None
    assignments = _assign_script_lines_to_blocks(blocks, lines, page_size, projection_cache=projection_cache)
    for block_index, block_lines in assignments.items():
        block = blocks[block_index]
        regions = [
            region
            for value in block.get("_inline_math_regions", [])
            if (region := _block_bbox_to_page_bbox(value, page_size)) is not None
        ]
        projected_lines = []
        eligible_ranges: dict[int, tuple[PDFTextScriptRange, ...]] = {}
        for line in block_lines:
            retained = tuple(
                script_range
                for script_range in line.script_ranges
                if not _script_range_hits_late_formula_region(script_range, regions)
            )
            eligible_ranges[id(line)] = retained
            ranges = tuple(
                PDFTextStyleRange(
                    script_range.start,
                    script_range.end,
                    (script_range.style,),
                )
                for script_range in retained
            )
            projected_lines.append(
                PDFTextStyleLine(
                    bbox=line.bbox,
                    text=line.text,
                    style_ranges=ranges,
                    source_index=line.source_index,
                )
            )
        projected_lines = _filter_line_styles_for_block(projected_lines, block.get("type"))
        projected_lines.sort(
            key=lambda line: (
                line.source_index,
                line.bbox[1],
                line.bbox[0],
                line.text,
                tuple((item.start, item.end, item.styles) for item in line.style_ranges),
            )
        )
        content = block.get("content")
        if not isinstance(content, str) or not content:
            continue
        projected = (
            projection_cache.project(block, content) if projection_cache is not None else _project_content_chars(content)
        )
        matches: list[_MatchedStyleRange] = []
        _match_style_ranges(projected, projected_lines, matches=matches)
        for line_index, projected_line in enumerate(projected_lines):
            next_line = projected_lines[line_index + 1] if line_index + 1 < len(projected_lines) else None
            _match_script_line_ranges(projected, projected_line, matches=matches, next_line=next_line)
        retained_matches = _resolve_script_matches(matches)
        combined_ranges = _merge_style_ranges(
            [PDFTextStyleRange(match.start, match.end, match.source_range.styles) for match in retained_matches]
        )
        intervals = _raw_style_intervals(content, projected, combined_ranges)
        if intervals:
            existing = block.get(_PDF_STYLE_INTERVALS_KEY, [])
            block[_PDF_STYLE_INTERVALS_KEY] = _merge_raw_style_intervals(
                content,
                [
                    *(interval for interval in existing if isinstance(interval, _RawStyleInterval)),
                    *intervals,
                ],
            )
        if materialized_diagnostics is not None:
            native = _parse_native_script_markup(content) if block.get(PDF_NATIVE_SCRIPT_MARKUP_KEY) is True else None
            effective_intervals = _effective_script_intervals(content, block.get(_PDF_STYLE_INTERVALS_KEY, []), native)
            for line in block_lines:
                _record_materialized_script_ranges(
                    content,
                    projected,
                    retained_matches,
                    effective_intervals,
                    block_index,
                    line,
                    eligible_ranges.get(id(line), ()),
                    materialized_diagnostics,
                )
    for block in blocks:
        block.pop("_inline_math_regions", None)


def _parse_native_script_markup(content: str) -> _NativeScriptMarkup | None:
    """严格解析 detector-owned 平坦 sup/sub 标签；畸形或嵌套结构返回 None。"""
    marker_ranges: list[tuple[int, int]] = []
    style_intervals: list[tuple[int, int, str]] = []
    active: tuple[str, int] | None = None
    for match in _NATIVE_SCRIPT_TAG_RE.finditer(content):
        marker_ranges.append((match.start(), match.end()))
        style = "superscript" if match.group("tag") == "sup" else "subscript"
        if match.group("closing") is None:
            if active is not None:
                return None
            active = (style, match.end())
            continue
        if active is None or active[0] != style:
            return None
        if active[1] < match.start():
            style_intervals.append((active[1], match.start(), style))
        active = None
    if active is not None or not marker_ranges:
        return None
    return _NativeScriptMarkup(
        marker_ranges=tuple(marker_ranges),
        style_intervals=tuple(style_intervals),
    )


def materialize_pdf_inline_spans(blocks: list[dict[str, Any]]) -> None:
    """把 PDF 原文、样式区间、链接区间和行内公式一次性物化为 Span。"""
    formula_pattern = re.compile(r"\\\((?P<latex>.*?)\\\)", re.DOTALL)
    for block in blocks:
        owns_native_script_markup = block.pop(PDF_NATIVE_SCRIPT_MARKUP_KEY, False) is True
        if block.get("type") not in _PDF_INLINE_SPAN_BLOCK_TYPES:
            continue
        content = block.get("content")
        link_intervals = block.pop(_PDF_LINK_INTERVALS_KEY, [])
        style_intervals = block.pop(_PDF_STYLE_INTERVALS_KEY, [])
        if not isinstance(content, str):
            continue
        native_scripts = _parse_native_script_markup(content) if owns_native_script_markup else None
        marker_ranges = native_scripts.marker_ranges if native_scripts is not None else ()
        script_intervals = native_scripts.style_intervals if native_scripts is not None else ()
        formulas = list(formula_pattern.finditer(content))
        boundaries = {0, len(content)}
        for interval in [*link_intervals, *style_intervals]:
            boundaries.update((interval.start, interval.end))
        for start, end in marker_ranges:
            boundaries.update((start, end))
        for start, end, _style in script_intervals:
            boundaries.update((start, end))
        for formula in formulas:
            boundaries.update((formula.start(), formula.end()))
        ordered = sorted(value for value in boundaries if 0 <= value <= len(content))
        spans: list[dict[str, Any]] = []
        for start, end in zip(ordered, ordered[1:]):
            if start >= end:
                continue
            if any(marker_start <= start and end <= marker_end for marker_start, marker_end in marker_ranges):
                continue
            formula = next((item for item in formulas if item.start() == start and item.end() == end), None)
            if formula is not None:
                append_equation_span(spans, formula.group("latex"))
                continue
            text = content[start:end]
            if not text:
                continue
            styles = _canonical_styles(
                style
                for interval in style_intervals
                if interval.start <= start and end <= interval.end
                for style in interval.styles
            )
            script_styles = tuple(
                style
                for interval_start, interval_end, style in script_intervals
                if interval_start <= start and end <= interval_end
            )
            combined_styles = _resolve_inline_styles(styles, script_styles)
            link = next(
                (interval for interval in link_intervals if interval.start <= start and end <= interval.end),
                None,
            )
            if link is None:
                if combined_styles and text.strip():
                    leading_length = len(text) - len(text.lstrip())
                    trailing_length = len(text) - len(text.rstrip())
                    core_end = len(text) - trailing_length if trailing_length else len(text)
                    append_text_span(spans, text[:leading_length])
                    append_text_span(spans, text[leading_length:core_end], combined_styles)
                    append_text_span(spans, text[core_end:])
                else:
                    append_text_span(spans, text, combined_styles)
                continue
            children: list[dict[str, Any]] = []
            append_text_span(children, text, (style for style in combined_styles if style != "underline"))
            if spans and spans[-1].get("type") == "hyperlink" and spans[-1].get("url") == link.target:
                existing = spans[-1].get("content")
                if isinstance(existing, list):
                    extend_inline_spans(existing, children)
                    continue
            append_hyperlink_span(spans, children, link.target)
        block["content"] = normalize_span_dicts(spans)


__all__ = [
    "apply_pdf_text_links",
    "_append_raw_style_interval",
    "_merge_raw_style_intervals",
    "_raw_style_intervals",
    "apply_pdf_text_styles",
    "_script_range_hits_late_formula_region",
    "_record_materialized_script_ranges",
    "apply_pdf_text_scripts",
    "_parse_native_script_markup",
    "materialize_pdf_inline_spans",
]


def apply_pdf_inline_evidence(
    blocks: list[dict[str, Any]],
    link_lines: list[PDFTextLinkLine],
    style_lines: list[PDFTextStyleLine],
    script_lines: list[PDFTextScriptLine],
    page_size: tuple[float, float],
    *,
    materialized_diagnostics: list[dict[str, Any]] | None = None,
) -> None:
    """在调用方已有页面准备之后统一执行行内语义物化顺序。"""
    apply_pdf_text_links(blocks, link_lines, page_size)
    apply_pdf_text_styles(blocks, style_lines, page_size)
    apply_pdf_text_scripts(blocks, script_lines, page_size, materialized_diagnostics=materialized_diagnostics)
    materialize_pdf_inline_spans(blocks)
