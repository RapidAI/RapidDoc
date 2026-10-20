"""PDF 模型输出的可见文字清洗；原始字符、布局证据及其它输入格式不在此处理。"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from ..foundation._text import full_to_half_exclude_marks
from ..schema import RAW_CAPTION, RAW_FOOTNOTE, RAW_PHONETIC, BlockType

_FULLWIDTH_MODEL_TEXT = re.compile("[Ａ-Ｚａ-ｚ０-９：．／＼－＿％＋＝＠＃＆＊]")
_PDF_SYMBOL_TRANSLATION = str.maketrans(
    {
        "：": ":",
        "．": ".",
        "／": "/",
        "＼": "\\",
        "－": "-",
        "＿": "_",
        "％": "%",
        "＋": "+",
        "＝": "=",
        "＠": "@",
        "＃": "#",
        "＆": "&",
        "＊": "*",
    }
)
_FORMULA_OPENING = re.compile(r"\\[\(\[]")
_NATURAL_LANGUAGE_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.ASIDE_TEXT,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
        BlockType.REF_TEXT,
        BlockType.LIST,
        BlockType.INDEX,
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.CHART_CAPTION,
        BlockType.CHART_FOOTNOTE,
        BlockType.CODE_CAPTION,
        BlockType.CODE_FOOTNOTE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
        RAW_PHONETIC,
    }
)
_OPAQUE_HTML_TAGS = frozenset(
    {"eq", "math", "pre", "code", "script", "style", "svg", "template", "textarea", "object", "embed", "canvas", "iframe"}
)


def _normalize_plain_text(content: str) -> str:
    """按一对一码点映射转换英数和 PDF 符号白名单，不改变通用英数工具的契约。"""
    return full_to_half_exclude_marks(content).translate(_PDF_SYMBOL_TRANSLATION)


def _normalize_text(content: str) -> str:
    """转换公式范围之外的英数及白名单符号；未闭合公式保护到逻辑文字段末尾。"""
    if not _FULLWIDTH_MODEL_TEXT.search(content):
        return content
    parts: list[str] = []
    cursor = 0
    while match := _FORMULA_OPENING.search(content, cursor):
        parts.append(_normalize_plain_text(content[cursor : match.start()]))
        closing = r"\)" if match.group() == r"\(" else r"\]"
        end = content.find(closing, match.end())
        if end < 0:
            parts.append(content[match.start() :])
            return "".join(parts)
        cursor = end + len(closing)
        parts.append(content[match.start() : cursor])
    parts.append(_normalize_plain_text(content[cursor:]))
    return "".join(parts)


def _normalize_parts(parts: Sequence[str]) -> list[str]:
    """先识别跨节点公式，再按原长度分回节点，保持样式与链接的边界。"""
    source = "".join(parts)
    normalized = _normalize_text(source)
    if source == normalized:
        return list(parts)
    # 英数及符号映射始终一对一；不要使用可能扩展字符的整体 Unicode 规范化。
    result: list[str] = []
    offset = 0
    for part in parts:
        result.append(normalized[offset : offset + len(part)])
        offset += len(part)
    return result


def _span_parts(spans: list[Any]) -> Iterator[tuple[dict[str, Any] | None, str]]:
    """递归遍历可见 TextSpan；不读取 URL，并用不可见屏障隔离公式和代码载荷。"""
    for span in spans:
        if not isinstance(span, dict):
            yield None, "\0"
            continue
        content = span.get("content")
        if span.get("type") == "text" and isinstance(content, str):
            yield span, content
        elif span.get("type") == "hyperlink" and isinstance(content, list):
            yield from _span_parts(content)
        else:
            yield None, "\0"


def _normalize_spans(spans: list[Any]) -> None:
    """只更新文字叶子的内容，既不重建 Span，也不合并相邻等样式片段。"""
    entries = list(_span_parts(spans))
    normalized = _normalize_parts([text for _span, text in entries])
    for (span, original), replacement in zip(entries, normalized):
        if span is not None and replacement != original:
            span["content"] = replacement


def _is_opaque_html_node(node: Any) -> bool:
    """识别现有 HTML 公式、代码及非文本载体，保持它们的内容与属性原样。"""
    name = str(node.name).split(":")[-1].lower()
    return (
        name in _OPAQUE_HTML_TAGS
        or node.get("data-block-type") in {"equation", "code", "code_body", "algorithm", "algorithm_body"}
        or node.has_attr("data-docvortex-latex")
        or node.has_attr("data-formula-display")
        or "docvortex-math" in (node.get("class") or [])
    )


def _normalize_table(markup: str) -> str:
    """只修改单元格的可见文本节点；没有实际变化时保留原 HTML 字节表示。"""
    if not _FULLWIDTH_MODEL_TEXT.search(markup) and "&#" not in markup:
        return markup
    # 与现有表格处理保持同一 HTML 解析器，公开模块导入不触发 HTML 依赖。
    from bs4 import BeautifulSoup, NavigableString, Tag

    soup = BeautifulSoup(markup, "html.parser")
    changed = False

    def cell_parts(node: Any) -> Iterator[tuple[NavigableString | None, str]]:
        """样式标签保持透明，嵌套表格由其自身单元格处理，换行和载荷不能拼出定界符。"""
        if type(node) is NavigableString:
            yield node, str(node)
        elif isinstance(node, Tag):
            if _is_opaque_html_node(node) or node.name in {"table", "br", "hr", "img"}:
                yield None, "\0"
            else:
                for child in node.children:
                    yield from cell_parts(child)

    for cell in soup.find_all(["td", "th"]):
        if any(isinstance(parent, Tag) and _is_opaque_html_node(parent) for parent in (cell, *cell.parents)):
            continue
        entries = [part for child in cell.children for part in cell_parts(child)]
        normalized = _normalize_parts([text for _node, text in entries])
        for (node, original), replacement in zip(entries, normalized):
            if node is not None and replacement != original:
                node.replace_with(NavigableString(replacement))
                changed = True
    return str(soup) if changed else markup


def normalize_pdf_model_text(model_list: list[list[dict[str, Any]]]) -> None:
    """统一 PDF 自然语言及表格可见英数与白名单符号，保留公式、代码、URL 和结构。

    应在样式、上下标和链接匹配结束后、ModelJson 构造前调用；函数幂等，不修改
    原始字符证据，也不会将字符串转换为 Span 或清理其它模型元数据。
    """
    for page in model_list:
        for block in page:
            kind, content = block.get("type"), block.get("content")
            if kind in _NATURAL_LANGUAGE_TYPES:
                if isinstance(content, str):
                    block["content"] = _normalize_text(content)
                elif isinstance(content, list):
                    _normalize_spans(content)
            elif kind in {BlockType.TABLE, BlockType.TABLE_BODY} and isinstance(content, str):
                block["content"] = _normalize_table(content)


__all__ = ["normalize_pdf_model_text"]
