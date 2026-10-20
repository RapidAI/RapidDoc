"""共享图注文字证据；独立图表题与正文引用句使用同一判定，不跨领域导入。"""

from __future__ import annotations

import re
import unicodedata


_IDENTIFIER_PATTERN = (
    r"(?:"
    r"(?:[A-Z]\s*[.-]?\s*)?\d+(?:\s*[-./–—]\s*[A-Z]?\d+)*[A-Z]?"
    r"|[IVXLCDM]+"
    r"|[零〇一二三四五六七八九十百千两]+"
    r")"
)
_ENGLISH_CAPTION_RE = re.compile(
    rf"^\s*(?:fig(?:ure)?|tab(?:le)?|alg(?:orithm)?|listing|chart|scheme|diagram)(?=[.\s0-9])\.?(?:\s*)"
    rf"(?P<identifier>{_IDENTIFIER_PATTERN})(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_CHINESE_CAPTION_RE = re.compile(
    rf"^\s*(?:程序清单|图表|表格|算法|图|表)\s*"
    rf"(?P<identifier>{_IDENTIFIER_PATTERN})(?P<tail>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_ENGLISH_REFERENCE_TAIL_RE = re.compile(
    r"^(?:[,;]|and\b|or\b|&|shows?\b|illustrates?\b|presents?\b|depicts?\b|"
    r"demonstrates?\b|lists?\b|summari[sz]es?\b|reports?\b|compares?\b|"
    r"provides?\b|is\b|are\b|was\b|were\b|has\b|have\b|can\b)",
    re.IGNORECASE,
)
_CHINESE_REFERENCE_TAIL_RE = re.compile(
    r"^(?:[,;，；、]|和|及|与|为|是|展示|显示|给出|说明|列出|分别|可见|表明|描述|呈现|汇总|所示|中(?:的|为)?)",
)
_SUBFIGURE_REFERENCE_TAIL_RE = re.compile(
    r"^[（(][^）)]+[）)]\s*(?:[、,，]|和|及|与)",
)


def _normalize_annotation_text(text: str) -> str:
    """统一全角字符和兼容罗马数字，保留原文仅供规则判断。"""

    return unicodedata.normalize("NFKC", text).strip()


def _caption_tail_is_reference(tail: str) -> bool:
    """排除编号后紧接叙述谓语、并列编号或正文连接词的引用句。"""

    stripped = tail.lstrip()
    if not stripped:
        return False
    if _SUBFIGURE_REFERENCE_TAIL_RE.match(stripped):
        return True
    if stripped[0] in ".:：-–—()[]（）":
        return False
    if _ENGLISH_REFERENCE_TAIL_RE.match(stripped) or _CHINESE_REFERENCE_TAIL_RE.match(stripped):
        return True
    return stripped[0].isascii() and stripped[0].isalpha() and stripped[0].islower()


def _is_strong_caption_text(text: str) -> bool:
    """判断文本是否以带编号的中英文强图表标题标记开头。"""

    normalized = _normalize_annotation_text(text)
    match = _ENGLISH_CAPTION_RE.match(normalized) or _CHINESE_CAPTION_RE.match(normalized)
    return match is not None and not _caption_tail_is_reference(match.group("tail"))


def _caption_identifier(text: str) -> str | None:
    """返回中英文强图表题共用的规范化编号。"""

    normalized = _normalize_annotation_text(text)
    match = _ENGLISH_CAPTION_RE.match(normalized) or _CHINESE_CAPTION_RE.match(normalized)
    return (
        unicodedata.normalize(
            "NFKC",
            match.group("identifier"),
        ).casefold()
        if match is not None
        else None
    )
