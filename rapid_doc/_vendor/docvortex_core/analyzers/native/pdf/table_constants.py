"""PDF 固定阈值与模式；保留原有认领顺序与判定规则。"""

from __future__ import annotations
import re


_TABLE_CAPTION_RE = re.compile(
    r"^(?:table|tab\.?|表格?)[\s:.–—-]*(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)\b(?P<suffix>.*)$",
    re.IGNORECASE,
)


_TABLE_CONTINUATION_RE = re.compile(
    r"^(?:续\s*表(?:\s*[0-9０-９ivxlcdm一二三四五六七八九十]+)?"
    r"|(?:table|tab\.?)\s*(?:[0-9ivxlcdm]+\s*)?(?:continued|cont\.?))$",
    re.IGNORECASE,
)


_TABLE_NOTE_RE = re.compile(
    r"^(?:notes?|sources?)\b|^(?:注释?|说明)\s*[:：]?|^for\s+[*†‡]"
    r"|^[*†‡]\s*\S",
    re.IGNORECASE,
)


_AUXILIARY_TABLE_NOTE_RE = re.compile(
    r"^\s*[([{（［【]?(?P<marker>[^\s)\]}）］】.:：、]{1,3})"
    r"[)\]}）］】]?[.:：、)]*\s+(?P<body>\S.*)$"
)


_TABLE_SPLIT_NUMBER_RE = re.compile(
    r"^(?:\d+|[ivxlcdm]+|[一二三四五六七八九十]+)[.:：]?$",
    re.IGNORECASE,
)


_FILLED_GRID_MIN_PAGE_AREA_RATIO = 0.005


_FILLED_GRID_MAX_PAGE_AREA_RATIO = 0.25


_FILLED_GRID_MIN_PAGE_WIDTH_RATIO = 0.12


_FILLED_GRID_MIN_PAGE_HEIGHT_RATIO = 0.03
