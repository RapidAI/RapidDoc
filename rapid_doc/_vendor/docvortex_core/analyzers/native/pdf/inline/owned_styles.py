"""仅让新鲜的同源页面证据使用快照样式内核，公开可变输入仍走原入口。"""

import math
from functools import lru_cache

from ....._compute_backend import get_native
from ..models import _AxisLine, _LineItem
from . import detection as reference


@lru_cache(maxsize=4096)
def _snapshot_text_properties(text):
    """每个 Unicode 文本值只计算一次宿主解释器的匹配、可见和项目符号属性。"""
    fragment = reference._normalize_match_fragment(text)
    return (
        fragment,
        text.isprintable() and not text.isspace(),
        text.isspace(),
        bool(fragment) and all(char in reference._PDF_LIST_MARKER_CHARS for char in fragment),
    )


def _snapshot_font_bold(name, flags, weight):
    """复用原正则和字体判断，字体大小不同但名称/flags/字重相同的字符共享结果。"""
    name = reference._PDF_FONT_SUBSET_PREFIX_RE.sub("", name)
    return "bold" in reference._font_styles_from_metadata(name, flags, weight if weight > 0 else None)


def _plain_box(box):
    """私有同源数据只准入有限普通坐标，特殊对象保留参考转换与错误语义。"""
    return (
        type(box) in (tuple, list)
        and len(box) == 4
        and all(type(v) is float and math.isfinite(v) and abs(v) <= 1e100 for v in box)
    )


def detect_owned_style_lines(owner, lines, drawings, identities):
    """仅传行框、成员索引和绘图线，最终才构造公开样式行；能力不适用返回 None。"""
    native = get_native()
    if native is None or type(owner) is not native.NativeTextSnapshot or identities is None:
        return None
    rows = []
    for line in lines:
        if type(line) is not _LineItem or type(line.angle) is not int:
            return None
        if line.angle % 360:
            continue
        if type(line.source_index) is not int or not -(2**63) <= line.source_index < 2**63 or not _plain_box(line.bbox):
            return None
        if type(line.chars) is not list:
            return None
        members = [identities.get(id(char)) for char in line.chars]
        if any(index is None for index in members):
            return None
        rows.append((line.bbox, line.source_index, members))
    rules = []
    for drawing in drawings:
        if type(drawing) is not _AxisLine or type(drawing.orientation) is not str:
            return None
        if drawing.orientation != "horizontal":
            continue
        if not _plain_box(drawing.bbox) or type(drawing.width) not in (float, int) or not math.isfinite(drawing.width):
            return None
        rules.append((drawing.bbox, drawing.width))
    thresholds = (
        reference.TEXT_DECORATION_MIN_LENGTH_HEIGHT_RATIO,
        reference.UNDERLINE_BOTTOM_TOLERANCE_HEIGHT_RATIO,
        reference.STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO,
        reference.TEXT_DECORATION_MAX_WIDTH_HEIGHT_RATIO,
        reference.TEXT_DECORATION_MIN_TEXT_COVERAGE_RATIO,
        reference.TEXT_DECORATION_ENDPOINT_TOLERANCE_HEIGHT_RATIO,
        reference.UNDERLINE_FRACTION_MAX_GAP_HEIGHT_RATIO,
        reference.UNDERLINE_FRACTION_MIN_LOWER_LINE_COVERAGE,
    )
    result = owner.detect_style_lines(
        rows, rules, _snapshot_text_properties, _snapshot_font_bold, thresholds, reference.PDF_BOLD_MIN_COMPARABLE_CHAR_COUNT
    )
    if result is None:
        return None
    styles = {
        mask: tuple(style for bit, style in ((1, "bold"), (2, "underline"), (4, "strikethrough")) if mask & bit)
        for mask in range(8)
    }
    return [
        reference.PDFTextStyleLine(
            tuple(box),
            text,
            tuple(reference.PDFTextStyleRange(start, end, styles[mask]) for start, end, mask in ranges),
            source,
        )
        for box, text, ranges, source in result
    ]
