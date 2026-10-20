"""自然语言物化时修复可靠词界及中文生成空格，不改写源字符。"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from ....schema import BBox
from ._contracts import Char


def _han_letter(text: str) -> bool:
    """只接受单个汉字及其扩展、兼容字形，不将英文或符号当作中文文字。"""
    return len(text) == 1 and (
        0x3400 <= ord(text) <= 0x4DBF
        or 0x4E00 <= ord(text) <= 0x9FFF
        or 0xF900 <= ord(text) <= 0xFAFF
        or 0x20000 <= ord(text) <= 0x323AF
    )


_CHINESE_PUNCTUATION = frozenset("、。，；：？！“”‘’（）【】《》〈〉「」『』〔〕［］｛｝—…·")
_LIGATURE_EXPANSIONS = frozenset({"fi", "fl", "ff", "ffi", "ffl", "ft", "st"})
_LIGATURE_GLYPHS = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}


def is_ligature_expansion_space(
    chars: Sequence[Char],
    index: int,
    *,
    tight_bboxes: Mapping[int, BBox] | None = None,
    origins: Mapping[int, tuple[float, float]] | None = None,
) -> bool:
    """仅忽略连字展开字形内部的空格，保留真实词界和原始字符记录。"""
    if not 0 < index < len(chars) - 1 or chars[index].get("char") != " ":
        return False
    right = chars[index + 1]
    right_text = str(right.get("char", ""))
    if len(right_text) != 1 or not right_text.isascii() or not right_text.isalpha():
        return False
    try:
        for count in range(1, min(3, index) + 1):
            glyph = chars[index - count : index]
            text = "".join(str(char.get("char", "")) for char in glyph)
            if _LIGATURE_GLYPHS.get(text, text) not in _LIGATURE_EXPANSIONS:
                continue
            first = glyph[0]
            records = [*glyph, chars[index], right]
            indices = [char["char_idx"] for char in records]
            if any(b != a + 1 for a, b in zip(indices, indices[1:])):
                continue
            object_id = first.get("text_object_id")
            if object_id is None or any(
                char.get("text_object_id") != object_id or char.get("font") != first.get("font") for char in records
            ):
                continue
            degrees = math.degrees(float(first["writing_angle"])) % 360
            angle = int(round(degrees / 90) * 90) % 360
            if any(
                not math.isfinite(float(char["writing_angle"]))
                or abs((math.degrees(float(char["writing_angle"])) - angle + 180) % 360 - 180) > 0.1
                for char in records
            ):
                continue
            boxes = [(tight_bboxes or {}).get(char["char_idx"], char.get("tight_bbox")) for char in records]
            points = [(origins or {}).get(char["char_idx"], char.get("origin")) for char in records]
            if any(box is None or len(box) != 4 or not all(math.isfinite(value) for value in box) for box in boxes):
                continue
            if any(point is None or len(point) != 2 or not all(math.isfinite(value) for value in point) for point in points):
                continue
            local_boxes = [_local_box(box, angle) for box in boxes]
            local_points = [_local_box((*point, *point), angle)[:2] for point in points]
            box = local_boxes[0]
            height = box[3] - box[1]
            if min(height, box[2] - box[0]) <= 0:
                continue
            # 同一字形的展开记录必须共享墨迹框及起点，不能只凭字母组合删空格。
            if any(
                max(abs(a - b) for a, b in zip(box, other)) > 0.01 * height
                or max(abs(a - b) for a, b in zip(local_points[0], point)) > 0.01 * height
                for other, point in zip(local_boxes[1:count], local_points[1:count])
            ):
                continue
            space_origin, right_origin = local_points[-2:]
            right_box = local_boxes[-1]
            if (
                box[0] <= space_origin[0] < box[2]
                and all(abs(point[1] - local_points[0][1]) <= 0.15 * height for point in local_points[count:])
                and -0.15 * height <= right_box[0] - box[2] <= 0.15 * height
                and abs(right_origin[0] - box[2]) <= 0.15 * height
                and min(box[3], right_box[3]) - max(box[1], right_box[1]) >= 0.5 * min(height, right_box[3] - right_box[1])
                and right_box[3] > right_box[1]
            ):
                return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return False


def is_generated_cjk_space(
    left: Char,
    space: Char,
    right: Char,
    *,
    origins: dict[int, tuple[float, float]] | None = None,
) -> bool:
    """仅忽略同向同基线中文边界上 PDFium 确认生成的空格，源字符记录保持不变。"""
    if space.get("char") != " " or space.get("is_generated") is not True:
        return False
    a, b = str(left.get("char", "")), str(right.get("char", ""))
    if not (
        (_han_letter(a) and (_han_letter(b) or b in _CHINESE_PUNCTUATION)) or (a in _CHINESE_PUNCTUATION and _han_letter(b))
    ):
        return False
    li, si, ri = left.get("char_idx"), space.get("char_idx"), right.get("char_idx")
    if not all(isinstance(index, int) for index in (li, si, ri)) or si != li + 1 or ri != si + 1:
        return False
    try:
        degrees = math.degrees(float(left["writing_angle"])) % 360
        angle = int(round(degrees / 90) * 90) % 360
        difference = (math.degrees(float(right["writing_angle"])) - angle + 180) % 360 - 180
        if not math.isfinite(degrees) or not math.isfinite(difference):
            return False
        if abs((degrees - angle + 180) % 360 - 180) > 0.1 or abs(difference) > 0.1:
            return False
        boxes = [tuple(char.get("tight_bbox") or char["bbox"]) for char in (left, right)]
        if any(len(box) != 4 or not all(math.isfinite(value) for value in box) for box in boxes):
            return False
        ab, bb = (_local_box(box, angle) for box in boxes)
        ah, bh = ab[3] - ab[1], bb[3] - bb[1]
        if min(ah, bh, ab[2] - ab[0], bb[2] - bb[0]) <= 0:
            return False
        if min(ab[3], bb[3]) - max(ab[1], bb[1]) < 0.5 * min(ah, bh):
            return False
        # 较大的生成空白可能分隔姓名、表格字段或栏目，不能仅凭中文相邻删除。
        if bb[0] - ab[2] > 0.5 * max(ah, bh):
            return False
        # PDF 的源字号可能为 1，再经文字矩阵放大；容差以页面字形高度为准。
        lo = (origins or {}).get(li, left.get("origin"))
        ro = (origins or {}).get(ri, right.get("origin"))
        tolerance = 0.15 * max(ah, bh)
        if lo is not None and ro is not None:
            axis = 0 if angle in (90, 270) else 1
            return all(math.isfinite(value) for value in (*lo, *ro)) and abs(lo[axis] - ro[axis]) <= tolerance
        return abs(ab[3] - bb[3]) <= tolerance
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


@lru_cache(maxsize=2048)
def _ordinary_non_cjk(text: str) -> bool:
    """仅接受单个普通字母或十进制数字，排除中日韩文字及兼容字形。"""
    if len(text) != 1:
        return False
    code = ord(text)
    if (
        0x1100 <= code <= 0x11FF
        or 0x2E80 <= code <= 0xA4CF
        or 0xA960 <= code <= 0xA97F
        or 0xAC00 <= code <= 0xD7FF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF66 <= code <= 0xFF9F
        or 0x1AFF0 <= code <= 0x1B16F
        or 0x20000 <= code <= 0x323AF
    ):
        return False
    category = unicodedata.category(text)
    return category.startswith("L") or category == "Nd"


def _local_box(box, angle):
    """只旋转坐标轴而不平移；相邻间隙不依赖页面尺寸。"""
    x0, y0, x1, y1 = box
    if angle == 90:
        return y0, -x1, y1, -x0
    if angle == 180:
        return -x1, -y1, -x0, -y0
    if angle == 270:
        return -y1, x0, -y0, x1
    return x0, y0, x1, y1


@lru_cache(maxsize=512)
def _font_allows_spacing(name: str, flags: int) -> bool:
    """等宽字体的窄字形有天然大留白，缺少 advance 证据时不新增词界。"""
    name = name.lower()
    return not flags & 1 and not any(token in name for token in ("mono", "courier", "consolas", "menlo", "typewriter", "fixed"))


def needs_tight_space(left, right, *, tight_bboxes=None, origins=None) -> bool:
    """判定同向相邻非 CJK 字符是否具有超过四分之一字号的墨迹间隙。"""
    if not _ordinary_non_cjk(str(left.get("char", ""))) or not _ordinary_non_cjk(str(right.get("char", ""))):
        return False
    # 连续数字中的窄字形（如 11）也会产生大墨迹留白，保守保持数字原样。
    if left["char"].isdecimal() and right["char"].isdecimal():
        return False
    li, ri = left.get("char_idx"), right.get("char_idx")
    # 不跨被过滤字符、显式空白、其他 span 或容器认领的字符补词界。
    if not isinstance(li, int) or not isinstance(ri, int) or ri != li + 1:
        return False
    try:
        for char in (left, right):
            if not _font_allows_spacing(str(char["font"].get("name", "")), int(char["font"].get("flags", 0))):
                return False
        ls, rs = float(left["font"]["size"]), float(right["font"]["size"])
        if not math.isfinite(ls) or not math.isfinite(rs) or min(ls, rs) <= 0:
            return False
        em = max(ls, rs)
        # 字号显著不同通常是上下标，证据不足时不额外补空格。
        if min(ls, rs) < 0.8 * em:
            return False
        la, ra = float(left["writing_angle"]), float(right["writing_angle"])
        if not math.isfinite(la) or not math.isfinite(ra):
            return False
        degrees = math.degrees(la) % 360
        angle = int(round(degrees / 90) * 90) % 360
        if abs((degrees - angle + 180) % 360 - 180) > 0.1:
            return False
        if abs((math.degrees(ra) - angle + 180) % 360 - 180) > 0.1:
            return False
        lb = (tight_bboxes or {}).get(li, left.get("tight_bbox"))
        rb = (tight_bboxes or {}).get(ri, right.get("tight_bbox"))
        if lb is None or rb is None or len(lb) != 4 or len(rb) != 4:
            return False
        if not all(math.isfinite(v) for v in (*lb, *rb)):
            return False
        a, b = _local_box(lb, angle), _local_box(rb, angle)
        if a[2] <= a[0] or b[2] <= b[0] or a[3] <= a[1] or b[3] <= b[1]:
            return False
        # 有些 PDF 把字号写成 1 再用文字矩阵放大；不拿源字号和页面墨迹强行比较。
        if a[3] - a[1] > 1.5 * ls or b[3] - b[1] > 1.5 * rs:
            return False
        if b[0] - a[2] <= 0.25 * em:
            return False
        overlap = min(a[3], b[3]) - max(a[1], b[1])
        if overlap < 0.5 * min(a[3] - a[1], b[3] - b[1]):
            return False
        lo = (origins or {}).get(li, left.get("origin"))
        ro = (origins or {}).get(ri, right.get("origin"))
        if lo is not None and ro is not None:
            if not all(math.isfinite(v) for v in (*lo, *ro)):
                return False
            perpendicular = 0 if angle in (90, 270) else 1
            if abs(lo[perpendicular] - ro[perpendicular]) > 0.15 * em:
                return False
        elif abs(a[3] - b[3]) > 0.15 * em:
            return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


_TIGHT_CANDIDATE_REFERENCES = (needs_tight_space, _ordinary_non_cjk, _font_allows_spacing, unicodedata.category)


def _tight_space_candidates(chars_by_source, glyphs, glyph_type):
    """本次表格只预计算可能补空格的右侧源索引，未知对象与替换规则完整回退。"""
    from ...._compute_backend import get_native

    native = get_native()
    if (
        native is not None
        and hasattr(native, "tight_space_candidates")
        and (needs_tight_space, _ordinary_non_cjk, _font_allows_spacing, unicodedata.category) == _TIGHT_CANDIDATE_REFERENCES
    ):
        return native.tight_space_candidates(chars_by_source, glyphs, glyph_type, _ordinary_non_cjk)
    return None


def join_tight_text(chars, *, tight_bboxes=None, origins=None) -> str:
    """按现有成员顺序重建短行，仅在相邻可靠词界插入单个空格。"""
    # 局部成员副本接受生成器，保留已有可迭代输入约定，不修改源字符。
    chars = tuple(chars)
    parts = []
    previous = None
    for index, char in enumerate(chars):
        if 0 < index < len(chars) - 1 and (
            is_generated_cjk_space(chars[index - 1], char, chars[index + 1], origins=origins)
            or is_ligature_expansion_space(chars, index, tight_bboxes=tight_bboxes, origins=origins)
        ):
            continue
        if previous is not None and needs_tight_space(previous, char, tight_bboxes=tight_bboxes, origins=origins):
            parts.append(" ")
        parts.append(str(char.get("char", "")))
        previous = char
    return "".join(parts)
