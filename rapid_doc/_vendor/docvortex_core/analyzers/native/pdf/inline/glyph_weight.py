"""仅在字重缺失时，以同页同字符的原生字形像素补充粗体证据。"""

from collections import defaultdict
from dataclasses import replace
import math
import statistics

from PIL import Image
from ....._compute_backend import get_native

from .detection import (
    _char_font_styles,
    _coerce_bbox,
    _font_styles_from_metadata,
    _pdf_font_metadata,
    detect_pdf_text_style_lines,
)


def _glyph_mask(image, bbox, scale):
    """归一化单字符墨迹，排除过小或无墨迹区域并保持相同字符比较。"""
    crop = image.crop(tuple(math.floor(v * scale) if i < 2 else math.ceil(v * scale) for i, v in enumerate(bbox)))
    try:
        if crop.width < 6 or crop.height < 9:
            return None
        mask = crop.point(lambda value: 255 if value < 128 else 0)
        try:
            normalized = mask.resize((32, 32), Image.Resampling.NEAREST)
            try:
                pixels = normalized.get_flattened_data() if hasattr(normalized, "get_flattened_data") else normalized.getdata()
                ink = frozenset(index for index, value in enumerate(pixels) if value)
            finally:
                normalized.close()
        finally:
            mask.close()
        if not 0.08 <= len(ink) / 1024 <= 0.85:
            return None
        return ink, crop.width / crop.height
    finally:
        crop.close()


_STANDARD_GLYPH_MASK = _glyph_mask
_STANDARD_FONT_STYLES = _char_font_styles
_STANDARD_FONT_METADATA = _pdf_font_metadata
_STANDARD_GLYPH_COERCE = _coerce_bbox


def _glyph_font_metadata(char, cache):
    """单次页内按不可变原始字段复用字体转换，任何字段变化或自定义对象均重新读取。"""
    font = char.get("font")
    if _pdf_font_metadata is not _STANDARD_FONT_METADATA or type(font) is not dict:
        return _pdf_font_metadata(char)
    values = tuple(font.get(name) for name in ("name", "flags", "weight"))
    if any(type(value) not in (str, int, float, bool, type(None)) for value in values):
        return _pdf_font_metadata(char)
    key = tuple((type(value), value) for value in values)
    if key not in cache:
        if len(cache) >= 4096:
            cache.pop(next(iter(cache)))
        cache[key] = _pdf_font_metadata(char)
    return cache[key]


def _ink_count(mask):
    """位集和参考集合使用相同墨迹基数，内部表示不进入公开输出。"""
    return mask.bit_count() if type(mask) is int else len(mask)


def _batch_glyph_masks(image, groups, fonts, scale):
    """单页一次传输灰度缓冲，保留字体、字符和样本的原有插入顺序。"""
    native = get_native()
    if native is None or _glyph_mask is not _STANDARD_GLYPH_MASK or not isinstance(image, Image.Image):
        return None
    queries = [(font, text, box) for font in fonts for text, boxes in groups[font].items() for box in boxes]
    bounds = [
        tuple(math.floor(v * scale) if i < 2 else math.ceil(v * scale) for i, v in enumerate(box)) for _, _, box in queries
    ]
    if any(abs(v) > 1_000_000_000 for box in bounds for v in box) or any(box[0] > box[2] or box[1] > box[3] for box in bounds):
        return None
    values = native.glyph_masks(image.tobytes(), image.width, image.height, bounds)
    samples = defaultdict(list)
    for (font, text, _), value in zip(queries, values):
        if value is not None:
            bits, aspect = value
            samples[font, text].append((int.from_bytes(bits, "little"), aspect))
    return samples


def _glyph_font_groups_python(lines, geometry):
    """完整参考分组保持字符转换、字体过滤、采样上限和首次出现顺序。"""
    groups = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    font_cache = {}
    for line in lines:
        if line.angle != 0:
            continue
        for char in line.chars:
            text = str(char.get("char", ""))
            if len(text) != 1 or not text.isascii() or not text.isalpha():
                continue
            font = _glyph_font_metadata(char, font_cache)
            if font[1] & (1 << 6):
                continue
            # 同一次字符检查复用刚读取的字体值，替换参考函数时仍尊重调用方行为。
            styles = (
                _font_styles_from_metadata(*font) if _char_font_styles is _STANDARD_FONT_STYLES else _char_font_styles(char)
            )
            if "bold" in styles:
                continue
            bbox = geometry.tight_bboxes.get(char.get("char_idx")) or _coerce_bbox(char.get("tight_bbox"))
            if bbox is None:
                continue
            counts[font] += 1
            if len(groups[font][text]) < 3:
                groups[font][text].append(bbox)
    return groups, counts


def _glyph_font_groups_owned(lines, geometry, owner, identities):
    """仅使用身份匹配且字段未改动的页面快照，修改文字、字体或 tight 几何时完整回退。"""
    native = get_native()
    if (
        native is None
        or type(owner) is not native.NativeTextSnapshot
        or type(identities) is not dict
        or _char_font_styles is not _STANDARD_FONT_STYLES
        or _pdf_font_metadata is not _STANDARD_FONT_METADATA
        or _coerce_bbox is not _STANDARD_GLYPH_COERCE
        or type(geometry.tight_bboxes) is not dict
    ):
        return None
    from ..models import _LineItem
    from .owned_styles import _snapshot_font_bold

    values = owner.glyph_line_font_groups(lines, geometry.tight_bboxes, identities, _LineItem, _snapshot_font_bold)
    if values is None:
        return None
    groups, counts = defaultdict(lambda: defaultdict(list)), defaultdict(int)
    for name, flags, weight, count, letters in values:
        font = (name, flags, weight)
        counts[font] = count
        for text, boxes in letters:
            groups[font][text] = [tuple(box) for box in boxes]
    return groups, counts


def glyph_weight_style_lines(lines, geometry, render, *, owner=None, identities=None):
    """缺失字重且有足量对照字符时才渲染；证据不足保持原结果，兼容只提供快照的来源。"""
    grouped = _glyph_font_groups_owned(lines, geometry, owner, identities)
    groups, counts = grouped if grouped is not None else _glyph_font_groups_python(lines, geometry)
    eligible = [font for font in groups if counts[font] >= 20 and len(groups[font]) >= 8]
    if render is None or len(eligible) < 2:
        return []
    reference = max(eligible, key=lambda font: counts[font])
    candidates = [
        font
        for font in eligible
        if font != reference and font[2] is None and len(groups[font].keys() & groups[reference].keys()) >= 8
    ]
    if not candidates:
        return []
    rendered = render()
    image = rendered.pil_image.convert("L")
    try:
        batch = _batch_glyph_masks(image, groups, [reference, *candidates], rendered.scale)
        masks = {}
        for font in [reference, *candidates]:
            masks[font] = {}
            for text, boxes in groups[font].items():
                samples = (
                    batch[font, text]
                    if batch is not None
                    else [value for box in boxes if (value := _glyph_mask(image, box, rendered.scale)) is not None]
                )
                if samples:
                    masks[font][text] = sorted(samples, key=lambda value: _ink_count(value[0]))[len(samples) // 2]
        bold_fonts = set()
        for font in candidates:
            ratios = []
            for text in masks[font].keys() & masks[reference].keys():
                target, target_aspect = masks[font][text]
                normal, normal_aspect = masks[reference][text]
                if (
                    not 0.75 <= target_aspect / normal_aspect <= 1.33
                    or _ink_count(target & normal) / _ink_count(target | normal) < 0.4
                ):
                    continue
                ratios.append(_ink_count(target) / _ink_count(normal))
            if (
                len(ratios) >= 8
                and statistics.median(ratios) >= 1.4
                and sum(ratio >= 1.3 for ratio in ratios) / len(ratios) >= 0.75
            ):
                bold_fonts.add(font)
    finally:
        image.close()
        rendered.pil_image.close()
    if not bold_fonts:
        return []
    native = get_native()
    if native is not None and _pdf_font_metadata is _STANDARD_FONT_METADATA:
        from ..models import _LineItem

        rows = native.glyph_bold_indices_owned(lines, bold_fonts, _pdf_font_metadata, _LineItem)
        if rows is not None:
            patched = []
            for line_index, positions in rows:
                line = lines[line_index]
                selected = set(positions)
                chars = [
                    dict(char, font=dict(char["font"], weight=700)) if i in selected else char
                    for i, char in enumerate(line.chars)
                ]
                patched.append(replace(line, chars=chars))
            return detect_pdf_text_style_lines(patched, [])
    patched = []
    for line in lines:
        if not any(_pdf_font_metadata(char) in bold_fonts for char in line.chars):
            continue
        chars = [
            dict(char, font=dict(char["font"], weight=700)) if _pdf_font_metadata(char) in bold_fonts else char
            for char in line.chars
        ]
        patched.append(replace(line, chars=chars))
    return detect_pdf_text_style_lines(patched, [])
