"""PDF 字符去重及原始文字几何提取，保持原生提取算法与资源语义。"""

from __future__ import annotations

import logging
from typing import cast

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from .native_contracts import (
    PDFPageImage,
    PDFPageTextGeometry,
)
from .native_lifecycle import _try_close
from .native_objects import _text_object_visibility
from .text._contracts import Char
from .text.dedup import deduplicate_chars
from .text.extract import get_chars

logger = logging.getLogger("rapid_doc._vendor.docvortex_core.document.pdf._document")


def _restore_pdfium_surrogate_pairs(
    chars: list[Char],
    textpage: pdfium.PdfTextPage,
    *,
    raw_codes: dict[int, int] | None = None,
) -> list[Char]:
    """利用 PDFium 原始 UTF-16 code unit 恢复 pdftext 丢失的补充平面字符。"""
    if not any(
        len(text := str(char.get("char", ""))) == 1 and (text == "\ufffd" or 0xD800 <= ord(text) <= 0xDFFF) for char in chars
    ):
        return chars

    try:
        textpage_raw = textpage.raw
        char_count = int(textpage.count_chars())
    except Exception:
        textpage_raw = None
        char_count = 0

    restored_chars: list[Char] = []
    consumed_char_indices: set[int] = set()

    def get_unicode(handle: object, index: int) -> int:
        """优先复用首次读取的原始码值，仅为独立辅助调用读取原生接口。"""
        return raw_codes[index] if raw_codes is not None else int(pdfium_c.FPDFText_GetUnicode(handle, index))

    for char in chars:
        text = str(char.get("char", ""))
        raw_char_idx = char.get("char_idx")
        try:
            char_idx = int(raw_char_idx) if raw_char_idx is not None else -1
        except (TypeError, ValueError):
            char_idx = -1

        if char_idx in consumed_char_indices:
            continue

        raw_code = None
        if (
            textpage_raw is not None
            and 0 <= char_idx < char_count
            and len(text) == 1
            and (text == "\ufffd" or 0xD800 <= ord(text) <= 0xDFFF)
        ):
            raw_code = int(get_unicode(textpage_raw, char_idx))

        high_surrogate = None
        low_surrogate = None
        if raw_code is not None and 0xD800 <= raw_code <= 0xDBFF and char_idx + 1 < char_count:
            next_code = int(get_unicode(textpage_raw, char_idx + 1))
            if 0xDC00 <= next_code <= 0xDFFF:
                high_surrogate = raw_code
                low_surrogate = next_code
                consumed_char_indices.add(char_idx + 1)
        elif raw_code is not None and 0xDC00 <= raw_code <= 0xDFFF and char_idx > 0:
            previous_code = int(get_unicode(textpage_raw, char_idx - 1))
            if 0xD800 <= previous_code <= 0xDBFF:
                high_surrogate = previous_code
                low_surrogate = raw_code

        if high_surrogate is not None and low_surrogate is not None:
            restored_char = cast(Char, dict(char))
            restored_char["char"] = chr(0x10000 + ((high_surrogate - 0xD800) << 10) + (low_surrogate - 0xDC00))
            restored_char["source_indices"] = tuple(
                sorted(
                    set(
                        (
                            *char.get("source_indices", (char_idx,)),
                            char_idx + 1 if raw_code is not None and raw_code <= 0xDBFF else char_idx - 1,
                        )
                    )
                )
            )
            restored_chars.append(restored_char)
            continue

        if len(text) == 1 and 0xD800 <= ord(text) <= 0xDFFF:
            restored_char = cast(Char, dict(char))
            restored_char["char"] = "\ufffd"
            restored_chars.append(restored_char)
            continue

        restored_chars.append(char)

    return restored_chars


def _page_to_image(page: pdfium.PdfPage, scale: float, max_edge: int) -> PDFPageImage:
    """按原缩放与长边上限复制页面像素，并返回独立持有的图片。"""
    long_edge_length = max(*page.get_size())
    if (long_edge_length * scale) > max_edge:
        scale = max_edge / long_edge_length

    bitmap = None
    try:
        bitmap = page.render(scale=scale)  # type: ignore
        bitmap = cast(pdfium.PdfBitmap, bitmap)
        pil_image = bitmap.to_pil()
    finally:
        _try_close(bitmap)

    return PDFPageImage(pil_image=pil_image, scale=scale)


def _extract_page_text_geometry(
    page: pdfium.PdfPage,
    *,
    include_extended_geometry: bool,
    visible_only: bool = False,
    compact: bool = False,
    compact_only: bool = False,
    paint_page_reader=None,
) -> PDFPageTextGeometry | None:
    """在调用方持有的页面和锁内读取字符，使批量提取与独立接口共用实现。"""
    textpage = None
    try:
        raw_page_bbox: list[float] = list(page.get_bbox())
        page_rotation: int = 0
        try:
            page_rotation = page.get_rotation()
        except Exception:
            pass
        visibility = (
            _text_object_visibility(page, tuple(raw_page_bbox), page_rotation, paint_page_reader) if visible_only else None
        )
        textpage = page.get_textpage()
        if compact:
            from .snapshot_bridge import read_text_snapshot

            # 消融未显示稳定收益，恢复由 Python 包装持有 textpage 的原快照入口。
            snapshot = read_text_snapshot(textpage, raw_page_bbox, page_rotation, include_extended_geometry, visibility)
            if snapshot is not None:
                return snapshot
            if compact_only:
                return None
        options = {"visibility_by_object": visibility} if visible_only else {}
        chars = get_chars(textpage, raw_page_bbox, page_rotation, include_geometry=include_extended_geometry, **options)
        raw_codes = {char["char_idx"]: char["raw_code"] for char in chars}
        chars = _restore_pdfium_surrogate_pairs(chars, textpage, raw_codes=raw_codes)
        chars = deduplicate_chars(chars)
        if include_extended_geometry:
            loose_bboxes = {
                char["char_idx"]: char["loose_bbox"]
                for char in chars
                if char.get("loose_bbox") is not None and abs(char["rotation"]) > 1e-9
            }
            tight_bboxes = {char["char_idx"]: char["tight_bbox"] for char in chars if char.get("tight_bbox") is not None}
            origins = {char["char_idx"]: char["origin"] for char in chars if char.get("origin") is not None}
        else:
            loose_bboxes, tight_bboxes, origins = {}, {}, {}
    finally:
        _try_close(textpage)
    return PDFPageTextGeometry(
        chars=chars,
        tight_bboxes=tight_bboxes,
        origins=origins,
        loose_bboxes=loose_bboxes,
    )
