"""仅为显式 NUL ToUnicode 建立临时文字视图，以字体资源和原字符码恢复单字符。"""

from __future__ import annotations

import logging
import re
import unicodedata
from contextlib import closing
from copy import copy
from io import BytesIO
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from .pdfium import pdfium_guard

logger = logging.getLogger(__name__)

# CMap 只读词法：注释、名称和字符串均作为整体跳过，绝不执行 PostScript。
_TOKENS = re.compile(
    rb"%[^\r\n]*|/[^\s<>{}\[\]()/%]+|\((?:\\[\s\S]|[^\\()])*\)|<<|>>|<[0-9A-Fa-f\s]*>|[^\s<>{}\[\]()/%]+|[^\s]"
)
_SECTIONS = {b"begincodespacerange": b"endcodespacerange", b"beginbfchar": b"endbfchar", b"beginbfrange": b"endbfrange"}
_MAX_CMAP_BYTES = 1024 * 1024


def _hex(token: bytes) -> bytes:
    """严格读取偶数位十六进制，不猜测损坏的 CMap 字节。"""
    if not token.startswith(b"<") or not token.endswith(b">") or token.startswith(b"<<"):
        raise ValueError("invalid CMap hex string")
    return bytes.fromhex(token[1:-1].decode("ascii"))


def _simple_cmap(data: bytes) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """读取有界单字节 bfchar/bfrange；继承、冲突或异常语法保持原样。"""
    if len(data) > _MAX_CMAP_BYTES:
        raise ValueError("simple-font CMap exceeds bound")
    tokens = [token for token in _TOKENS.findall(data) if not token.startswith(b"%")]
    if tokens.count(b"begincmap") != 1 or tokens.count(b"endcmap") != 1:
        raise ValueError("invalid CMap boundary")
    if any(token in {b"usecmap", b"begincidchar", b"begincidrange", b"(", b")"} for token in tokens):
        raise ValueError("unsupported inherited or malformed CMap")
    mapping: dict[int, bytes] = {}
    codespaces: list[tuple[int, int]] = []
    index = tokens.index(b"begincmap") + 1
    end = tokens.index(b"endcmap")
    while index < end:
        operator = tokens[index]
        if operator not in _SECTIONS:
            index += 1
            continue
        count = int(tokens[index - 1])
        if not 0 <= count <= 256:
            raise ValueError("invalid simple-font CMap count")
        index += 1
        for _ in range(count):
            first, second = _hex(tokens[index]), _hex(tokens[index + 1])
            index += 2
            if len(first) != 1:
                raise ValueError("not a single-byte font CMap")
            start = first[0]
            if operator == b"beginbfchar":
                pairs = [(start, second)]
            else:
                if len(second) != 1 or second[0] < start:
                    raise ValueError("invalid single-byte CMap range")
                stop = second[0]
                if operator == b"begincodespacerange":
                    codespaces.append((start, stop))
                    continue
                destination = tokens[index]
                index += 1
                if destination == b"[":
                    pairs = [(code, _hex(tokens[index + code - start])) for code in range(start, stop + 1)]
                    index += stop - start + 1
                    if tokens[index] != b"]":
                        raise ValueError("invalid CMap destination array")
                    index += 1
                else:
                    base = _hex(destination)
                    value = int.from_bytes(base, "big")
                    pairs = [(code, (value + code - start).to_bytes(len(base), "big")) for code in range(start, stop + 1)]
            for code, destination in pairs:
                if code in mapping or not destination or len(destination) % 2 or len(destination) > 512:
                    raise ValueError("ambiguous or malformed CMap destination")
                mapping[code] = destination
        if tokens[index] != _SECTIONS[operator]:
            raise ValueError("invalid CMap section boundary")
        index += 1
    if not codespaces or any(not any(lo <= code <= hi for lo, hi in codespaces) for code in mapping):
        raise ValueError("CMap code outside declared codespace")
    return mapping, codespaces


def _repaired_cmap(font: Any) -> tuple[bytes, set[int]] | None:
    """只用该字体 Differences 的标准 glyph name 替换 NUL，保留全部有效映射。"""
    from fontTools.agl import toUnicode

    if font.get("/Subtype") not in {"/Type1", "/TrueType"} or "/ToUnicode" not in font:
        return None
    encoding = font.get("/Encoding")
    encoding = encoding.get_object() if encoding is not None else None
    if not isinstance(encoding, dict) or not encoding.get("/Differences"):
        return None
    data = font["/ToUnicode"].get_data()
    mapping, codespaces = _simple_cmap(data)
    recovered: set[int] = set()
    code = -1
    seen = set()
    for value in encoding["/Differences"]:
        if isinstance(value, int):
            code = int(value)
            continue
        if not 0 <= code <= 255 or code in seen or not str(value).startswith("/"):
            raise ValueError("invalid font Differences")
        seen.add(code)
        if mapping.get(code) == b"\0\0":
            text = toUnicode(str(value)[1:])
            # 一对一恢复保证来源索引不变；未知名称、多字符 ligature 和控制码不作猜测。
            if len(text) == 1 and text != "\ufffd" and not unicodedata.category(text).startswith("C"):
                mapping[code] = text.encode("utf-16-be")
                recovered.add(ord(text))
        code += 1
    if not recovered:
        return None
    lines = [
        b"/CIDInit /ProcSet findresource begin",
        b"12 dict begin",
        b"begincmap",
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
        b"/CMapName /DocVortexGlyphFallback def",
        b"/CMapType 2 def",
        f"{len(codespaces)} begincodespacerange".encode(),
    ]
    lines.extend(f"<{lo:02x}><{hi:02x}>".encode() for lo, hi in codespaces)
    lines.append(b"endcodespacerange")
    entries = list(sorted(mapping.items()))
    for offset in range(0, len(entries), 100):
        batch = entries[offset : offset + 100]
        lines.append(f"{len(batch)} beginbfchar".encode())
        lines.extend(f"<{code:02x}><{text.hex()}>".encode() for code, text in batch)
        lines.append(b"endbfchar")
    lines.extend([b"endcmap", b"CMapName currentdict /CMap defineresource pop", b"end", b"end"])
    return b"\n".join(lines), recovered


def _repair_page_fonts(page: Any, writer: Any) -> set[int]:
    """遍历页面和嵌套 Form 的资源对象，字体共享 CMap 时也独立创建恢复映射。"""
    from pypdf.generic import DecodedStreamObject, NameObject

    visited: set[int] = set()
    recovered: set[int] = set()

    def visit(resources: Any, depth: int) -> None:
        """资源图按对象身份去重并限制深度，不按字体名称合并不同编码。"""
        if resources is None or depth >= 32:
            return
        resources = resources.get_object()
        if not isinstance(resources, dict) or id(resources) in visited:
            return
        visited.add(id(resources))
        fonts = resources.get("/Font", {})
        fonts = fonts.get_object() if hasattr(fonts, "get_object") else fonts
        for ref in fonts.values():
            font = ref.get_object()
            if id(font) in visited:
                continue
            visited.add(id(font))
            try:
                repair = _repaired_cmap(font)
            except Exception as exc:
                logger.debug("Skip unsupported PDF font CMap: %s", exc)
                continue
            if repair is not None:
                data, codes = repair
                stream = DecodedStreamObject()
                stream.set_data(data)
                font[NameObject("/ToUnicode")] = writer._add_object(stream)
                recovered.update(codes)
        objects = resources.get("/XObject", {})
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        for ref in objects.values():
            obj = ref.get_object()
            if obj.get("/Subtype") == "/Form":
                visit(obj.get("/Resources"), depth + 1)

    visit(page.get("/Resources"), 0)
    return recovered


def _matching_char(textpage: pdfium.PdfTextPage, other: pdfium.PdfTextPage, index: int) -> bool:
    """确认恢复视图中的同一索引仍指向相同字形位置，拒绝重排或生成字符变化。"""
    return (
        textpage.get_charbox(index) == other.get_charbox(index)
        and textpage.get_charbox(index, loose=True) == other.get_charbox(index, loose=True)
        and raw.FPDFText_GetCharAngle(textpage.raw, index) == raw.FPDFText_GetCharAngle(other.raw, index)
        and raw.FPDFText_IsGenerated(textpage.raw, index) == raw.FPDFText_IsGenerated(other.raw, index) == 0
    )


def recover_null_unicode(textpage: pdfium.PdfTextPage, indices: list[int], reader_page: Any) -> dict[int, int]:
    """在同库锁内短暂打开修复文字视图；仅返回经索引和几何核验的 Unicode，不改源 PDF。"""
    if reader_page is None or not indices:
        return {}
    from pypdf import PdfWriter

    try:
        with pdfium_guard(), PdfWriter() as writer:
            # 只克隆目标页及字体资源，不复制批注关联的其它页，也不修改 reader 持有的原对象。
            source_page = copy(reader_page)
            source_page.pop("/Annots", None)
            source_page.pop("/B", None)
            page = writer.add_page(source_page)
            recovered = _repair_page_fonts(page, writer)
            if not recovered:
                return {}
            output = BytesIO()
            writer.write(output)
            with pdfium.PdfDocument(output.getvalue()) as document, closing(document[0]) as page:
                with closing(page.get_textpage()) as other:
                    if other.count_chars() != textpage.count_chars():
                        return {}
                    original = [raw.FPDFText_GetUnicode(textpage.raw, i) for i in range(textpage.count_chars())]
                    candidate = [raw.FPDFText_GetUnicode(other.raw, i) for i in range(other.count_chars())]
                    # 完整字符序列仅允许 NUL 位置变化；有效文本变化代表不能安全对齐的副作用。
                    if any(
                        before != after and (before != 0 or after not in recovered)
                        for before, after in zip(original, candidate, strict=True)
                    ):
                        return {}
                    return {
                        i: candidate[i]
                        for i in indices
                        if original[i] == 0 and candidate[i] in recovered and _matching_char(textpage, other, i)
                    }
    except Exception as exc:
        # 可选恢复不能让损坏的资源字典阻断 PDFium 原有读取，也不凭文本上下文填补失败字符。
        logger.debug("Skip PDF NUL glyph recovery: %s", exc)
        return {}
