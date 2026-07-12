# Copyright (c) Opendatalab. All rights reserved.
import math
from typing import Any, List

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from ctypes import c_float, c_int
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_blocks, get_lines, get_spans

try:
    from pdftext.pdf.chars import PageChars
    from pdftext.schema import Bbox
except ImportError:  # pdftext 0.6.x
    PageChars = None
    Bbox = None

from rapid_doc.utils.pdfium_guard import close_pdfium_child, pdfium_guard


def _is_page_chars(chars: Any) -> bool:
    """Return whether chars uses pdftext 0.7's column-oriented container."""
    return PageChars is not None and isinstance(chars, PageChars)


def _materialize_page_chars(chars) -> list[dict[str, Any]]:
    """Convert pdftext 0.7 PageChars to RapidDoc's legacy char dictionaries."""
    boxes = chars.boxes.tolist()
    rotations = chars.rotations.tolist()
    font_ids = chars.font_ids.tolist()
    char_indices = chars.char_indices.tolist()
    return [
        {
            "bbox": Bbox([float(value) for value in boxes[index]]),
            "char": chars.text[index],
            "rotation": float(rotations[index]),
            "font": chars.fonts[int(font_ids[index])],
            "char_idx": int(char_indices[index]),
        }
        for index in range(len(chars))
    ]


def _ensure_legacy_chars(chars) -> list[dict[str, Any]]:
    """Expose one stable char schema to RapidDoc across pdftext versions."""
    if _is_page_chars(chars):
        return _materialize_page_chars(chars)
    return chars


def _bbox_coords(char: dict[str, Any]) -> list[float]:
    bbox = char.get("bbox")
    bbox = getattr(bbox, "bbox", bbox)
    return [float(value) for value in bbox]


def _legacy_chars_to_page_chars(chars):
    """Pack legacy chars for pdftext 0.7 get_spans; 0.6 keeps the list input."""
    if PageChars is None or _is_page_chars(chars):
        return chars

    fonts: list[dict[str, Any]] = []
    font_ids_by_key: dict[tuple[Any, ...], int] = {}
    text_parts, codes, rotations, boxes, font_ids, char_indices = [], [], [], [], [], []
    for fallback_index, char in enumerate(chars):
        text = str(char.get("char", ""))[:1] or "\uFFFD"
        font = char.get("font") or {}
        font_key = (
            font.get("name"), font.get("flags"), font.get("size"), font.get("weight")
        )
        font_id = font_ids_by_key.get(font_key)
        if font_id is None:
            font_id = len(fonts)
            font_ids_by_key[font_key] = font_id
            fonts.append(dict(font))
        text_parts.append(text)
        codes.append(ord(text))
        rotations.append(float(char.get("rotation") or 0.0))
        boxes.append(_bbox_coords(char))
        font_ids.append(font_id)
        char_indices.append(int(char.get("char_idx", fallback_index)))

    return PageChars(
        "".join(text_parts),
        np.asarray(codes, dtype=np.uint32),
        np.asarray(rotations, dtype=np.float64),
        np.asarray(boxes, dtype=np.float64).reshape((-1, 4)),
        np.asarray(font_ids, dtype=np.int32),
        fonts,
        np.asarray(char_indices, dtype=np.int64),
    )


def _get_pdf_object_bounds(obj) -> tuple[float, float, float, float]:
    """Read object bounds across pypdfium2 4.x and 5.x."""
    get_bounds = getattr(obj, "get_bounds", None)
    if get_bounds is not None:
        return get_bounds()
    return obj.get_pos()


def get_page(
    page: pdfium.PdfPage,
    quote_loosebox: bool = True,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> dict:
    page_chars = get_page_chars(page, quote_loosebox=quote_loosebox)
    lines = get_lines_from_chars(
        page_chars["chars"],
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    blocks = get_blocks(lines)

    return {
        "size": page_chars["size"],
        "bbox": page_chars["bbox"],
        "width": page_chars["width"],
        "height": page_chars["height"],
        "rotation": page_chars["rotation"],
        "blocks": blocks,
        # Keep native characters for the pre-OCR ruled-table path.  They are
        # already extracted above, so this adds no PDFium pass.
        "chars": page_chars["chars"],
    }


def get_page_vector_lines(page: pdfium.PdfPage, max_depth: int = 8) -> list[dict]:
    """Extract axis-aligned PDF path segments in top-left page coordinates.

    Only line geometry is retained. Curves, fills and styling are deliberately
    ignored: the ruled-table detector needs borders, not a rendering model.
    Object bounds are used as a conservative rectangle fallback for PDF
    producers that encode cell borders as filled or closed paths.
    """
    with pdfium_guard():
        page_width, page_height = page.get_size()
        path_objects = list(
            page.get_objects(
                filter=(pdfium_c.FPDF_PAGEOBJ_PATH,),
                max_depth=max_depth,
            )
        )
    lines: list[dict] = []
    for obj in path_objects:
        try:
            with pdfium_guard():
                fill_mode, stroked = c_int(), c_int()
                if not pdfium_c.FPDFPath_GetDrawMode(obj.raw, fill_mode, stroked):
                    continue
                if not stroked.value:
                    # Many office/PDF generators draw table rules as filled
                    # hairline rectangles. Liteparse treats those as graphic
                    # primitives too; reduce each one to its center line.
                    if fill_mode.value == pdfium_c.FPDF_FILLMODE_NONE:
                        continue
                    left, bottom, right, top = _get_pdf_object_bounds(obj)
                    width, height = abs(right-left), abs(top-bottom)
                    if min(width, height) <= 2.0 and max(width, height) > 1.0:
                        if width >= height:
                            y = page_height-(bottom+top)/2
                            lines.append({"x1": left, "y1": y, "x2": right, "y2": y})
                        else:
                            x = (left+right)/2
                            lines.append({"x1": x, "y1": page_height-top, "x2": x, "y2": page_height-bottom})
                        continue
                    # A filled rectangular border is commonly encoded as an
                    # even-odd compound path (outer rectangle + inner rectangle),
                    # not as a stroke. It has at least two closed 4-edge subpaths.
                    if pdfium_c.FPDFPath_CountSegments(obj.raw) < 9:
                        continue
                matrix = obj.get_matrix()
                a, b, c, d, e, f = matrix.get()
                count = pdfium_c.FPDFPath_CountSegments(obj.raw)
                points = []
                for idx in range(max(0, count)):
                    seg = pdfium_c.FPDFPath_GetPathSegment(obj.raw, idx)
                    if not seg:
                        continue
                    x, y = c_float(), c_float()
                    if not pdfium_c.FPDFPathSegment_GetPoint(seg, x, y):
                        continue
                    px = a * x.value + c * y.value + e
                    py = b * x.value + d * y.value + f
                    points.append((pdfium_c.FPDFPathSegment_GetType(seg), px, page_height - py))
                prev = None
                useful = 0
                for kind, x, y in points:
                    if kind == pdfium_c.FPDF_SEGMENT_MOVETO:
                        prev = (x, y)
                    elif kind == pdfium_c.FPDF_SEGMENT_LINETO and prev is not None:
                        x0, y0 = prev
                        if (abs(y-y0) <= 1.0 and abs(x-x0) > 1.0) or (abs(x-x0) <= 1.0 and abs(y-y0) > 1.0):
                            lines.append({"x1": x0, "y1": y0, "x2": x, "y2": y})
                            useful += 1
                        prev = (x, y)
                if useful == 0:
                    left, bottom, right, top = _get_pdf_object_bounds(obj)
                    if right-left > 1.0 and top-bottom > 1.0:
                        y0, y1 = page_height-top, page_height-bottom
                        lines.extend([
                            {"x1": left, "y1": y0, "x2": right, "y2": y0},
                            {"x1": left, "y1": y1, "x2": right, "y2": y1},
                            {"x1": left, "y1": y0, "x2": left, "y2": y1},
                            {"x1": right, "y1": y0, "x2": right, "y2": y1},
                        ])
        except Exception:
            # A malformed decorative path must never make PDF parsing fail.
            continue
        finally:
            close_pdfium_child(obj)
    return lines


def get_page_chars(
    page: pdfium.PdfPage,
    textpage=None,
    quote_loosebox: bool = True,
    page_char_count: int | None = None,
) -> dict:
    """轻量读取页面字符坐标，供只需要 char 级信息的路径复用。"""
    owns_textpage = textpage is None
    try:
        with pdfium_guard():
            if textpage is None:
                textpage = page.get_textpage()
            page_bbox: List[float] = page.get_bbox()
            page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
            page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

            page_rotation = 0
            try:
                page_rotation = page.get_rotation()
            except Exception:
                pass

            if page_char_count is None:
                page_char_count = textpage.count_chars()

            chars = deduplicate_chars(
                get_chars(textpage, page_bbox, page_rotation, quote_loosebox)
            )
            chars = _ensure_legacy_chars(chars)
            page_size = page.get_size()
    finally:
        if owns_textpage:
            close_pdfium_child(textpage)

    return {
        "size": page_size,
        "bbox": page_bbox,
        "width": page_width,
        "height": page_height,
        "rotation": page_rotation,
        "char_count": page_char_count,
        "chars": chars,
    }


def get_lines_from_chars(
    chars,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
):
    """从已提取的字符构建 pdftext lines，避免重复读取 PDFium textpage。"""
    chars = _legacy_chars_to_page_chars(chars)
    spans = get_spans(
        chars,
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    lines = get_lines(spans)
    assign_scripts(
        lines,
        height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    return lines
