"""Optional DocVortex adapters for native-text PDF layout and tables."""

from __future__ import annotations

import re

from rapid_doc.utils.enum_class import CategoryId


DOCVORTEX_MODES = {"off", "rule", "rule_plus_s"}


def get_docvortex_mode(layout_config: dict | None) -> str:
    """Resolve the PDF layout route, including the original model_type alias."""
    config = layout_config or {}
    model_type = getattr(config.get("model_type"), "value", config.get("model_type"))
    mode = config.get("docvortex_mode")
    if mode is None:
        mode = "rule" if model_type == "docvortex_rule" else "off"
    if mode not in DOCVORTEX_MODES:
        raise ValueError(f"Invalid docvortex_mode: {mode!r}; expected one of {sorted(DOCVORTEX_MODES)}")
    if model_type == "docvortex_rule" and mode == "off":
        raise ValueError("model_type=docvortex_rule conflicts with docvortex_mode=off")
    return mode


_SUPPLEMENT_IDS = {
    CategoryId.InterlineEquation_YOLO,
    CategoryId.InlineEquation,
    CategoryId.InterlineEquationNumber_Layout,
}
_FORMULA_IDS = {CategoryId.InterlineEquation_YOLO, CategoryId.InlineEquation}
_FORMULA_MIN_SCORE = 0.50


def _bbox(item: dict) -> tuple[float, float, float, float]:
    poly = item["poly"]
    return poly[0], poly[1], poly[4], poly[5]


def _intersection_ratio(first: dict, second: dict) -> float:
    ax0, ay0, ax1, ay1 = _bbox(first)
    bx0, by0, bx1, by1 = _bbox(second)
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    first_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    second_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    return intersection / min(first_area, second_area) if min(first_area, second_area) else 0.0


def merge_docvortex_supplements(native: list[dict], pp_s: list[dict], *,
                                include_formula: bool = True,
                                include_seal: bool = True) -> list[dict]:
    """Keep native reading order and replace matching formula/seal regions."""
    native_text = [item for item in native if item["category_id"] in {
        CategoryId.Text, CategoryId.Title, CategoryId.TableCaption,
        CategoryId.ImageCaption,
    }]
    supplements = [
        item for item in pp_s
        if ((include_formula and item["category_id"] in _SUPPLEMENT_IDS)
            or (include_seal and item.get("original_label") == "seal"))
        and (item["category_id"] not in _FORMULA_IDS
             or item["score"] >= _FORMULA_MIN_SCORE)
        and not (
            item["category_id"] in _FORMULA_IDS
            and any(_intersection_ratio(item, text) >= 0.8
                    and _area(item) >= 0.5 * _area(text) for text in native_text)
        )
    ]
    native_tables = [item for item in native if item.get("original_label") == "table"]
    supplements = [
        item for item in supplements
        if item.get("original_label") == "seal"
        or not any(_intersection_ratio(item, table) >= 0.5 for table in native_tables)
    ]
    if not supplements:
        return native

    # Replace a native equation at its original reading-order slot. Looking
    # for the nearest *remaining* block moves formulas past intervening prose.
    replacements = {}
    consumed = set()
    for index, item in enumerate(native):
        is_equation = item.get("original_label") == "equation"
        is_image = item.get("original_label") in {"image", "chart"}
        matches = [
            extra for extra in supplements
            if id(extra) not in consumed and _intersection_ratio(item, extra) >= 0.5
            and ((is_equation and extra["category_id"] in _FORMULA_IDS
                  and _area(extra) <= 3 * _area(item))
                 or (is_image and extra.get("original_label") == "seal"
                     and _area(item) <= 2 * _area(extra)))
        ]
        if matches:
            chosen = max(matches, key=lambda extra: extra["score"])
            replacements[index] = chosen
            consumed.add(id(chosen))

    ordered = [(float(index), replacements.get(index, item))
               for index, item in enumerate(native)]
    equation_anchors = [(order, item) for order, item in ordered
                        if item["category_id"] == CategoryId.InterlineEquation_YOLO]
    for extra in sorted(supplements, key=lambda item: (_bbox(item)[1], _bbox(item)[0])):
        if id(extra) in consumed:
            continue
        # Suppress alternative detections of a formula already used to replace
        # the native equation.
        if extra["category_id"] in _FORMULA_IDS and any(
            _intersection_ratio(extra, replacement) >= 0.7
            for replacement in replacements.values()
            if replacement["category_id"] in _FORMULA_IDS
        ):
            continue
        if extra["category_id"] == CategoryId.InterlineEquationNumber_Layout:
            same_row = [(order, equation) for order, equation in equation_anchors
                        if _vertical_overlap_ratio(extra, equation) >= 0.5]
            if same_row:
                anchor_order, _ = min(same_row, key=lambda pair: abs(
                    _center(extra)[1] - _center(pair[1])[1]
                ))
                ordered.append((anchor_order + 0.25, extra))
                continue
        if native:
            cx, cy = _center(extra)
            anchor = min(range(len(native)), key=lambda i: (
                (cx - _center(native[i])[0]) ** 2
                + (cy - _center(native[i])[1]) ** 2
            ))
            order = anchor + (0.5 if cy >= _center(native[anchor])[1] else -0.5)
        else:
            order = 0.0
        ordered.append((order, extra))
        if extra["category_id"] == CategoryId.InterlineEquation_YOLO:
            equation_anchors.append((order, extra))
    ordered.sort(key=lambda pair: pair[0])
    result = [item.copy() for _, item in ordered]
    for order, item in enumerate(result):
        item["original_order"] = order
    return result


def _area(item: dict) -> float:
    x0, y0, x1, y1 = _bbox(item)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _center(item: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = _bbox(item)
    return (x0 + x1) / 2, (y0 + y1) / 2


def _vertical_overlap_ratio(first: dict, second: dict) -> float:
    _, ay0, _, ay1 = _bbox(first)
    _, by0, _, by1 = _bbox(second)
    height = min(ay1 - ay0, by1 - by0)
    return max(0, min(ay1, by1) - max(ay0, by0)) / height if height > 0 else 0.0


_LABELS = {
    "text": CategoryId.Text,
    "doc_title": CategoryId.Title,
    "paragraph_title": CategoryId.Title,
    "title": CategoryId.Title,
    "image": CategoryId.ImageBody,
    "chart": CategoryId.ImageBody,
    "caption": CategoryId.ImageCaption,
    "table": CategoryId.TableBody,
    "table_caption": CategoryId.TableCaption,
    "equation": CategoryId.InterlineEquation_YOLO,
    "code": CategoryId.Text,
    "list": CategoryId.Text,
    "index": CategoryId.Text,
    "ref_text": CategoryId.Text,
    "footnote": CategoryId.Text,
    "page_footnote": CategoryId.Text,
    "aside_text": CategoryId.Text,
    "header": CategoryId.Abandon,
    "footer": CategoryId.Abandon,
    "page_number": CategoryId.Abandon,
}


def _require_docvortex():
    try:
        from docvortex.document.pdf import PDFDocument
    except ImportError as exc:
        raise ImportError(
            "DocVortex support requires `pip install rapid-doc[docvortex]`"
        ) from exc
    return PDFDocument


def analyze_docvortex_layout(pdf_bytes: bytes) -> list[list[dict]]:
    """Run the document-level native pipeline once, preserving cross-page context."""
    PDFDocument = _require_docvortex()
    from docvortex.analyzers.native import PdfModel

    with PDFDocument(pdf_bytes) as document:
        return PdfModel().predict(document)


def _block_text(content) -> str:
    """Read plain text from DocVortex's string or rich inline content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(part) for part in content)
    if isinstance(content, dict):
        return _block_text(content.get("content"))
    return ""


def docvortex_layout_fallback_reason(blocks: list[dict], layout: list[dict],
                                     width: int, height: int,
                                     native_char_count: int = 0) -> str | None:
    """Detect native page layouts that would discard substantial page content."""
    if not blocks or not layout:
        return "empty layout"

    page_area = width * height
    if page_area > 0:
        for item in layout:
            if item["category_id"] != CategoryId.Abandon or item.get("original_label") not in {
                "header", "footer", "aside_text",
            }:
                continue
            x0, y0, x1, y1 = _bbox(item)
            if (x1 - x0) * (y1 - y0) >= 0.4 * page_area:
                return f"oversized discarded {item['original_label']}"

    retained = [item for item in layout if item["category_id"] != CategoryId.Abandon]
    discarded_text_length = sum(
        len(_block_text(block.get("content")))
        for block in blocks if block.get("type") in {"header", "footer", "aside_text"}
    )
    if not retained and (native_char_count >= 80 or discarded_text_length >= 80):
        return "text page has no retained content"
    if (len(retained) == 1 and retained[0]["category_id"] == CategoryId.ImageBody
            and native_char_count >= 80):
        return "text page reduced to one image"
    return None


def convert_docvortex_layout(blocks: list[dict], width: int, height: int,
                             ignored_labels: set[str] | None = None) -> list[dict]:
    """Convert normalized DocVortex blocks to RapidDoc's pixel layout contract."""
    ignored_labels = ignored_labels or set()
    output = []
    for order, block in enumerate(blocks):
        label = str(block.get("type", "text"))
        bbox = block.get("bbox")
        if label not in _LABELS or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(bbox[0]) * width, float(bbox[1]) * height,
                              float(bbox[2]) * width, float(bbox[3]) * height)
        except (TypeError, ValueError):
            continue
        x0, x1 = max(0.0, min(width, x0)), max(0.0, min(width, x1))
        y0, y1 = max(0.0, min(height, y0)), max(0.0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        category_id = _LABELS[label]
        rotated = int(block.get("angle") or 0) % 360 != 0
        if label == "caption":
            caption = _block_text(block.get("content"))
            if rotated:
                # Distance based table grouping assumes horizontal captions.
                category_id = CategoryId.Text
            elif re.match(r"^\s*(?:table|tab\.|表\s*\d|表格\s*\d)", caption, re.IGNORECASE):
                category_id = CategoryId.TableCaption
            elif not re.match(r"^\s*(?:fig(?:ure)?\.?|图\s*\d)", caption, re.IGNORECASE):
                category_id = CategoryId.Text
        item = {
            "category_id": CategoryId.Abandon if label in ignored_labels else category_id,
            "original_label": label,
            "original_order": order,
            "poly": [x0, y0, x1, y0, x1, y1, x0, y1],
            "polygon_points": None,
            "score": 1.0,
        }
        content = block.get("content")
        if ((label in {"caption", "table_caption", "footnote", "page_footnote"}
             or (rotated and category_id in (CategoryId.Text, CategoryId.Title)))
                and item["category_id"] != CategoryId.Abandon):
            item["native_text"] = _block_text(content)
        if label == "table" and isinstance(content, str) and "<table" in content.lower():
            item["html"] = content
            item["table_parse_method"] = "docvortex_native"
        output.append(item)
    return output


def recover_docvortex_table(page_dict: dict, table_bbox: tuple[float, float, float, float],
                            angle: int = 0):
    """Reuse detached page primitives for all table regions on one page."""
    PDFDocument = _require_docvortex()
    from docvortex.analyzers.pdf import prepare_table_page, recover_table_region

    table_page = page_dict.get("_docvortex_table_page")
    if table_page is None:
        with PDFDocument(page_dict["_source_pdf_bytes"]) as document:
            table_page = prepare_table_page(document[page_dict["_source_page_index"]])
        page_dict["_docvortex_table_page"] = table_page
    return recover_table_region(table_page, table_bbox, angle=angle)
