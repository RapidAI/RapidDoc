# Copyright (c) RapidAI. All rights reserved.
"""Recover runs of text lines missed by the model layout."""

from functools import lru_cache
from unicodedata import normalize

import cv2
import numpy as np
from bs4 import BeautifulSoup

from .model_list import AtomicModel
from ...model.reading_order.xycut_plus import xycut_plus_sort
from ...utils.enum_class import CategoryId, ContentType
from ...utils.ocr_utils import get_ocr_result_list
from ...utils.span_block_fix import merge_spans_to_line


def _bounds(poly):
    xs, ys = poly[::2], poly[1::2]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _poly(box):
    x0, y0, x1, y1 = box
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def format_image_text(spans):
    text_spans = [
        {"bbox": _bounds(span["poly"]), "type": ContentType.TEXT, "text": span["text"].strip()}
        for span in spans
        if span["category_id"] == CategoryId.OcrText and span["text"].strip()
    ]
    return "\n".join(
        " ".join(span["text"] for span in sorted(line, key=lambda span: span["bbox"][0]))
        for line in merge_spans_to_line(text_spans)
    )


def _covered(line, regions):
    x0, y0, x1, y1 = line
    area = (x1 - x0) * (y1 - y0)
    return any(
        max(0, min(x1, right) - max(x0, left))
        * max(0, min(y1, bottom) - max(y0, top)) > area * 0.5
        for left, top, right, bottom in regions
    )


def _conflicts_with_recognized(line, layout):
    x0, y0, x1, y1 = line
    line_area = (x1 - x0) * (y1 - y0)
    for item in layout:
        category = item["category_id"]
        if item.get("layout_recovered_text") or category not in (
            CategoryId.OcrText, CategoryId.InterlineEquation_YOLO,
        ):
            continue
        content = item.get("text") if category == CategoryId.OcrText else item.get("latex")
        if not content or not content.strip():
            continue
        left, top, right, bottom = _bounds(item["poly"])
        area = (right - left) * (bottom - top)
        if category == CategoryId.OcrText:
            area = min(area, line_area)
        overlap = max(0, min(x1, right) - max(x0, left)) * max(0, min(y1, bottom) - max(y0, top))
        if area > 0 and overlap > 0.65 * area:
            return True
    return False


def _native_lines(page, scale):
    for block in page["blocks"]:
        for line in block["lines"]:
            chars = [char for span in line["spans"] for char in span["chars"]]
            if line.get("rotation", 0):
                visible = [char for char in chars if char["char"].strip()]
                if len(visible) < 2:
                    continue
                left, top, right, bottom = visible[0]["bbox"]
                end_left, end_top, end_right, end_bottom = visible[-1]["bbox"]
                dx = (end_left + end_right - left - right) / 2
                dy = (end_top + end_bottom - top - bottom) / 2
                # PDFium may report a glyph's italic shear as its angle.
                if abs(dx) <= 2 * abs(dy):
                    continue
            text = "".join(char["char"] for char in chars)
            if text.strip():
                yield tuple(float(v) * scale for v in line["bbox"]), text


def _table_text_key(text):
    return "".join(normalize("NFKC", text).casefold().split())


def _native_table_boundary_lines(lines, regions, tables):
    @lru_cache(maxsize=None)
    def table_rows(html):
        soup = BeautifulSoup(html, "html.parser")
        return [_table_text_key(row.get_text()) for row in soup.find_all("tr")]

    for box, text in lines:
        left, top, right, bottom = box
        if right <= left or bottom <= top or _covered(box, regions):
            continue
        overlapping = [(bounds, item) for bounds, item in tables if _covered(box, [bounds])]
        if not overlapping:
            continue
        key = _table_text_key(text)
        if key and all(
            (top < bounds[1] < bottom or top < bounds[3] < bottom)
            and item.get("html")
            and (rows := table_rows(item["html"]))
            and not any(key in row for row in rows)
            for bounds, item in overlapping
        ):
            yield box, text


def _paragraphs(lines, *, join_fragments=False):
    groups = []
    for box, line in sorted(lines, key=lambda item: (item[0][1], item[0][0])):
        x0, y0, x1, y1 = box
        height = y1 - y0
        matches = []
        for group in groups:
            left, top, right, bottom = group[-1][0]
            typical_height = max(height, bottom - top)
            gap = y0 - bottom
            if (-0.25 * typical_height <= gap <= 1.5 * typical_height
                    and abs(x0 - left) <= 2 * typical_height
                    and min(x1, right) > max(x0, left)):
                matches.append((abs(gap), abs(x0 - left), group))
        if matches:
            min(matches, key=lambda match: match[:2])[2].append((box, line))
        else:
            groups.append([(box, line)])
    paragraphs = [group for group in groups if len(group) >= 2]
    if join_fragments:
        # Det can split a label and its value into separate boxes on one row.
        for group in groups:
            if len(group) != 1:
                continue
            fragment, = group
            x0, y0, x1, y1 = fragment[0]
            for paragraph in paragraphs:
                for (left, top, right, bottom), _ in paragraph:
                    gap = max(x0 - right, left - x1)
                    overlap = min(y1, bottom) - max(y0, top)
                    if (0 <= gap <= max(y1 - y0, bottom - top)
                            and overlap > 0.5 * max(y1 - y0, bottom - top)):
                        paragraph.append(fragment)
                        group.clear()
                        break
                else:
                    continue
                break
    return paragraphs + [group for group in groups if len(group) == 1]


def _reading_order(box, anchors, text_anchors=()):
    if not anchors:
        return None
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    neighbors = [
        (bounds, order) for bounds, order in text_anchors
        if min(y1, bounds[3]) > max(y0, bounds[1])
    ]
    if neighbors and all(right < x0 or left > x1 for (left, _, right, _), _ in neighbors):
        # A recovered column may have no text anchors of its own. Use local
        # geometry to place it relative to the adjacent column, then map that
        # position back to the original model order.
        slot = len(neighbors)
        indices = xycut_plus_sort([bounds for bounds, _ in neighbors] + [box], direction="horizontal")
        position = indices.index(slot)
        before = [neighbors[i][1] for i in indices[:position]]
        after = [neighbors[i][1] for i in indices[position + 1:]]
        if before and not after:
            return float(max(before)) + 0.5
        if after and not before and min(after) > 0:
            return float(min(after)) - 0.5
    (_, top, _, bottom), order = min(
        anchors, key=lambda item: (
            max(item[0][0] - cx, cx - item[0][2], 0),
            abs(cy - (item[0][1] + item[0][3]) / 2),
        )
    )
    delta = cy - (top + bottom) / 2
    return max(0.0, float(order) + delta / (2 * (abs(delta) + max(1, bottom - top))))


def _between_text_blocks(box, text_regions):
    x0, y0, x1, y1 = box
    height = y1 - y0
    above = below = False
    for left, top, right, bottom in text_regions:
        if min(x1, right) - max(x0, left) <= (x1 - x0) * 0.5:
            continue
        above |= -height / 2 <= y0 - bottom <= height
        below |= -height / 2 <= top - y1 <= height
        if above and below:
            return True
    return False


def _has_foreground_ink(image, box):
    """Reject very pale bleed-through only on otherwise unowned OCR crops."""
    x0, y0, x1, y1 = box
    crop = image[max(0, int(y0)):min(image.shape[0], int(y1)),
                 max(0, int(x0)):min(image.shape[1], int(x1)), :3]
    if not crop.size:
        return False
    gray = crop.astype(np.uint16).sum(axis=2) // 3
    background = np.percentile(gray, 90, method='higher')
    return background < 200 or np.count_nonzero(gray <= min(180, background - 60)) * 100 >= gray.size


def _between_layout_blocks(box, text_regions, media_regions):
    x0, y0, x1, y1 = box
    height = y1 - y0
    above = below = text_neighbor = False
    for is_text, regions in ((True, text_regions), (False, media_regions)):
        for left, top, right, bottom in regions:
            if min(x1, right) - max(x0, left) <= (x1 - x0) * .5:
                continue
            a = -height / 2 <= y0 - bottom <= height
            b = -height / 2 <= top - y1 <= height
            above |= a
            below |= b
            text_neighbor |= is_text and (a or b)
    return above and below and text_neighbor


def recover_layout_text(pages, pdf_pages, scales, model_manager, ocr_config, image_ocr_enable=False):
    scanned_pages = []
    image_ocr = []
    for page, pdf_page, scale in zip(pages, pdf_pages, scales):
        rule_layout = "_docvortex_layout" in pdf_page
        if rule_layout and not image_ocr_enable:
            continue
        layout = page["layout_res"]
        anchors = [(_bounds(item["poly"]), item["original_order"]) for item in layout
                   if item["category_id"] not in (CategoryId.OcrText, CategoryId.Abandon)
                   and item.get("original_order") is not None and item["original_order"] >= 0]
        text_anchors = [(_bounds(item["poly"]), item["original_order"]) for item in layout
                        if item["category_id"] == CategoryId.Text
                        and item.get("original_order") is not None and item["original_order"] >= 0]
        needs_ocr = (page["ocr_enable"] or ocr_config.get("use_det_mode") == "ocr"
                     or pdf_page.get("rotate_label") in ("90", "180", "270"))
        height, width = page["np_img"].shape[:2]
        figures = []
        for item in layout:
            if item["category_id"] != CategoryId.ImageBody or item.get("original_label") == "seal":
                continue
            x0, y0, x1, y1 = _bounds(item["poly"])
            # A page-sized image detection cannot establish ownership of body text.
            if x1 - x0 >= 0.9 * width and y1 - y0 >= 0.9 * height:
                continue
            figures.append((item, (x0, y0, x1, y1)))
        if rule_layout and not figures:
            continue
        # Image-only table fallbacks have no text to duplicate.
        regions = [_bounds(item["poly"]) for item in layout
                   if item["category_id"] != CategoryId.ImageBody
                   and not (needs_ocr and item["category_id"] == CategoryId.TableBody
                            and not (item.get("html") or item.get("latex")))]
        if needs_ocr:
            image = cv2.cvtColor(page["np_img"], cv2.COLOR_RGB2BGR)
            model = model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR, lang=page["lang"], ocr_config=ocr_config,
            )
            boxes, _ = model.det_batch_predict([image], 1)[0]
            # Keep detector lines separate for recognition. The generic merge
            # can combine vertically adjacent lines in dense page images.
            lines = [] if boxes is None else [(_bounds(box.flatten()), box) for box in boxes]
            if not rule_layout:
                scanned_pages.append((page, lines))
        else:
            lines = list(_native_lines(pdf_page, scale))

        missing = [(box, line) for box, line in lines
                   if box[2] > box[0] and box[3] > box[1] and not _covered(box, regions)
                   and (not needs_ocr or not _conflicts_with_recognized(box, layout))]
        boundary_boxes = set()
        if not needs_ocr and not rule_layout:
            tables = [(_bounds(item["poly"]), item) for item in layout
                      if item["category_id"] == CategoryId.TableBody]
            if tables:
                non_table_regions = [_bounds(item["poly"]) for item in layout
                                     if item["category_id"] not in (CategoryId.TableBody, CategoryId.ImageBody)]
                boundary_lines = list(_native_table_boundary_lines(lines, non_table_regions, tables))
                missing.extend(boundary_lines)
                boundary_boxes.update(box for box, _ in boundary_lines)
        text_regions = [_bounds(item["poly"]) for item in layout
                        if item["category_id"] in (CategoryId.Title, CategoryId.Text)]
        for figure, bounds in sorted(figures, key=lambda f: (
            (f[1][2] - f[1][0]) * (f[1][3] - f[1][1])
        )):
            owned = [(box, line) for box, line in missing if _covered(box, [bounds])
                     and not _between_text_blocks(box, text_regions)]
            if owned:
                owned_ids = {id(line) for _, line in owned}
                missing = [(box, line) for box, line in missing if id(line) not in owned_ids]
            if not image_ocr_enable:
                continue
            owned.sort(key=lambda item: (item[0][1], item[0][0]))
            if needs_ocr:
                spans = get_ocr_result_list(
                    [line for _, line in owned], (0, 0, 0, 0, width, height, width, height),
                    True, image, page["lang"], "text",
                ) if owned else []
            elif owned:
                spans = [
                    {"category_id": CategoryId.OcrText, "poly": _poly(box), "text": line}
                    for box, line in owned
                ]
            else:
                x0, y0, x1, y1 = bounds
                x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = cv2.cvtColor(page["np_img"][y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
                model = model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR, lang=page["lang"], ocr_config=ocr_config,
                )
                boxes, _ = model.det_batch_predict([crop], 1)[0]
                spans = get_ocr_result_list(
                    sorted(boxes, key=lambda box: (box[0][1], box[0][0])),
                    (0, 0, x0, y0, x1, y1, x1-x0, y1-y0), True, crop, page["lang"], "text",
                ) if boxes is not None and len(boxes) else []
            if spans:
                image_ocr.append((figure, spans))
        if rule_layout:
            continue
        if needs_ocr:
            missing = [(box, line) for box, line in missing if _has_foreground_ink(image, box)]
        groups = _paragraphs(missing, join_fragments=needs_ocr)
        text_regions.extend(
            (min(box[0] for box, _ in group), min(box[1] for box, _ in group),
             max(box[2] for box, _ in group), max(box[3] for box, _ in group))
            for group in groups if len(group) > 1
        )
        for group in groups:
            boxes, lines = zip(*group)
            box = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes))
            order = _reading_order(box, anchors, text_anchors)
            if needs_ocr:
                h, w = image.shape[:2]
                spans = get_ocr_result_list(
                    lines, (0, 0, 0, 0, w, h, w, h), True, image, page["lang"], "text", order,
                )
            else:
                spans = [
                    {"category_id": CategoryId.OcrText, "poly": _poly(bounds), "score": 1.0,
                     "text": text, "docvortex_native_text": True, "original_order": order,
                     "original_label": "text"}
                    for bounds, text in group
                ]
            if spans:
                media_regions = [_bounds(item['poly']) for item in layout
                                 if item['category_id'] in (CategoryId.ImageBody, CategoryId.TableBody)]
                if len(group) == 1 and box not in boundary_boxes and not _between_layout_blocks(box, text_regions, media_regions):
                    for span in spans:
                        span["unassigned_text"] = True
                    layout.extend(spans)
                    continue
                for span in spans:
                    span["layout_recovered_text"] = True
                layout.append({"category_id": CategoryId.Text, "poly": _poly(box), "score": 1.0,
                               "original_label": "text", "original_order": order})
                layout.extend(spans)
    return scanned_pages, image_ocr


def recover_unread_body_lines(scanned_pages, recognize, ocr_config):
    pending = []
    for page, lines in scanned_pages:
        layout = page["layout_res"]
        read = [_bounds(item["poly"]) for item in layout
                if item["category_id"] == CategoryId.OcrText
                and not item.get("unassigned_text") and item.get("text", "").strip()]
        owners = [(item, _bounds(item["poly"])) for item in page["ocr_res_list"]
                  if item["category_id"] == CategoryId.Text]
        protected = [_bounds(item["poly"]) for item in layout
                     if item["category_id"] not in (CategoryId.Text, CategoryId.ImageBody,
                                                    CategoryId.OcrText, CategoryId.LowScoreText)]
        by_owner = {}
        for box, points in lines:
            if (_covered(box, protected) or _covered(box, read)
                    or _conflicts_with_recognized(box, layout)):
                continue
            # A full-page line may include a fragment already read by the crop.
            if any(_covered(other, [box]) for other in read):
                continue
            for i, (_, (x0, y0, x1, y1)) in enumerate(owners):
                if x0 <= box[0] < box[2] <= x1 and y0 <= box[1] < box[3] <= y1:
                    by_owner.setdefault(i, []).append(points)
                    break

        if not by_owner:
            continue
        image = cv2.cvtColor(page["np_img"], cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        spans = []
        for i, boxes in by_owner.items():
            owner, _ = owners[i]
            spans.extend(get_ocr_result_list(
                boxes, (0, 0, 0, 0, width, height, width, height), True, image, page["lang"],
                owner.get("original_label", "text"), owner.get("original_order"),
            ))
        for span in spans:
            span["layout_recovered_text"] = True
        if spans:
            pending.append((layout, spans))

    if pending:
        recognize([spans for _, spans in pending], ocr_config)
        for layout, spans in pending:
            layout.extend(span for span in spans
                          if span["category_id"] == CategoryId.OcrText and span["text"].strip())
