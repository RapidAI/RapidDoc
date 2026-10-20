"""Check whether text over large page images actually contributes visible pixels."""

import math

from PIL import ImageChops
import pypdfium2.raw as raw

from rapid_doc.utils.pdfium_guard import close_pdfium_child, pdfium_guard


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersect(a, b):
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _union_area(boxes):
    boxes = [box for box in boxes if _area(box)]
    edges = sorted({box[i] for box in boxes for i in (0, 2)})
    total = 0.0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((box[1], box[3]) for box in boxes if box[0] < right and box[2] > left)
        end = float('-inf')
        for bottom, top in intervals:
            total += (right - left) * max(0.0, top - max(bottom, end))
            end = max(end, top)
    return total


def _visual_box(box, frame, rotation):
    left, bottom, right, top = frame
    def point(x, y):
        if rotation == 90:
            return y - bottom, x - left
        if rotation == 180:
            return right - x, y - bottom
        if rotation == 270:
            return top - y, right - x
        return x - left, top - y
    points = [point(x, y) for x in (box[0], box[2]) for y in (box[1], box[3])]
    return min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)


def _render(page, scale):
    bitmap = page.render(scale=scale, may_draw_forms=False)
    try:
        # Own the pixels independently of PDFium's bitmap lifetime.
        return bitmap.to_pil().convert('RGB')
    finally:
        bitmap.close()


def _difference_without_text(page, scale):
    before = _render(page, scale)
    objects, changed = [], []
    after = None
    try:
        for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT], max_depth=10):
            objects.append(obj)
            mode = raw.FPDFTextObj_GetTextRenderMode(obj.raw)
            if mode in (0, 1, 2, 4, 5, 6):
                # Preserve clipping text: clip-only (7), rather than invisible (3).
                if not raw.FPDFTextObj_SetTextRenderMode(obj.raw, 7 if mode >= 4 else 3):
                    raise RuntimeError('Cannot suppress PDF text painting')
                changed.append((obj.raw, mode))
        after = _render(page, scale)
        diff = ImageChops.difference(before, after)
        try:
            # Any channel difference counts, including faint native text.
            bands = diff.split()
            try:
                partial = ImageChops.lighter(bands[0], bands[1])
                try:
                    return ImageChops.lighter(partial, bands[2])
                finally:
                    partial.close()
            finally:
                for band in bands:
                    band.close()
        finally:
            diff.close()
    finally:
        restored = True
        for obj, mode in reversed(changed):
            restored = bool(raw.FPDFTextObj_SetTextRenderMode(obj, mode)) and restored
        before.close()
        if after is not None:
            after.close()
        for obj in objects:
            close_pdfium_child(obj)
        if not restored:
            raise RuntimeError('Cannot restore PDF text painting')


def image_backed_text_signal(page, bounds_reader, min_chars=50, min_coverage=0.8):
    """Probe large-image pages only; caller retains the unmodified PDF bytes."""
    result = {'image_backed_ocr': False, 'candidate_chars': 0, 'unpainted_chars': 0, 'probed': False}
    with pdfium_guard():
        frame = page.get_bbox()
        images = []
        for obj in page.get_objects(filter=[raw.FPDF_PAGEOBJ_IMAGE], max_depth=3):
            try:
                box = _intersect(bounds_reader(obj), frame)
                if _area(box):
                    images.append(box)
            finally:
                close_pdfium_child(obj)
        if not _area(frame) or _union_area(images) / _area(frame) < min_coverage:
            return result

        boxes, valid_chars = [], 0
        rotation = page.get_rotation()
        textpage = page.get_textpage()
        try:
            for index in range(textpage.count_chars()):
                code = raw.FPDFText_GetUnicode(textpage.raw, index)
                if not 0 < code <= 0x10ffff or chr(code).isspace():
                    continue
                valid_chars += 1
                box = _intersect(textpage.get_charbox(index), frame)
                if _area(box) and _union_area([_intersect(box, image) for image in images]) >= _area(box) * 0.8:
                    boxes.append(_visual_box(box, frame, rotation))
        finally:
            close_pdfium_child(textpage)
        result['candidate_chars'] = len(boxes)
        if len(boxes) < max(min_chars, valid_chars * 0.5):
            return result

        width, height = page.get_size()
        difference = _difference_without_text(page, min(1.5, 1024 / max(width, height)))
        try:
            result['probed'] = True
            sx, sy = difference.width / width, difference.height / height
            for box in boxes:
                crop_box = (max(0, math.floor(box[0] * sx)), max(0, math.floor(box[1] * sy)),
                            min(difference.width, math.ceil(box[2] * sx)), min(difference.height, math.ceil(box[3] * sy)))
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    continue
                crop = difference.crop(crop_box)
                try:
                    if crop.getbbox() is None:
                        result['unpainted_chars'] += 1
                finally:
                    crop.close()
        finally:
            difference.close()
        result['image_backed_ocr'] = result['unpainted_chars'] >= max(min_chars, valid_chars * 0.5)
        return result
