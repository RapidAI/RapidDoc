import pytest

from rapid_doc.backend.pipeline.analyze_utils import _native_row_gaps, _without_native_rows
from rapid_doc.backend.pipeline.layout_recovery import _conflicts_with_recognized, _native_lines
from rapid_doc.backend.pipeline.layout_recovery import _has_foreground_ink, _between_layout_blocks
import numpy as np


def test_partial_native_block_only_recognizes_missing_rows():
    def quad(left, top, right, bottom):
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    image = np.full((60, 120, 3), 255, dtype=np.uint8)
    crop = (5, 5, 0, 0, 110, 50, 120, 60)
    native = [quad(40, 35, 55, 45), quad(65, 35, 70, 45)]
    assert _native_row_gaps(image, native, crop) == []

    image[14:23, 20:90] = 0
    bands = _native_row_gaps(image, native, crop)
    missing = quad(20, 14, 90, 23)
    same_row = quad(56, 34, 64, 44)
    assert bands
    assert _without_native_rows([missing, same_row], bands) == [missing]


def test_native_recovery_geometry_ignores_trailing_line_breaks():
    line = {"bbox": (10, 5, 120, 15), "spans": [{"chars": [
        {"char": "A", "bbox": (10, 5, 18, 15)},
        {"char": "\r", "bbox": (120, 10, 120, 10)},
        {"char": "\n", "bbox": (120, 10, 120, 10)},
    ]}]}
    page = {"blocks": [{"lines": [line]}]}
    assert list(_native_lines(page, 2)) == [((20, 10, 36, 30), "A\r\n")]


def test_single_caption_between_title_and_image_has_body_owner():
    assert _between_layout_blocks((0, 20, 100, 30), [(0, 5, 100, 18)], [(0, 32, 100, 100)])
    assert not _between_layout_blocks((0, 20, 100, 30), [], [(0, 0, 100, 18), (0, 32, 100, 100)])
    assert not _between_layout_blocks((0, 20, 100, 30), [(150, 5, 250, 18)], [(0, 32, 100, 100)])


def test_pale_bleed_through_is_not_foreground_but_gray_text_is():
    image = np.full((20, 100, 3), 250, dtype=np.uint8)
    image[5:15, 10:40] = 225
    assert not _has_foreground_ink(image, (0, 0, 100, 20))
    image[5:15, 10:40] = 170
    assert _has_foreground_ink(image, (0, 0, 100, 20))
    image[:] = 60
    image[5:15, 10:40] = 240
    assert _has_foreground_ink(image, (0, 0, 100, 20))
from rapid_doc.utils.enum_class import CategoryId, ContentType
from rapid_doc.utils.span_pre_proc import (
    remove_overlaps_low_confidence_spans,
    remove_overlaps_min_spans,
)


@pytest.mark.parametrize("category,field", [
    (CategoryId.OcrText, "text"),
    (CategoryId.InterlineEquation_YOLO, "latex"),
])
def test_full_page_line_cannot_replace_smaller_recognized_content(category, field):
    item = {"category_id": category, "poly": [50, 0, 90, 0, 90, 10, 50, 10], field: "w^2+3w+4"}
    assert _conflicts_with_recognized((0, 0, 100, 10), [item])
    assert not _conflicts_with_recognized((0, 11, 100, 21), [item])
    item[field] = ""
    assert not _conflicts_with_recognized((0, 0, 100, 10), [item])


@pytest.mark.parametrize("remove", [
    remove_overlaps_low_confidence_spans, remove_overlaps_min_spans,
])
@pytest.mark.parametrize("kind", [
    ContentType.TEXT, ContentType.INLINE_EQUATION, ContentType.INTERLINE_EQUATION,
])
@pytest.mark.parametrize("reverse", [False, True])
def test_recognized_content_wins_over_larger_high_confidence_recovery(remove, kind, reverse):
    existing = {"type": kind, "bbox": [50, 0, 90, 10], "content": "w^2", "score": 0.7}
    recovered = {
        "type": ContentType.TEXT, "bbox": [0, 0, 100, 10],
        "content": "49.w2", "score": 0.99, "layout_recovered_text": True,
    }
    spans = [existing, recovered]
    if reverse:
        spans.reverse()
    kept, dropped = remove(spans)
    assert kept == [existing]
    assert dropped == [recovered]
