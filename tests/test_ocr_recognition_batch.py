import numpy as np

from rapid_doc.model.ocr.recognition_batch import batch_ranges


def test_short_title_is_not_padded_to_long_text_width():
    ratios = [438 / 79, 20, 22, 24, 10, 8, 5]
    indices = np.argsort(ratios)
    ranges = list(batch_ranges(ratios, indices, 6))
    batches = [list(indices[a:b]) for a, b in ranges]
    title = next(b for b in batches if 0 in b)
    assert max(48 * ratios[i] for i in title) <= 320
    assert [i for b in batches for i in b] == list(indices)
    assert all(0 < len(b) <= 6 for b in batches)


def test_same_width_crops_still_fill_batches():
    assert list(batch_ranges([5] * 13, list(range(13)), 6)) == [(0, 6), (6, 12), (12, 13)]
