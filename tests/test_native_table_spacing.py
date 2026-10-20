import pytest

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph, NativeTableInput
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_cell_text, build_native_table_text


@pytest.mark.parametrize('with_advance_bounds', [False, True])
def test_geometric_word_gap_survives_without_explicit_space(with_advance_bounds):
    chars = []
    for index, (text, left) in enumerate([('A', 5), ('B', 15)]):
        char = {'char': text, 'char_idx': index * 7, 'bbox': (left, 5, left+5, 15)}
        if with_advance_bounds:
            char['loose_bbox'] = (left, 4, left+6, 16)
        chars.append(char)
    text = build_native_table_text(NativeTableInput((0, 0, 100, 50), (100, 50), 0, tuple(chars)))
    assert build_cell_text(list(text.glyphs), text.median_glyph_height) == 'A B'
    assert [g.source_index for g in text.glyphs] == [0, 7]


@pytest.mark.parametrize('space', [False, True])
def test_explicit_numeric_space_is_preserved_even_with_touching_advance_bounds(space):
    chars = [{'char': '1', 'bbox': (1, 5, 3, 12), 'loose_bbox': (0, 2, 5, 14)}]
    if space:
        chars.append({'char': ' ', 'bbox': (4, 5, 5, 12)})
    chars.append({'char': '2', 'bbox': (6, 5, 9, 12), 'loose_bbox': (5, 2, 10, 14)})
    text = build_native_table_text(NativeTableInput((0, 0, 100, 50), (100, 50), 0, tuple(chars)))
    assert build_cell_text(list(text.glyphs), text.median_glyph_height) == ('1 2' if space else '12')


def test_spacing_uses_local_font_height_in_a_mixed_size_table():
    glyphs = [
        NativeTableGlyph(0, 0, 'A', (0, 0, 2, 4), 0, spacing_bbox=(0, 0, 3, 6)),
        NativeTableGlyph(1, 1, 'B', (5, 0, 7, 4), 0, spacing_bbox=(4.5, 0, 7.5, 6)),
    ]
    # A large title elsewhere must not raise the threshold for these small words.
    assert build_cell_text(glyphs, median_height=40) == 'A B'
