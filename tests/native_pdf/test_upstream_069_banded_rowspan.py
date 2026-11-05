"""居中跨行名称不能把相邻的独立数值记录桥接成一行。"""

from pathlib import Path
from dataclasses import replace

import pytest
from bs4 import BeautifulSoup

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import NativeTableInput, NativeTableRule, recover_native_pdf_table
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.aligned_numeric import regroup_numeric_text
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.banded_rowspan import build_banded_rowspan_candidate
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from test_native_pdf_table import _char_items, _local_to_page_bbox

SOURCE = Path(__file__).parent / "pdfs/grouped_tables/emnlp2022_grouped_tables.pdf"
BBOX = (308.9011717637695, 155.43338113439654, 521.6738891601562, 341.8959279439844)
EXPECTED = [
    ("0.3292", "0.2256", "0.5021"),
    ("0.0459", "0.1488", "0.4464"),
    ("0.5597", "0.0963", "0.3686"),
    ("0.1615", "0.0783", "0.3556"),
    ("0.2993", "0.4039", "0.7406"),
    ("0.0551", "0.3137", "0.7017"),
    ("0.5134", "0.4067", "0.6461"),
    ("0.1052", "0.2887", "0.5717"),
    ("0.4254", "0.2831", "0.5644"),
    ("0.0919", "0.2074", "0.5188"),
]


def _paired_input(cols=5, angle=0, scale=1.0):
    width, height = cols * 50.0, 122.0
    entries = [("Flag" if col == 1 else f"M{col}", (col * 50 + 8, 4, col * 50 + 25, 17.3)) for col in range(1, cols)]
    for group in range(3):
        top = 23.0 + group * 34
        entries.append((f"Set{group}", (8, top + 6.85, 28, top + 20.15)))
        for subrow in range(2):
            y = top + subrow * 13.7
            for col in range(1, cols):
                value = ("✓" if subrow == 0 else "✗") if col == 1 else f"{group + 2}.{col}{subrow}00"
                entries.append((value, (col * 50 + 8, y, col * 50 + 8 + len(value) * 3.5, y + 13.3)))
    page_width, page_height = (height, width) if angle in {90, 270} else (width, height)

    def transform(box):
        rotated = _local_to_page_bbox(box, page_width, page_height, angle)
        return tuple(v * scale + (17 if i % 2 == 0 else 23) for i, v in enumerate(rotated))

    chars = tuple(
        {**c, "bbox": transform(c["bbox"]), "font": {"name": "DifferentFont", "size": 13.3 * scale}}
        for c in _char_items(entries)
    )
    rules = tuple(
        NativeTableRule(
            transform((0, y - 0.2, width, y + 0.2)), 0.4 * scale, "vertical" if angle in {90, 270} else "horizontal"
        )
        for y in (0.2, 20, 54, 88, 121.8)
    )
    bbox = (17, 23, 17 + page_width * scale, 23 + page_height * scale)
    return NativeTableInput(bbox, (bbox[2] + 10, bbox[3] + 10), angle, chars, rules)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("cols,scale", [(4, 0.75), (5, 1.0), (6, 1.8)])
def test_centered_rowspan_is_independent_of_words_columns_rotation_and_scale(cols, scale, angle):
    result = recover_native_pdf_table(_paired_input(cols, angle, scale))
    assert result is not None
    assert (result.rows, result.cols) == (7, cols)
    assert [(c.row, c.col, c.rowspan, c.content) for c in result.cells if c.rowspan > 1] == [
        (1 + group * 2, 0, 2, f"Set{group}") for group in range(3)
    ]
    for group in range(3):
        for subrow in range(2):
            row = 1 + group * 2 + subrow
            assert [c.content for c in result.cells if c.row == row and c.col >= 2] == [
                f"{group + 2}.{col}{subrow}00" for col in range(2, cols)
            ]
    sources = [i for c in result.cells for i in c.source_char_indices]
    assert len(sources) == len(set(sources)) == len(result.text.glyphs)


@pytest.mark.parametrize(
    "damage",
    ["missing_rule", "partial_rule", "missing_value", "prose", "vertical", "tall_formula", "aligned_label", "multiline_label"],
)
def test_reject_unproven_rowspan(damage):
    table = _paired_input()
    rules, chars = list(table.drawing_lines), list(table.chars)
    if damage == "missing_rule":
        del rules[2]
    elif damage == "partial_rule":
        rules[2] = replace(rules[2], bbox=(17, 76.8, 130, 77.2))
    elif damage == "vertical":
        rules.append(NativeTableRule((61.8, 23, 62.2, 145), 0.4, "vertical"))
    elif damage == "missing_value":
        chars = [c for c in chars if not (125 <= c["bbox"][0] < 150 and 45 < c["bbox"][1] < 47)]
    elif damage == "prose":
        chars = [{**c, "char": "a"} if 125 <= c["bbox"][0] < 150 else c for c in chars]
    elif damage == "tall_formula":
        index = next(i for i, c in enumerate(chars) if c["char"] == "2" and c["bbox"][1] > 40)
        box = chars[index]["bbox"]
        chars[index] = {**chars[index], "bbox": (box[0], box[1] - 12, box[2], box[3] + 12)}
    elif damage == "aligned_label":
        chars = [
            {**c, "bbox": (c["bbox"][0], c["bbox"][1] - 6.85, c["bbox"][2], c["bbox"][3] - 6.85)} if c["bbox"][0] < 50 else c
            for c in chars
        ]
    else:
        # A second label line cannot be inferred as a single centered row label.
        for c in table.chars:
            if c["bbox"][0] < 50:
                x0, y0, x1, y1 = c["bbox"]
                chars.append({**c, "char_idx": len(chars), "bbox": (x0, y0 + 6, x1, y1 + 6)})
    table = replace(table, chars=tuple(chars), drawing_lines=tuple(rules))
    text = build_native_table_text(table)
    assert text is not None
    assert build_banded_rowspan_candidate(table, text, regroup_numeric_text(text)) is None
