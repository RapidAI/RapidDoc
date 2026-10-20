"""表格底色与原生图表的冲突判定，覆盖局部候选及完整页面输出。"""

from io import BytesIO
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics, pipeline
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _native_multicolumn_grid_bboxes
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout


def _source(payload):
    with PDFDocument(payload) as document:
        return pipeline._collect_document_sources(document).page_sources[0]


def _merged_grid(scale=1.0, left=40, color=(0.82, 0.82, 0.82)):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(650 * scale, 800 * scale))
    canvas.scale(scale, scale)
    canvas.setFont("Helvetica", 9)
    columns = [left + x for x in (0, 110, 200, 270, 400)]
    # 每层表头的物理列不同；只有后续数据行共享列轨。
    bands = [(590, 560, [columns[0], columns[2], columns[-1]]),
             (560, 540, [columns[0], columns[-1]]),
             (540, 510, columns[:3] + columns[-1:]), (510, 480, columns),
             (480, 450, columns), (450, 420, columns)]
    for row, (top, bottom, tracks) in enumerate(bands):
        if row < 3:
            canvas.setFillColorRGB(*color)
            for a, b in zip(tracks, tracks[1:]):
                canvas.rect(a + 3, bottom + 3, b - a - 6, top - bottom - 6, stroke=0, fill=1)
        canvas.setFillColorRGB(0, 0, 0)
        for col, (a, b) in enumerate(zip(tracks, tracks[1:])):
            canvas.drawString(a + 6, bottom + 10, f"R{row}C{col}")
        for x in tracks:
            canvas.line(x, bottom, x, top)
    for y in (590, 560, 540, 510, 480, 450, 420):
        canvas.line(columns[0], y, columns[-1], y)
    canvas.save()
    bbox = tuple(v * scale for v in (left, 210, left + 400, 380))
    return _source(stream.getvalue()), bbox


@pytest.mark.parametrize("scale,left,color", [
    (1, 40, (.82, .82, .82)), (1.5, 85, (.2, .5, .8)), (.8, 110, (.7, .2, .1)),
])
def test_partial_merged_header_uses_complete_connected_table(scale, left, color):
    source, table = _merged_grid(scale, left, color)
    # 候选只覆盖表头和第一行数据，不能因未包含第二行数据丢掉表格。
    region = (table[0], table[1], table[2], 320 * scale)
    assert _native_multicolumn_grid_bboxes(source, 9 * scale, region)
    blocks = pipeline._analyze_page_source(source)
    tables = [block for block in blocks if block["type"] == "table"]
    assert len(tables) == 1
    assert "R5C3" in str(tables[0]["content"])
    assert not any(block["type"] == "image" for block in blocks)


@pytest.mark.parametrize("kind", ["single_cell", "thin_band", "missing_outer", "missing_columns", "form"])
def test_surrounding_table_is_not_sufficient_to_override_graphic(kind):
    source, table = _merged_grid()
    region = (table[0], table[1], table[2], 320)
    if kind == "single_cell":
        region = (45, 285, 145, 375)
    elif kind == "thin_band":
        region = (45, 282, 430, 295)
    elif kind == "missing_outer":
        source.drawing_lines = [r for r in source.drawing_lines if r.orientation != "vertical" or abs(r.bbox[0] - table[0]) > 1]
    elif kind == "missing_columns":
        source.drawing_lines = [r for r in source.drawing_lines if r.orientation != "vertical" or abs(r.bbox[0] - table[0]) < 1 or abs(r.bbox[0] - table[2]) < 1]
    else:
        source.form_bboxes = [table]
    assert _native_multicolumn_grid_bboxes(source, 9, region) == []


@pytest.mark.parametrize("scale,left", [(1, 45), (1.5, 85), (.8, 110)])
@pytest.mark.parametrize("horizontal", [True, False])
@pytest.mark.parametrize("grid", [False, True])
def test_real_bar_charts_keep_graphic_classification(scale, left, horizontal, grid):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(800 * scale, 850 * scale))
    canvas.scale(scale, scale)
    canvas.setFont("Helvetica", 10)
    canvas.setStrokeColorRGB(.6, .6, .6)
    if grid:
        for x in (left, left + 60, left + 120, left + 180):
            canvas.line(x, 350, x, 550)
        for y in (350, 400, 450, 500, 550):
            canvas.line(left, y, left + 180, y)
    for row, value in enumerate((70, 110, 155)):
        canvas.setFillColorRGB(.25, .45, .8)
        if horizontal:
            y = 500 - row * 40
            canvas.rect(left, y, value, 15, stroke=0, fill=1)
            label = (left + value + 8, y)
        else:
            x = left + row * 40
            canvas.rect(x, 350, 15, value, stroke=0, fill=1)
            label = (x, 350 + value + 8)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.drawString(*label, str(value))
    canvas.save()
    source = _source(stream.getvalue())
    assert graphics._detect_native_bar_graphics(source, 10 * scale)
    assert graphics._detect_strong_graphic_bboxes(source)


def test_byd_shareholder_table_preserves_all_rows():
    payload = (Path(__file__).resolve().parents[2] / "demo/pdfs/比亚迪财报.pdf").read_bytes()
    page = analyze_docvortex_layout(payload)[2]
    tables = [block for block in page if block["type"] == "table"]
    matches = [block for block in tables if "HKSCC NOMINEES LIMITED" in str(block["content"])]
    assert len(matches) == 1
    assert all(text in str(matches[0]["content"]) for text in ("王传福", "吕向阳", "179,421,465"))
    assert not any(block["type"] == "image" and block["bbox"][1] > .7 for block in page)

