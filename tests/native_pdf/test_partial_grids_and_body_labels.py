"""无词表、无底色的局部正文保护，以及开底跨行单元格的完整网格恢复。"""

from dataclasses import replace
from io import BytesIO
import xml.etree.ElementTree as ET

import pytest
from reportlab.pdfgen.canvas import Canvas

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import auxiliary_text, pipeline
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_open_grid import detect_partial_rule_grids
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout, _block_text


def local_label_pdf(label, shaded=False, outer_header=True, continuation=False):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(600, 800))
    for page in range(3):
        canvas.setFont("Helvetica", 9)
        if outer_header:
            canvas.drawString(460, 778, "Service manual")
        if shaded:
            canvas.setFillColorRGB(.95, .84, .78)
            canvas.rect(40, 630, 520, 125, fill=1, stroke=0)
            canvas.setFillColorRGB(0, 0, 0)
        canvas.drawString(50, 742, label)
        canvas.drawString(50, 730, f"Unique instruction {page} must stay with its local label.")
        if continuation:
            canvas.drawString(50, 718, f"Different continuation {page} is part of the same paragraph.")
        canvas.drawString(50, 480, f"Main body on page {page} has plenty of ordinary text.")
        canvas.drawCentredString(300, 24, str(page + 1))
        canvas.showPage()
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize("label", ["XQZ-Label:", "Custom marker", "ZX-42"])
@pytest.mark.parametrize("shaded,outer_header,continuation", [(False, True, False),
                                                          (False, False, True),
                                                          (True, False, False)])
def test_arbitrary_local_labels_survive_marginal_classification(label, shaded, outer_header, continuation):
    payload = local_label_pdf(label, shaded, outer_header, continuation)
    with PDFDocument(payload) as doc:
        sources = pipeline._collect_document_sources(doc)
        auxiliary_text._classify_raw_page_marginals(sources.page_sources)
        for source in sources.page_sources:
            line = next(line for line in source.lines if line.text == label)
            assert line.marginal_body_context
            assert line.semantic_type is None
    pages = analyze_docvortex_layout(payload)
    for page in pages:
        assert any(label in _block_text(block) and block["type"] not in {"header", "footer", "page_number"}
                   for block in page)
        if outer_header:
            assert any(block["type"] == "header" and "Service manual" in _block_text(block) for block in page)


def test_isolated_repeated_unknown_header_is_still_marginal():
    payload = local_label_pdf("Unknown masthead", False, False, False)
    with PDFDocument(payload) as doc:
        sources = pipeline._collect_document_sources(doc)
        for source in sources.page_sources:
            source.lines = [line for line in source.lines if not line.text.startswith("Unique instruction")]
        auxiliary_text._classify_raw_page_marginals(sources.page_sources)
        for source in sources.page_sources:
            line = next(line for line in source.lines if line.text == "Unknown masthead")
            assert not line.marginal_body_context
            assert line.semantic_type == "header"


def merged_open_grid_pdf(scale=1, left=50, shaded=False, closed=False):
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(650 * scale, 800 * scale))
    canvas.scale(scale, scale)
    edges = [left + offset for offset in (0, 90, 190, 260, 330, 400, 470)]
    if shaded:
        canvas.setFillColorRGB(.6, .7, .9)
        canvas.rect(edges[0], 620, 470, 60, fill=1, stroke=0)
        # 底色跨行，并不构成覆盖真实数据列的额外单元格边界。
        canvas.rect(edges[0], 120, 90, 500, fill=1, stroke=0)
        canvas.setFillColorRGB(0, 0, 0)
    for index, x in enumerate(edges):
        canvas.line(x, 120, x, 650 if index in (3, 5) else 680)
    canvas.line(edges[0], 680, edges[-1], 680)
    canvas.line(edges[2], 650, edges[-1], 650)
    canvas.line(edges[0], 620, edges[-1], 620)
    for y in (580, 540, 500, 460, 120):
        canvas.line(edges[1], y, edges[-1], y)
    if closed:
        canvas.line(edges[0], 120, edges[1], 120)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(edges[0] + 5, 650, "Paper types")
    canvas.drawString(edges[1] + 5, 650, "Size")
    canvas.drawString(edges[2] + 5, 665, "Feed group A")
    canvas.drawString(edges[4] + 5, 665, "Feed group B")
    for index in range(2, 6):
        canvas.drawString(edges[index] + 5, 630, f"Out {index}")
    for row, (top, bottom) in enumerate(zip((620, 580, 540, 500, 460), (580, 540, 500, 460, 120))):
        for col in range(1, 6):
            canvas.drawString(edges[col] + 5, (top + bottom) / 2, f"R{row}C{col}")
    for item in range(12):
        canvas.drawString(edges[0] + 5, 606 - 35 * item, f"Material {item}")
    canvas.save()
    return stream.getvalue()


@pytest.mark.parametrize("scale,left", [(1, 50), (.75, 70), (1.4, 20)])
@pytest.mark.parametrize("shaded,closed", [(False, False), (True, False), (False, True)])
def test_partial_row_lines_and_open_merged_cell_form_one_table(scale, left, shaded, closed):
    pages = analyze_docvortex_layout(merged_open_grid_pdf(scale, left, shaded, closed))
    tables = [block for block in pages[0] if block["type"] == "table"]
    assert len(tables) == 1
    table = ET.fromstring(tables[0]["content"])
    rows = table.findall(".//tr")
    assert len(rows) == 7
    assert rows[0][0].get("rowspan") == "2"
    assert rows[2][0].get("rowspan") == "5"
    for item in range(12):
        assert f"Material {item}" in "".join(rows[2][0].itertext())
    for row in range(5):
        data_cells = list(rows[row + 2])[1:] if row == 0 else list(rows[row + 2])
        assert len(data_cells) == 5
        for col, cell in enumerate(data_cells, start=1):
            assert f"R{row}C{col}" in "".join("".join(cell.itertext()).split())
    assert not any("Material" in _block_text(block) for block in pages[0] if block["type"] != "table")


@pytest.mark.parametrize("missing", ["outer", "column", "common_end", "excluded", "form"])
def test_partial_grid_requires_continuous_physical_evidence(missing):
    with PDFDocument(merged_open_grid_pdf()) as doc:
        source = pipeline._collect_document_sources(doc).page_sources[0]
    exclusions = []
    if missing == "outer":
        source.drawing_lines = [rule for rule in source.drawing_lines
                                if rule.orientation != "vertical" or abs(rule.bbox[0] - 50) > 1]
    elif missing == "column":
        source.drawing_lines = [rule for rule in source.drawing_lines
                                if rule.orientation != "vertical" or rule.bbox[0] < 51 or rule.bbox[0] > 519]
    elif missing == "common_end":
        source.drawing_lines = [replace(rule, bbox=(*rule.bbox[:3], rule.bbox[3] - 40))
                                if rule.orientation == "vertical" and rule.bbox[0] > 519 else rule
                                for rule in source.drawing_lines]
    elif missing == "excluded":
        exclusions = [(50, 120, 520, 680)]
    else:
        source.form_bboxes = [(50, 120, 520, 680)]
    assert detect_partial_rule_grids(source, exclusions) == []
