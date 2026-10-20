"""含矢量图标的框线表和连续目录页不能丢失结构或正文。"""

from dataclasses import replace
from io import BytesIO
import base64
import re

import pytest
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline, table_detection
from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout, _block_text


def icon_table_pdf(scale=1, left=35, media="vector"):
    stream = BytesIO()
    c = Canvas(stream, pagesize=(480 * scale, 520 * scale))
    c.beginForm("icon", 0, 0, 40, 30)
    if media != "text":
        c.setFillColorRGB(.95, .8, .1)
        c.circle(12, 15, 11, fill=1)
        c.rect(23, 7, 16, 16)
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(24, 12, "1x")
    c.endForm()
    c.scale(scale, scale)
    c.setFont("Helvetica", 9)
    for x in (left, left + 100, left + 200, left + 300, left + 400):
        c.line(x, 180, x, 480)
    for y in (480, 455, 400, 345, 290, 235, 180):
        c.line(left, y, left + 400, y)
    for col, label in enumerate(("Symbol", "Description", "Symbol", "Description")):
        c.drawString(left + col * 100 + 5, 463, label)
    for row in range(5):
        for col in (0, 2):
            c.saveState()
            c.translate(left + col * 100 + 25, 411 - row * 55)
            if media == "raster" or (media == "mixed" and col == 2):
                c.drawImage(ImageReader(Image.new("RGB", (40, 30), "black")), 0, 0, 40, 30)
            else:
                c.doForm("icon")
            c.restoreState()
            c.drawString(left + (col + 1) * 100 + 5, 432 - row * 55, f"Action {row}-{col}")
    c.save()
    return stream.getvalue()


@pytest.mark.parametrize("scale,left,media", [(1, 35, "vector"), (1.4, 20, "vector"),
                                              (.75, 60, "vector"), (1, 35, "raster"), (1, 35, "mixed")])
def test_vector_icons_stay_in_correct_table_cells(scale, left, media):
    page = analyze_docvortex_layout(icon_table_pdf(scale, left, media))[0]
    tables = [b for b in page if b["type"] == "table"]
    assert len(tables) == 1
    markup = tables[0]["content"]
    assert isinstance(markup, str)
    rows = re.findall(r"<tr>(.*?)</tr>", markup, re.S)
    assert len(rows) == 6
    for row, html in enumerate(rows):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", html, re.S)
        assert len(cells) == 4
        if row:
            for col in (0, 2):
                assert 'src="data:image/png;base64,' in cells[col]
                assert "Action" not in cells[col]
                assert f"Action {row-1}-{col}" in cells[col + 1]
                # 图标内部文字留在截图里，不能在同一单元格重复输出。
                assert "1x" not in re.sub(r"<[^>]+>", "", cells[col])
                data = re.search(r"base64,([^\"]+)", cells[col]).group(1)
                with Image.open(BytesIO(base64.b64decode(data))) as icon:
                    assert icon.width > 20 and icon.height > 20
    assert not any(b["type"] == "image" for b in page)


def test_enclosing_form_still_excludes_nested_table():
    with PDFDocument(icon_table_pdf()) as doc:
        source = pipeline._collect_document_sources(doc).page_sources[0]
    source = replace(source, form_bboxes=[(35, 40, 435, 340)])
    assert table_detection._detect_table_candidates(source) == []


def test_text_only_forms_remain_native_text_in_table_cells():
    blocks = analyze_docvortex_layout(icon_table_pdf(media="text"))[0]
    table = next(b for b in blocks if b["type"] == "table")
    assert table["content"].count("1x") == 10
    assert "<img" not in table["content"]
    assert table["content"].count("<tr>") == 6


def toc_pdf():
    stream = BytesIO()
    c = Canvas(stream, pagesize=(600, 800))
    for page in range(4):
        c.setFont("Helvetica", 9)
        c.drawString(520, 775, "Contents")
        for row in range(40):
            # 同位置、近似长度的不同目录名，不能让点线主导重复文本匹配。
            label = f"Topic {chr(65+page)}{row:02}"
            c.drawString(55, 747 - row * 18, label + "." * 85 + f" {page*100+row+1}")
        c.drawString(290, 20, ("i", "ii", "iii", "iv")[page])
        c.showPage()
    c.save()
    return stream.getvalue()


def test_toc_edges_survive_while_running_headers_and_page_numbers_are_filtered():
    pages = analyze_docvortex_layout(toc_pdf())
    for page, blocks in enumerate(pages):
        retained = "\n".join(_block_text(b["content"]) for b in blocks if b["type"] not in {"header", "footer", "page_number"})
        for row in range(40):
            assert f"Topic {chr(65+page)}{row:02}" in retained
        assert any(b["type"] == "header" and "Contents" in _block_text(b["content"]) for b in blocks)
        assert any(b["type"] == "page_number" for b in blocks)


@pytest.mark.parametrize("leader", [".", "．", "·", "…", ". "])
def test_leader_entries_are_not_marginal_candidates(leader):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.auxiliary_text import _build_marginal_candidate
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem

    line = _LineItem("概述" + leader * 20 + " 644", (40, 765, 550, 775), 0, 0, effective_height=10)
    assert _build_marginal_candidate(0, line, (600, 800)) is None


def test_table_image_render_failure_preserves_icon_text(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("test renderer unavailable")

    monkeypatch.setattr(PDFDocument, "render_page", fail)
    pages = analyze_docvortex_layout(icon_table_pdf())
    table = next(b for b in pages[0] if b["type"] == "table")
    assert table["content"].count("1x") == 10
    assert "docvortex-table-image" not in table["content"]
    assert "_table_images" not in table


def test_inline_table_images_are_written_once_with_pdf_relative_paths():
    from rapid_doc.backend.utils.html_image_utils import replace_inline_base64_img_src
    from rapid_doc.data.data_reader_writer import DummyDataWriter

    class Writer(DummyDataWriter):
        def __init__(self):
            self.files = {}
            self.writes = 0

        def write(self, path, data):
            self.files[path] = data
            self.writes += 1

    table = next(b for b in analyze_docvortex_layout(icon_table_pdf())[0] if b["type"] == "table")
    writer = Writer()
    saved = replace_inline_base64_img_src(table["content"], writer, 0, image_path_prefix="images/")
    paths = re.findall(r'<img src="images/([^\"]+)"', saved)
    assert len(paths) == 10
    assert "base64," not in saved
    assert writer.writes == len(set(paths))
    for path in paths:
        with Image.open(BytesIO(writer.files[path])) as img:
            assert img.format == "PNG"


@pytest.mark.parametrize("angle", [90, 270])
def test_rotated_table_keeps_images_in_symbol_columns(angle):
    import fitz

    with fitz.open(stream=icon_table_pdf(), filetype="pdf") as pdf:
        pdf[0].set_rotation(angle)
        payload = pdf.tobytes()
    blocks = analyze_docvortex_layout(payload)[0]
    table = next(b for b in blocks if b["type"] == "table")
    rows = re.findall(r"<tr>(.*?)</tr>", table["content"], re.S)
    assert len(rows) == 6
    assert table["content"].count("<img") == 10
    for row in rows[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        assert len(cells) == 4
        assert "<img" in cells[0] and "<img" in cells[2]
