from io import BytesIO

import pypdf.filters
import pytest
from pypdf import PdfReader
from reportlab.pdfgen import canvas

from rapid_doc.utils.draw_bbox import (
    draw_layout_bbox,
    draw_line_sort_bbox,
    draw_span_bbox,
)


@pytest.mark.parametrize("draw", [draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox])
def test_large_stream_draws_without_changing_process_limit(draw, tmp_path, monkeypatch):
    pdf = BytesIO()
    writer = canvas.Canvas(pdf, pagesize=(200, 200), pageCompression=0)
    for i in range(50):
        writer.drawString(10, 10 + i, "original document content")
    writer.save()
    info = [{
        "discarded_blocks": [],
        "para_blocks": [{"type": "text", "bbox": [10, 10, 100, 50]}],
        "preproc_blocks": [{"type": "text", "lines": [{
            "bbox": [10, 10, 100, 50], "index": 0,
            "spans": [{"type": "text", "bbox": [10, 10, 100, 50]}],
        }]}],
    }]
    monkeypatch.setattr(pypdf.filters, "MAX_DECLARED_STREAM_LENGTH", 100)
    draw(info, pdf.getvalue(), tmp_path, "result.pdf")
    assert pypdf.filters.MAX_DECLARED_STREAM_LENGTH == 100
    monkeypatch.undo()
    page = PdfReader(tmp_path / "result.pdf").pages[0]
    assert "original document content" in page.extract_text()
    assert b" re" in page.get_contents().get_data()
