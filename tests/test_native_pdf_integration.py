from io import BytesIO
from unittest.mock import Mock

import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from rapid_doc.model.docvortex_adapter import (
    analyze_docvortex_layout, convert_docvortex_layout, recover_docvortex_table,
)
from rapid_doc.model.layout.rapid_layout import RapidLayoutModel
from rapid_doc.utils.enum_class import CategoryId


def table_pdf():
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(200, 160))
    for x in (20, 70, 120, 170):
        canvas.line(x, 50, x, 140)
    for y in (50, 80, 110, 140):
        canvas.line(20, y, 170, y)
    for r in range(3):
        for c in range(3):
            canvas.drawString(25 + c * 50, 125 - r * 30, f"{r}{c}")
    canvas.save()
    return stream.getvalue()


def test_table_evidence_remains_valid_after_document_close_and_is_reused():
    page = {"_source_pdf_bytes": table_pdf(), "_source_page_index": 0}
    first = recover_docvortex_table(page, (20, 20, 170, 110))
    evidence = page["_docvortex_table_page"]
    # Subsequent regions must use detached evidence, rather than open the source again.
    page["_source_pdf_bytes"] = b"invalid"
    second = recover_docvortex_table(page, (20, 20, 170, 110))
    assert first and second and first.html == second.html
    assert "22" in first.html
    assert page["_docvortex_table_page"] is evidence
    assert "_docvortex_table_page" not in {"_source_pdf_bytes": table_pdf()}


def test_document_native_tables_convert_to_pixel_contract():
    blocks = analyze_docvortex_layout(table_pdf())[0]
    layout = convert_docvortex_layout(blocks, 400, 320)
    tables = [item for item in layout if item["original_label"] == "table"]
    assert tables and "22" in tables[0]["html"]
    assert tables[0]["table_parse_method"] == "docvortex_native"
    assert [item["original_order"] for item in layout] == sorted(item["original_order"] for item in layout)
    for item in layout:
        x0, y0, x1, _, _, y1, *_ = item["poly"]
        assert 0 <= x0 < x1 <= 400 and 0 <= y0 < y1 <= 320


def test_mixed_document_preserves_pages_and_text_rotation():
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 400))
    for i in range(8):
        canvas.drawString(20, 340 - i * 18, f"Left column row {i}")
        canvas.drawString(230, 340 - i * 18, f"Right column row {i}")
    canvas.saveState()
    canvas.translate(380, 80)
    canvas.rotate(90)
    canvas.drawString(0, 0, "Rotated text")
    canvas.restoreState()
    canvas.showPage()
    canvas.drawImage(ImageReader(Image.new("RGB", (100, 100), "gray")), 0, 0, 400, 400)
    canvas.showPage()
    canvas.drawString(20, 350, "Text after scanned page")
    canvas.save()
    pages = analyze_docvortex_layout(stream.getvalue())
    assert len(pages) == 3
    assert "Left column row" in str(pages[0]) and "Right column row" in str(pages[0])
    assert "Text after scanned page" in str(pages[2])
    rotation = [block for block in pages[0] if block.get("angle")]
    assert rotation and "Rotated text" in str(rotation)
    assert any(item.get("native_text") == "Rotated text" for item in convert_docvortex_layout(rotation, 400, 400))


def test_rule_route_keeps_good_pages_and_falls_back_only_for_missing_layout(monkeypatch):
    monkeypatch.setattr("rapid_doc.model.layout.rapid_layout.get_device", lambda: "cpu")
    model = RapidLayoutModel({"model_type": "pp_doclayoutv2", "docvortex_mode": "rule"})
    assert model.model is None
    fallback = Mock()
    fallback.batch_predict.return_value = [[{"fallback": True}]]
    model._fallback_model = fallback
    images = [np.zeros((160, 200, 3), dtype=np.uint8) for _ in range(2)]
    results = model.batch_predict(images, 2, [
        {"_docvortex_layout": analyze_docvortex_layout(table_pdf())[0]}, {},
    ])
    assert results[0][0]["original_label"] == "table"
    assert results[1] == [{"fallback": True}]
    assert len(fallback.batch_predict.call_args.args[0]) == 1


def test_rule_plus_s_preserves_table_and_adds_seal(monkeypatch):
    monkeypatch.setattr("rapid_doc.model.layout.rapid_layout.get_device", lambda: "cpu")
    model = RapidLayoutModel({"model_type": "docvortex_rule", "docvortex_mode": "rule_plus_s"})
    supplement = Mock()
    supplement.batch_predict.return_value = [[{
        "category_id": CategoryId.ImageBody, "original_label": "seal", "score": .99,
        "poly": [175, 120, 195, 120, 195, 145, 175, 145],
    }]]
    model._supplement_model = supplement
    results = model.batch_predict([np.zeros((160, 200, 3), dtype=np.uint8)], 1, [
        {"_docvortex_layout": analyze_docvortex_layout(table_pdf())[0]},
    ])
    assert {item["original_label"] for item in results[0]} == {"table", "seal"}
