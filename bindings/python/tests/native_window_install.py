"""Real multi-window native host: selection, per-window analysis and cancellation."""

import io
import json
from reportlab.pdfgen.canvas import Canvas
from rapid_doc_rust import _prepare_pdf
from rapid_doc_rust._native import PdfEngine

payload = io.BytesIO()
pdf = Canvas(payload, pagesize=(400, 400))
for i in range(7):
    pdf.drawString(60, 200, f"Unique body token {i + 1}")
    pdf.showPage()
pdf.save()
library, data = _prepare_pdf(payload.getvalue(), None, None)
request = dict(
    start_page=1,
    end_page=6,
    window=2,
    dpi=72,
    docvortex_mode="rule",
    native_only=True,
    native_text=True,
    formula_enabled=False,
    table_enabled=False,
    seal_enabled=False,
)


def unexpected_model(role):
    raise AssertionError(f"Native text fixture unexpectedly requested model: {role}")


events = []
metadata, assets = PdfEngine().parse(
    library, data, json.dumps(request), unexpected_model, lambda raw: events.append(json.loads(raw))
)
result = json.loads(metadata)
native = [e for e in events if e["stage"] == "native_analysis"]
assert [e["page_index"] for e in native] == [1, 3, 5], native
assert [e["detail"] for e in native] == ["pages 2-3 of 7", "pages 4-5 of 7", "pages 6-6 of 7"]
stages = [e["stage"] for e in events if e["stage"] in ("batch", "native_analysis")]
assert stages == ["batch", "native_analysis"] * 3, stages
assert result["page_range"] == [1, 6] and len(result["pages"]) == 5
assert [p["page_idx"] for p in result["pages"]] == list(range(5))
for i, page in enumerate(result["pages"]):
    assert f"Unique body token {i + 2}" in json.dumps(page), (i, page)
request["window"] = 5
single, single_assets = PdfEngine().parse(
    library, data, json.dumps(request), unexpected_model, lambda _: None
)
single = json.loads(single)
assert result["pages"] == single["pages"] and dict(assets) == dict(single_assets)


def cancel(raw):
    if json.loads(raw)["stage"] == "native_analysis":
        raise RuntimeError("cancel native window")


try:
    PdfEngine().parse(library, data, json.dumps(request), unexpected_model, cancel)
except RuntimeError as error:
    assert "cancel native window" in str(error)
else:
    raise AssertionError("Cancellation was swallowed or converted into a model fallback")
print(
    "Native windows: nonzero selection, partial last batch, contents, single-window parity, cancellation passed"
)
