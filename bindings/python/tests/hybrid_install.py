"""混合 PDF 页面路由、坐标、模型缺失和累计预算的安装测试。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import json
import sys
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import OnnxSession, RapidDocError, parse_pdf_ocr, ocr_png

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/crop-baseline.json").read_text(encoding="utf-8"))
db = json.loads((root / "build/migration/db-baseline.json").read_text(encoding="utf-8"))
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
detector = OnnxSession(runtime, root / db["model"], db["sha256"])
recognizer = OnnxSession(runtime, root / fixture["model"], fixture["sha256"])
options = dict(detector=detector, recognizer=recognizer)
source = fixture["hybrid_pdf"]
native = parse_pdf_ocr(source, end_page=1)
assert native.text == "Native text page"
assert native.pages[0]["method"] == "native"
blank = parse_pdf_ocr(source, start_page=2)
assert blank.text == "" and blank.pages[0]["method"] == "native"
result = parse_pdf_ocr(source, **options)
assert [p["method"] for p in result.pages] == ["native", "ocr", "native"]
assert [p["page_index"] for p in result.pages] == [0, 1, 2]
assert result.pages[1]["text"] == "RapidDoc 123\n中文识别测试"
assert result.document["pages"][1]["text"] == ""
assert any(
    d["code"] == "ocr_completed" and d["page_index"] == 1 for d in result.document["diagnostics"]
)
expected = ocr_png(fixture["png_cases"][0]["input"], **options)
for line, pixel in zip(result.pages[1]["ocr_lines"], expected):
    assert line["points"] == [[x / 2, y / 2] for x, y in pixel["points"]]
    assert line["text"] == pixel["text"]
stream = BytesIO(Path(source).read_bytes())
selected = parse_pdf_ocr(stream, start_page=1, end_page=2, **options)
assert not stream.closed and selected.pages == result.pages[1:2]
for code, path, kwargs in [
    ("model_missing", source, {}),
    ("resource_limit", fixture["pdf"], dict(max_ocr_pages=2, **options)),
    ("resource_limit", source, dict(max_pixels=100, **options)),
    ("resource_limit", fixture["pdf"], dict(max_total_pixels=640 * 256, **options)),
    ("invalid_input", source, dict(dpi=float("nan"), **options)),
]:
    try:
        parse_pdf_ocr(path, **kwargs)
    except RapidDocError as error:
        assert error.code == code, (error.code, code)
    else:
        raise AssertionError(f"expected {code}")
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(lambda _: parse_pdf_ocr(source, **options).text, range(3)))
assert results == [result.text] * 3
print(
    "Installed wheel: native/OCR/blank routing, source page range, point coordinates, stream ownership, budgets and concurrency pass"
)
