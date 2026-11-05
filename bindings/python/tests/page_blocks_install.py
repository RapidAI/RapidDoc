"""已安装 wheel 的正文/表格有序页面块；正文对照独立旧 CPU 参考。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
import hashlib, json, math, sys
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    recognize_page_png,
    recognize_pdf_page,
    recognize_page_tables_png,
)
from rapid_doc_rust._native import _PageImage

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("page-blocks")
page_tables = load("page-tables")
cls = load("table-classification")
wired = load("wired-pipeline")
wireless = load("table-match")
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
model = lambda path, sha: OnnxSession(runtime, root / path, sha)
layout = model(page_tables["model"], page_tables["sha256"])
detector = model(wired["det_model"], wired["det_sha256"])
recognizer = model(wired["rec_model"], wired["rec_sha256"])
classifier = model(cls["model"], cls["sha256"])
secondary = model(cls["qmodel"], cls["qsha256"])
unet = model(wired["unet_model"], wired["unet_sha256"])
case = next(v for v in wireless["png_cases"] if v["variant"] == "slanet-plus")
slanet = model(case["model"], case["sha256"])
options = dict(
    detector=detector,
    recognizer=recognizer,
    classifier=classifier,
    secondary_classifier=secondary,
    wired_model=unet,
    wireless_model=slanet,
)


class FixedLayout:
    def __init__(self, regions):
        self.regions = regions
        self.calls = 0

    def layout_png(self, data, **kwargs):
        self.calls += 1
        return self.regions


saved = {}
text_count = 0
for case in fixture["cases"]:
    selected = FixedLayout(case["layout"]) if case["synthetic"] else layout
    result = recognize_page_png(case["input"], layout_model=selected, **options)
    if case["synthetic"]:
        assert selected.calls == 1
        assert [b["layout_index"] for b in result["blocks"]] == [1, 2, 0, 3]
        assert [b["kind"] for b in result["blocks"]] == [
            "text",
            "unsupported",
            "text",
            "unsupported",
        ]
        assert [d["layout_index"] for d in result["diagnostics"]] == [2, 3]
    else:
        saved[Path(case["input"]).stem] = result
    assert len(result["blocks"]) == len(case["layout"])
    assert result["coordinate_space"] == "page_pixels"
    assert result["text"] == "\n\n".join(b["text"] for b in result["blocks"] if b["text"])
    assert result["crop_pixels"] == sum(
        b["crop"]["width"] * b["crop"]["height"] for b in result["blocks"] if "crop" in b
    )
    blocks = {b["layout_index"]: b for b in result["blocks"]}
    image = _PageImage(Path(case["input"]).read_bytes())
    for reference in case["text_regions"]:
        text_count += 1
        actual = blocks[reference["layout_index"]]
        assert actual["kind"] == "text" and actual["crop"]["bounds"] == reference["bounds"]
        assert (
            hashlib.sha256(image.crop_rgb(actual["crop"]["bounds"])).hexdigest()
            == reference["rgb_sha256"]
        )
        assert len(actual["local_lines"]) == len(reference["lines"])
        x, y = reference["bounds"][:2]
        for local, page, expected in zip(
            actual["local_lines"], actual["lines"], reference["lines"]
        ):
            assert local["text"] == expected["text"]
            assert math.isclose(local["confidence"], expected["confidence"], abs_tol=1e-5)
            assert all(
                abs(a - b) < 1e-4
                for p, q in zip(local["points"], expected["points"])
                for a, b in zip(p, q)
            )
            assert page == dict(
                text=local["text"],
                confidence=local["confidence"],
                points=[[a + x, b + y] for a, b in local["points"]],
            )
        assert actual["text"] == "\n".join(line["text"] for line in actual["lines"])
    if not case["synthetic"]:
        tables = recognize_page_tables_png(case["input"], layout_model=layout, **options)["tables"]
        assert len(tables) == sum(b["kind"] == "table" for b in result["blocks"])
        for table in tables:
            block = blocks[table["layout_index"]]
            assert block["recognition"] == table["recognition"]
            assert block["page_geometry"] == table["page_geometry"]
            assert block["html"] == table["recognition"]["html"]
assert text_count == 3 and any(
    b["text"] for b in saved["multiple"]["blocks"] if b["kind"] == "text"
)
pdf = recognize_pdf_page(
    page_tables["pdf_input"], layout_model=layout, page_index=1, dpi=72, **options
)
assert {k: v for k, v in pdf.items() if k not in ("page_index", "dpi")} == saved["single"]
assert pdf["page_index"] == 1 and pdf["dpi"] == 72
blank = recognize_pdf_page(
    page_tables["pdf_input"],
    layout_model=layout,
    page_index=0,
    dpi=144,
    detector=None,
    recognizer=None,
)
assert not blank["blocks"] and not blank["text"] and not blank["diagnostics"]

synthetic = fixture["cases"][-1]
fixed = FixedLayout(synthetic["layout"])
for limits in [dict(max_regions=0), dict(max_total_crop_pixels=1)]:
    try:
        recognize_page_png(
            synthetic["input"], layout_model=fixed, detector=None, recognizer=None, **limits
        )
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("whole-page budget was not checked before OCR")
for limits in [
    dict(crop_grid=0),
    dict(max_regions=1001),
    dict(max_total_crop_pixels=128000001),
    dict(min_confidence=float("nan")),
]:
    try:
        recognize_page_png(
            synthetic["input"], layout_model=None, detector=None, recognizer=None, **limits
        )
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError("invalid page configuration accepted")
only_unsupported = FixedLayout([synthetic["layout"][2]])
result = recognize_page_png(
    synthetic["input"], layout_model=only_unsupported, detector=None, recognizer=None
)
assert (
    result["crop_pixels"] == 0
    and result["blocks"][0]["kind"] == "unsupported"
    and result["diagnostics"]
)
try:
    recognize_page_png(synthetic["input"], layout_model=fixed, detector=None, recognizer=None)
except ValueError as e:
    assert "detector" in str(e)
else:
    raise AssertionError("missing text models accepted")
filtered = recognize_page_png(
    synthetic["input"],
    layout_model=fixed,
    detector=detector,
    recognizer=recognizer,
    min_confidence=1,
)
assert all(
    line["confidence"] >= 1
    for block in filtered["blocks"]
    if block["kind"] == "text"
    for line in block["lines"]
)
print(
    "Installed wheel: 3 independent CPU text-region references, stable model order/unsupported blocks, shared page budgets, table equivalence, 2 PDF page/DPI routes and confidence filtering passed"
)
