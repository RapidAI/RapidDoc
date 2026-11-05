"""原生/扫描逐行融合：独立 PDF 字符事实与旧 CPU 检测/识别参考。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from io import BytesIO
from pathlib import Path
import hashlib, json, math, sys
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    recognize_pdf_page_hybrid,
    render_pdf_page,
    capabilities,
)
from rapid_doc_rust._native import _PageImage

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("native-fusion")
models = load("wired-pipeline")
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
detector = OnnxSession(runtime, root / models["det_model"], models["det_sha256"])
recognizer = OnnxSession(runtime, root / models["rec_model"], models["rec_sha256"])


class FixedLayout:
    def __init__(self, width, height):
        self.regions = [
            dict(cls_id=22, label="text", score=1.0, coordinate=[0, 0, width, height], order=1)
        ]
        self.calls = 0

    def layout_png(self, *args, **kwargs):
        self.calls += 1
        return self.regions


class CountRecognizer:
    def __init__(self):
        self.count = 0

    def recognize_regions_png(self, data, regions):
        self.count += len(regions)
        return recognizer.recognize_regions_png(data, regions)


expected = {
    0: ["native"] * 3,
    1: ["native", "ocr", "native"],
    2: ["ocr"],
    3: ["ocr"],
    4: ["ocr"],
    5: [],
}
counter = CountRecognizer()
saved = {}
for case in fixture["cases"]:
    layout = FixedLayout(case["width"], case["height"])
    result = recognize_pdf_page_hybrid(
        fixture["pdf_input"],
        experimental=True,
        layout_model=layout,
        detector=detector,
        recognizer=counter,
        page_index=case["page_index"],
        dpi=case["dpi"],
    )
    assert layout.calls == 1 and result["coordinate_space"] == "page_pixels"
    assert (result["width"], result["height"]) == (case["width"], case["height"])
    assert len(result["blocks"]) == 1
    lines = result["blocks"][0]["lines"]
    local = result["blocks"][0]["local_lines"]
    assert [line["method"] for line in lines] == expected[case["page_index"]], result[
        "native_routes"
    ]
    assert len(lines) == len(case["ocr"]) and local == lines
    baseline = fixture["facts"]["pages"][case["page_index"]]["chars"]
    assert result["native_page"]["text"] == "".join(g["char"] for g in baseline)
    assert len(result["native_page"]["characters"]) == len(baseline)
    for line, reference in zip(lines, case["ocr"]):
        assert all(
            abs(a - b) < 1e-4
            for p, q in zip(line["points"], reference["points"])
            for a, b in zip(p, q)
        )
        if line["method"] == "native":
            assert line["confidence"] is None and line["reason"] == "native_coverage"
            assert line["text"] == "".join(baseline[i]["char"] for i in line["character_indices"])
        else:
            assert line["text"] == reference["text"]
            assert math.isclose(line["confidence"], reference["confidence"], abs_tol=1e-5)
    assert result["text"] == "\n".join(line["text"] for line in lines)
    page = render_pdf_page(fixture["pdf_input"], page_index=case["page_index"], dpi=case["dpi"])
    image = _PageImage(page.png)
    assert (
        hashlib.sha256(image.crop_rgb([0, 0, image.width, image.height])).hexdigest()
        == case["rgb_sha256"]
    )
    saved[case["page_index"], case["dpi"]] = result
assert counter.count == 4, counter.count
assert saved[2, 72]["native_routes"][0]["reason"] == "raster_overlap"
assert saved[4, 72]["native_routes"][0]["reason"] == "non_horizontal"

case = fixture["cases"][0]
options = dict(layout_model=FixedLayout(case["width"], case["height"]), detector=detector, dpi=72)
native = recognize_pdf_page_hybrid(fixture["pdf_input"], experimental=True, **options)
assert native["text"] == saved[0, 72]["text"]  # 没有 rec 会话也能完成全部原生行。
offset = FixedLayout(600, 500)
offset.regions[0]["coordinate"] = [40, 40, 450, 220]
cropped = recognize_pdf_page_hybrid(
    fixture["pdf_input"], experimental=True, layout_model=offset, detector=detector, dpi=72
)
assert cropped["text"] == native["text"]
block = cropped["blocks"][0]
assert block["crop"]["bounds"] == [40, 40, 450, 220]
for local, page in zip(block["local_lines"], block["lines"]):
    assert page == dict(local, points=[[x + 40, y + 40] for x, y in local["points"]])
source = BytesIO(Path(fixture["pdf_input"]).read_bytes())
assert (
    recognize_pdf_page_hybrid(source, experimental=True, **options)["text"] == native["text"]
    and not source.closed
)
try:
    recognize_pdf_page_hybrid(fixture["pdf_input"], experimental=True, page_index=1, **options)
except ValueError as e:
    assert "recognizer" in str(e)
else:
    raise AssertionError("missing OCR fallback recognizer accepted")

for value in [1, native["native_work"] - 1]:
    try:
        recognize_pdf_page_hybrid(
            fixture["pdf_input"],
            experimental=True,
            max_native_work=value,
            recognizer=counter,
            **options,
        )
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("native work budget ignored")
split = FixedLayout(600, 500)
split.regions = [
    dict(cls_id=22, label="text", score=1.0, coordinate=b, order=i + 1)
    for i, b in enumerate([[0, 0, 600, 150], [0, 150, 600, 500]])
]
try:
    recognize_pdf_page_hybrid(
        fixture["pdf_input"],
        experimental=True,
        layout_model=split,
        detector=detector,
        max_native_work=native["native_work"] - 1,
    )
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("native work budget reset between regions")
for value in [-1, 10_000_001, float("nan")]:
    try:
        recognize_pdf_page_hybrid(
            b"not PDF", experimental=True, layout_model=None, detector=None, max_native_work=value
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid budget not checked before PDF runtime")
filtered = recognize_pdf_page_hybrid(
    fixture["pdf_input"],
    experimental=True,
    page_index=1,
    min_confidence=1,
    recognizer=counter,
    **options,
)
assert [line["text"] for line in filtered["blocks"][0]["lines"] if line["method"] == "native"] == [
    "Native title 123",
    "Native footer 789",
]
assert capabilities()["pdf_native_line_fusion"]
print(
    "Installed wheel: 7 page/DPI references; 8 native lines preserve source characters, only 4 OCR fallbacks; same-line raster, crop offsets, rotation, fractional DPI, blank, stream and cumulative work budgets passed"
)
