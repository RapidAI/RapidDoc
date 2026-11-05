"""真实 wheel：悬挂列表续行、拒绝段界、源行保留及独立 OCR/原生 PDF 参考。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib, json, sys
from rapid_doc_rust import (
    OnnxSession,
    organize_page_text,
    recognize_page_png,
    recognize_pdf_page_hybrid,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("list-paragraphs")


def check(page, groups, kinds, texts):
    before = copy.deepcopy(page)
    result = organize_page_text(page, experimental=True)
    assert page == before
    assert [p["line_indices"] for p in result["paragraphs"]] == groups, result
    assert [p["kind"] for p in result["paragraphs"]] == kinds, result
    assert [p["text"] for p in result["paragraphs"]] == texts, result
    assert sorted(i for p in result["paragraphs"] for i in p["line_indices"]) == list(
        range(len(page["blocks"][0]["lines"]))
    )
    assert result["text"] == "\n\n".join(t for t in texts if t)
    return result


for case in fixture["cases"]:
    for scale in (0.5, 1.0, 2.0):
        page = copy.deepcopy(case["page"])
        page["width"] = int(page["width"] * scale)
        page["height"] = int(page["height"] * scale)
        for block in page["blocks"]:
            block["region"]["coordinate"] = [v * scale for v in block["region"]["coordinate"]]
            for line in block["lines"]:
                line["points"] = [[x * scale, y * scale] for x, y in line["points"]]
        result = check(page, case["groups"], case["kinds"], case["texts"])
        source_order = organize_page_text(page, experimental=True, split_columns=False)
        expected_source_groups = (
            [[0], [1], [2], [3]] if case["name"] == "two-columns" else case["groups"]
        )
        assert [p["line_indices"] for p in source_order["paragraphs"]] == expected_source_groups
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(
            pool.map(lambda _: organize_page_text(case["page"], experimental=True), range(3))
        )
        assert all(o == outputs[0] for o in outputs)

runtime = fixture["runtime"]
models = load("wired-pipeline")
assert hashlib.sha256(Path(runtime["pdf_input"]).read_bytes()).hexdigest() == runtime["pdf_sha256"]
assert len(runtime["facts"]["pages"]) == 1
ort = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
detector = OnnxSession(ort, root / models["det_model"], models["det_sha256"])
recognizer = OnnxSession(ort, root / models["rec_model"], models["rec_sha256"])


class FullPage:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def layout_png(self, *args, **kwargs):
        return [
            dict(
                cls_id=22,
                label="text",
                score=1.0,
                order=0,
                coordinate=[0, 0, self.width, self.height],
            )
        ]


ocr = recognize_page_png(
    runtime["png_input"],
    layout_model=FullPage(runtime["width"], runtime["height"]),
    detector=detector,
    recognizer=recognizer,
)
actual = ocr["blocks"][0]["lines"]
assert len(actual) == len(runtime["ocr"])
for a, b in zip(actual, runtime["ocr"]):
    assert a["text"] == b["text"]
    assert a["points"] == b["points"]
    assert abs(a["confidence"] - b["confidence"]) < 1e-5
check(ocr, runtime["groups"], runtime["kinds"], runtime["expected_ocr"])
for dpi in (72, 137):
    width = round(runtime["width"] * dpi / 72)
    height = round(runtime["height"] * dpi / 72)
    hybrid = recognize_pdf_page_hybrid(
        runtime["pdf_input"],
        experimental=True,
        dpi=dpi,
        layout_model=FullPage(width, height),
        detector=detector,
    )
    assert all(l["method"] == "native" for l in hybrid["blocks"][0]["lines"])
    check(hybrid, runtime["groups"], runtime["kinds"], runtime["expected_native"])
print(
    f"Installed wheel: {len(fixture['cases'])} list continuation contracts at 3 scales, source preservation/concurrency, real OCR and native PDF at 72/137 DPI passed"
)
