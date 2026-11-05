"""真实 wheel：可选栏内行序、歧义保留、原索引/预算和打乱真实页面后的恢复。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy, json, sys
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    organize_page_text,
    recognize_page_png,
    recognize_pdf_page_hybrid,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("line-order")
for case in fixture["cases"]:
    for scale in (0.5, 1.0, 2.0):
        page = copy.deepcopy(case["page"])
        page["width"] = int(page["width"] * scale)
        page["height"] = int(page["height"] * scale)
        for block in page["blocks"]:
            block["region"]["coordinate"] = [v * scale for v in block["region"]["coordinate"]]
            for line in block["lines"]:
                line["points"] = [[x * scale, y * scale] for x, y in line["points"]]
        before = copy.deepcopy(page)
        result = organize_page_text(
            page, experimental=True, split_columns=case["split_columns"], line_order="geometry"
        )
        assert page == before
        assert [p["line_indices"] for p in result["paragraphs"]] == case["groups"], (
            case["name"],
            result,
        )
        assert [p["kind"] for p in result["paragraphs"]] == case["kinds"]
        assert [p["text"] for p in result["paragraphs"]] == case["texts"]
        assert sorted(d["code"] for d in result["diagnostics"]) == case["diagnostics"], (
            case["name"],
            result,
        )
        assert sorted(i for p in result["paragraphs"] for i in p["line_indices"]) == list(
            range(len(page["blocks"][0]["lines"]))
        )
        assert organize_page_text(page, experimental=True) == organize_page_text(
            page, experimental=True, line_order="source"
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(
            pool.map(
                lambda _: organize_page_text(
                    case["page"], experimental=True, line_order="geometry"
                ),
                range(3),
            )
        )
        assert all(o == outputs[0] for o in outputs)

for value in ("unknown", False, None, 1, []):
    try:
        organize_page_text(fixture["cases"][0]["page"], experimental=True, line_order=value)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid line_order accepted")

# 重排后的累计前缀工作量也必须受全页限制；来源顺序本身未超限。
page = copy.deepcopy(fixture["cases"][0]["page"])
page["blocks"] = []
for i in range(17):
    b = copy.deepcopy(fixture["cases"][0]["page"]["blocks"][0])
    b["layout_index"] = i
    b["lines"] = []
    for j in range(100):
        y = (99 - j) * 3
        b["lines"].append(
            dict(
                points=[[10, y], [240, y], [240, y + 2], [10, y + 2]],
                text="x" * 20000 if j == 99 else "x",
                confidence=None,
            )
        )
    page["blocks"].append(b)
organize_page_text(page, experimental=True)
try:
    organize_page_text(page, experimental=True, line_order="geometry")
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("reordered aggregate work budget ignored")

# 两块位置相反也不重排布局块，不跨块连接文字。
page = copy.deepcopy(fixture["cases"][0]["page"])
other = copy.deepcopy(page["blocks"][0])
other["layout_index"] = 8
page["blocks"].append(other)
result = organize_page_text(page, experimental=True, line_order="geometry")
assert [p["block_index"] for p in result["paragraphs"]] == [0, 1]
assert [p["layout_index"] for p in result["paragraphs"]] == [7, 8]

runtime = load(fixture["runtime_fixture"])["runtime"]
models = load("wired-pipeline")
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


def shuffled_check(page, texts):
    before = copy.deepcopy(page)
    permutation = fixture["permutation"]
    lines = page["blocks"][0]["lines"]
    assert len(lines) == 6
    page["blocks"][0]["lines"] = [lines[i] for i in permutation]
    shuffled = copy.deepcopy(page)
    result = organize_page_text(page, experimental=True, line_order="geometry")
    assert page == shuffled
    assert [p["line_indices"] for p in result["paragraphs"]] == [[3, 2], [5, 0], [4, 1]], result
    assert [p["text"] for p in result["paragraphs"]] == texts
    assert result["text"] == organize_page_text(before, experimental=True)["text"]
    assert sorted(d["code"] for d in result["diagnostics"]) == ["lines_reordered"]


ocr = recognize_page_png(
    runtime["png_input"],
    layout_model=FullPage(runtime["width"], runtime["height"]),
    detector=detector,
    recognizer=recognizer,
)
assert [l["text"] for l in ocr["blocks"][0]["lines"]] == [l["text"] for l in runtime["ocr"]]
shuffled_check(ocr, runtime["expected_ocr"])
for dpi in (72, 137):
    page = recognize_pdf_page_hybrid(
        runtime["pdf_input"],
        experimental=True,
        dpi=dpi,
        layout_model=FullPage(
            round(runtime["width"] * dpi / 72), round(runtime["height"] * dpi / 72)
        ),
        detector=detector,
    )
    assert all(l["method"] == "native" for l in page["blocks"][0]["lines"])
    shuffled_check(page, runtime["expected_native"])
print(
    f"Installed wheel: {len(fixture['cases'])} line-order contracts at 3 scales, source defaults/indices, ambiguity diagnostics/concurrency/aggregate budgets and shuffled real OCR/native pages passed"
)
