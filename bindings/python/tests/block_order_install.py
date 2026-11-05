"""真实 wheel：显式跨块顺序、通栏/多栏、歧义保留、来源和真实页面参考。"""

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
    RapidDocError,
    organize_page_text,
    recognize_page_png,
    recognize_pdf_page_hybrid,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("block-order")
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
        result = organize_page_text(page, experimental=True, block_order="geometry")
        assert page == before
        plan = result["block_order"]
        assert plan["indices"] == case["indices"] and plan["reason"] == case["reason"], (
            case["name"],
            result,
        )
        assert 0 <= plan["work_used"] <= 2_000_000
        assert [p["block_index"] for p in result["paragraphs"]] == case["indices"]
        for p in result["paragraphs"]:
            block = page["blocks"][p["block_index"]]
            assert p["layout_index"] == block["layout_index"]
            assert p["text"] == block["text"]
            assert p["line_indices"] == ([0] if block["kind"] == "text" else [])
        assert result["text"] == "\n\n".join(
            page["blocks"][i]["text"] for i in case["indices"] if page["blocks"][i]["text"]
        )
        source = organize_page_text(page, experimental=True)
        assert "block_order" not in source
        assert source == organize_page_text(page, experimental=True, block_order="source")
        assert [p["block_index"] for p in source["paragraphs"]] == list(range(len(page["blocks"])))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(
            pool.map(
                lambda _: organize_page_text(
                    case["page"], experimental=True, block_order="geometry"
                ),
                range(3),
            )
        )
        assert all(r == results[0] for r in results)

for option in ("unknown", None, False, 1, []):
    try:
        organize_page_text(fixture["cases"][1]["page"], experimental=True, block_order=option)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid block_order accepted")
for limit in (0, -1, True, None, 2_000_001):
    try:
        organize_page_text(
            fixture["cases"][1]["page"], experimental=True, max_block_order_work=limit
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid block-order budget accepted")
try:
    organize_page_text(
        fixture["cases"][2]["page"],
        experimental=True,
        block_order="geometry",
        max_block_order_work=1,
    )
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("reading-order budget became a silent fallback")

runtime = fixture["runtime"]
models = load("wired-pipeline")
for name in ("pdf", "png"):
    assert (
        hashlib.sha256(Path(runtime[f"{name}_input"]).read_bytes()).hexdigest()
        == runtime[f"{name}_sha256"]
    )
ort = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
detector = OnnxSession(ort, root / models["det_model"], models["det_sha256"])
recognizer = OnnxSession(ort, root / models["rec_model"], models["rec_sha256"])


class Layout:
    def __init__(self, dpi):
        self.dpi = dpi

    def layout_png(self, *args, **kwargs):
        return [
            dict(
                label=runtime["regions"][i]["label"],
                cls_id=22,
                score=1.0,
                order=j,
                coordinate=[v * self.dpi / 72 for v in runtime["regions"][i]["bounds"]],
            )
            for j, i in enumerate(runtime["permutation"])
        ]


def check(page, native):
    before = copy.deepcopy(page)
    result = organize_page_text(
        page, experimental=True, block_order="geometry", line_order="geometry"
    )
    assert page == before
    assert result["block_order"]["indices"] == runtime["indices"], result
    assert result["block_order"]["reason"] == "geometry"
    assert [p["block_index"] for p in result["paragraphs"]] == runtime["indices"]
    expected = [r["text"] if native else r["lines"][0]["text"] for r in runtime["regions"]]
    assert [p["text"] for p in result["paragraphs"]] == expected, result
    assert [p["layout_index"] for p in result["paragraphs"]] == runtime["indices"]
    assert result["source_line_count"] == 6


ocr = recognize_page_png(
    runtime["png_input"], layout_model=Layout(72), detector=detector, recognizer=recognizer
)
for b, i in zip(ocr["blocks"], runtime["permutation"]):
    reference = runtime["regions"][i]["lines"][0]
    actual = b["local_lines"][0]
    assert actual["text"] == reference["text"] and actual["points"] == reference["points"]
    assert abs(actual["confidence"] - reference["confidence"]) < 1e-5
check(ocr, False)
for dpi in (72, 137):
    page = recognize_pdf_page_hybrid(
        runtime["pdf_input"],
        experimental=True,
        layout_model=Layout(dpi),
        detector=detector,
        dpi=dpi,
    )
    assert all(l["method"] == "native" for b in page["blocks"] for l in b["lines"])
    check(page, True)
print(
    f"Installed wheel: {len(fixture['cases'])} block-order contracts at 3 scales, source defaults/indices/opaque blocks, full-page fallback/budgets/concurrency and real OCR/native columns at 72/137 DPI passed"
)
