"""已安装 wheel：旧文本连接差分、段落几何合同和真实页面结果接入。"""

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
    merge_text_lines,
    organize_page_text,
    recognize_page_png,
    recognize_pdf_page_hybrid,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text())
fixture = load("paragraphs")
for case in fixture["join_cases"]:
    actual = merge_text_lines(case["lines"])
    assert actual == case["expected"], (case["name"], case["lines"], actual, case["expected"])
for case in fixture["page_cases"]:
    page = case["page"]
    before = copy.deepcopy(page)
    result = organize_page_text(page, experimental=True, split_columns=case["split_columns"])
    assert page == before
    assert [p["line_indices"] for p in result["paragraphs"]] == case["groups"], (
        case["name"],
        result,
    )
    assert [p["kind"] for p in result["paragraphs"]] == case["kinds"]
    assert [p["text"] for p in result["paragraphs"]] == case["texts"]
    assert result["text"] == "\n\n".join(text for text in case["texts"] if text)
    assert result["source_line_count"] == sum(
        len(b["lines"]) for b in page["blocks"] if b["kind"] == "text"
    )
    for index, block in enumerate(page["blocks"]):
        groups = [p for p in result["paragraphs"] if p["block_index"] == index]
        if block["kind"] == "text":
            assert sorted(i for p in groups for i in p["line_indices"]) == list(
                range(len(block["lines"]))
            )
        for para in groups:
            assert para["layout_index"] == block["layout_index"]
            if para["line_indices"]:
                points = [
                    point for i in para["line_indices"] for point in block["lines"][i]["points"]
                ]
                assert para["bbox"] == [
                    min(p[0] for p in points),
                    min(p[1] for p in points),
                    max(p[0] for p in points),
                    max(p[1] for p in points),
                ]
with ThreadPoolExecutor(max_workers=3) as pool:
    case = fixture["page_cases"][0]
    outputs = list(
        pool.map(lambda _: organize_page_text(case["page"], experimental=True), range(6))
    )
    assert all(value == outputs[0] for value in outputs)

for lines in [["x"] * 1001, ["x" * 1_000_001], ["x" * 1000] * 1000]:
    try:
        merge_text_lines(lines)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("line joining budget ignored")
source = copy.deepcopy(fixture["page_cases"][0]["page"])
source["blocks"][0]["lines"][0]["points"] = [[0, 0], [20, 20], [20, 0], [0, 20]]
try:
    organize_page_text(source, experimental=True)
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("self-intersecting line accepted")
source = copy.deepcopy(fixture["page_cases"][0]["page"])
source["blocks"].append(copy.deepcopy(source["blocks"][0]))
try:
    organize_page_text(source, experimental=True)
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("duplicate source block accepted")
source = copy.deepcopy(fixture["page_cases"][0]["page"])
source["coordinate_space"] = "visual_points"
try:
    organize_page_text(source, experimental=True)
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("mixed point/pixel coordinate space accepted")
empty = dict(width=600, height=500, coordinate_space="page_pixels", blocks=[])
assert organize_page_text(empty, experimental=True)["paragraphs"] == []
missing = copy.deepcopy(fixture["page_cases"][0]["page"])
missing["blocks"][0]["lines"] = []
missing["blocks"][0]["text"] = "preserved text"
kept = organize_page_text(missing, experimental=True)
assert kept["text"] == "preserved text" and kept["diagnostics"][0]["code"] == "empty_text_block"

runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
models = load("wired-pipeline")
native = load("native-fusion")
tables = load("native-tables")
detector = OnnxSession(runtime, root / models["det_model"], models["det_sha256"])
recognizer = OnnxSession(runtime, root / models["rec_model"], models["rec_sha256"])


class FixedLayout:
    def __init__(self, regions):
        self.regions = regions

    def layout_png(self, *args, **kwargs):
        return self.regions


case = native["cases"][0]
layout = FixedLayout(
    [
        dict(
            cls_id=22,
            label="text",
            score=1.0,
            coordinate=[0, 0, case["width"], case["height"]],
            order=1,
        )
    ]
)
ocr = recognize_page_png(
    case["input"], layout_model=layout, detector=detector, recognizer=recognizer
)
expected = next(
    c["expected"] for c in fixture["runtime_joins"] if c["page_index"] == 0 and c["dpi"] == 72
)
assert organize_page_text(ocr, experimental=True)["text"] == expected
hybrid = recognize_pdf_page_hybrid(
    native["pdf_input"], experimental=True, layout_model=layout, detector=detector, dpi=72
)
assert all(line["confidence"] is None for line in hybrid["blocks"][0]["lines"])
assert organize_page_text(hybrid, experimental=True)["text"] == expected
table_case = tables["cases"][0]
table = recognize_pdf_page_hybrid(
    tables["pdf_input"],
    experimental=True,
    prefer_native_tables=True,
    layout_model=FixedLayout(table_case["layout"]),
    dpi=72,
)
paragraphs = organize_page_text(table, experimental=True)
assert paragraphs["paragraphs"][0]["kind"] == "table" and paragraphs["text"] == table["text"]
assert paragraphs["source_line_count"] == 0
assert capabilities()["text_line_joining"] and capabilities()["page_paragraphs"]
print(
    f"Installed wheel: {len(fixture['join_cases'])} legacy line joins, 7 paragraph geometry contracts, column/source indices, invalid input/budgets/concurrency and real OCR/native/table pages passed"
)
