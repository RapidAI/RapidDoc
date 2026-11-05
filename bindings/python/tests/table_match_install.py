"""在无 NumPy/OpenCV/ORT Python 包的隔离环境验证表格 OCR 匹配。"""

import json
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from rapid_doc_rust import OnnxSession, RapidDocError, match_table, recognize_table_png
from rapid_doc_rust._native import match_cells_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-match.json").read_text(encoding="utf-8"))
for case in fixture["cases"]:
    actual = json.loads(match_cells_json(case["cell_bboxes"], case["ocr_boxes"]))
    assert actual == case["expected"], (case["id"], actual, case["expected"])


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.texts = []
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        if tag == "td":
            self.cells.append(dict(attrs))
            self.texts.append("")
        if tag == "br":
            self.texts[-1] += "\n"

    def handle_data(self, data):
        if self.texts and self.tags[-1] in ("td", "br"):
            self.texts[-1] += data

    def handle_endtag(self, tag):
        self.tags.append("/" + tag)


runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
det = json.loads((root / "build/migration/db-baseline.json").read_text(encoding="utf-8"))
rec = json.loads((root / "build/migration/crop-baseline.json").read_text(encoding="utf-8"))
detector = OnnxSession(runtime, root / det["model"], det["sha256"])
recognizer = OnnxSession(runtime, root / rec["model"], rec["sha256"])
models = {}
for case in fixture["png_cases"]:
    variant = case["variant"]
    if variant not in models:
        models[variant] = OnnxSession(runtime, root / case["model"], case["sha256"])
    if "expected_error" in case:
        for call in (
            lambda: match_table(case["structure"], case["ocr_lines"]),
            lambda: recognize_table_png(
                case["input"],
                structure_model=models[variant],
                detector=detector,
                recognizer=recognizer,
                variant=variant,
            ),
        ):
            try:
                call()
            except RapidDocError as error:
                assert error.code == case["expected_error"]
            else:
                raise AssertionError("malformed blank model structure accepted")
        continue
    result = match_table(case["structure"], case["ocr_lines"])
    assert result["matches"] == case["expected"], case
    expected_texts = [
        " ".join(case["ocr_lines"][i]["text"] for i in indices)
        for indices in case["expected"]["cells"]
    ]
    assert result["cell_texts"] == expected_texts
    parser = TableParser()
    parser.feed(result["html"])
    assert parser.texts == expected_texts, (case["input"], parser.texts, expected_texts)
    end_to_end = recognize_table_png(
        case["input"],
        structure_model=models[variant],
        detector=detector,
        recognizer=recognizer,
        variant=variant,
    )
    assert end_to_end["matches"] == case["expected"], case["input"]
    assert end_to_end["cell_texts"] == expected_texts, (
        case["input"],
        end_to_end["cell_texts"],
        expected_texts,
    )
    assert end_to_end["html"] == result["html"]

structure = dict(
    tokens=[
        "<html>",
        "<body>",
        "<table>",
        "<tr>",
        "<td></td>",
        "</tr>",
        "</table>",
        "</body>",
        "</html>",
    ],
    cell_bboxes=[[0, 0, 100, 100]],
)
lines = [
    dict(points=[[1, 1], [90, 1], [90, 10], [1, 10]], text='<script>alert("x")</script>& 中文  ')
]
expected = match_table(structure, lines)
assert "<script>" not in expected["html"] and "&lt;script&gt;" in expected["html"]
assert expected["cell_texts"] == [lines[0]["text"]]
with ThreadPoolExecutor(max_workers=4) as pool:
    assert all(
        value == expected for value in pool.map(lambda _: match_table(structure, lines), range(16))
    )
for changed in [
    dict(structure, tokens=structure["tokens"][:3] + ["<script>"] + structure["tokens"][3:]),
    dict(structure, cell_bboxes=[]),
    dict(structure, cell_bboxes=[[5, 0, 1, 10]]),
]:
    try:
        match_table(changed, lines)
    except RapidDocError as error:
        assert error.code == "invalid_input"
    else:
        raise AssertionError("malformed structure accepted")
assert match_table(structure, lines) == expected
case = fixture["png_cases"][0]
command = [
    sys.executable,
    "-I",
    "-m",
    "rapid_doc_rust",
    "recognize-table-png",
    case["input"],
    "--runtime",
    str(runtime),
    "--structure-model",
    str(root / case["model"]),
    "--structure-sha256",
    case["sha256"],
    "--variant",
    case["variant"],
    "--det-model",
    str(root / det["model"]),
    "--det-sha256",
    det["sha256"],
    "--rec-model",
    str(root / rec["model"]),
    "--rec-sha256",
    rec["sha256"],
]
result = json.loads(
    subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True).stdout
)
assert result["matches"] == case["expected"]
print(
    "Installed wheel: 8 matcher contracts, 4 real SLANet→OCR→HTML pipelines, 2 malformed blank model rejections, escaping and concurrency passed"
)
