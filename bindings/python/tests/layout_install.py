"""版面框规则与 PNG 推理安装差分，不导入旧解析器。"""

import json
import math
import sys
import subprocess
from pathlib import Path
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import layout_rectangles_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/layout-post.json").read_text(encoding="utf-8"))


def compare(actual, expected, name):
    assert len(actual) == len(expected), (name, actual, expected)
    for a, b in zip(actual, expected):
        assert a.keys() == b.keys(), name
        for key in ["cls_id", "label", "order"]:
            assert a[key] == b[key], (name, key, a, b)
        assert math.isclose(a["score"], b["score"], abs_tol=1e-5, rel_tol=1e-5), (name, a, b)
        assert all(
            math.isclose(x, y, abs_tol=1e-5, rel_tol=1e-5)
            for x, y in zip(a["coordinate"], b["coordinate"])
        ), (name, a, b)


for case in fixture["cases"]:
    actual = json.loads(
        layout_rectangles_json(
            case["rows"], fixture["labels"], case["width"], case["height"], case["threshold"]
        )
    )
    compare(actual, case["expected"], case["id"])
model = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
for case in fixture["png_cases"]:
    compare(model.layout_png(case["input"]), case["expected"], case["input"])
try:
    model.layout_png(fixture["png_cases"][0]["input"], threshold=float("nan"))
except RapidDocError as error:
    assert error.code == "invalid_input"
else:
    raise AssertionError("invalid threshold accepted")
command = [
    sys.executable,
    "-I",
    "-m",
    "rapid_doc_rust",
    "layout-png",
    fixture["png_cases"][0]["input"],
    "--runtime",
    str(root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
    "--model",
    str(root / fixture["model"]),
    "--sha256",
    fixture["sha256"],
]
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
compare(json.loads(result.stdout), fixture["png_cases"][0]["expected"], "CLI")
print(
    "Installed wheel: 15 PP-DocLayoutV3 rectangle rule cases and 3 PNG model layouts match baseline"
)
