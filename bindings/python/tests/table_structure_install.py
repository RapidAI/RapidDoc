"""表格结构安装验证，不把结构 token 等同于完整表格 HTML。"""

from array import array
from pathlib import Path
import hashlib
import json
import math
import sys
import subprocess
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import prepare_table_bgr_json, decode_table_structure_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-structure.json").read_text(encoding="utf-8"))
for case in fixture["preprocess_cases"]:
    try:
        prepared = json.loads(
            prepare_table_bgr_json(bytes(case["bgr"]), case["width"], case["height"])
        )
    except RapidDocError as error:
        assert error.code == case.get("expected_error")
        continue
    assert "expected_error" not in case
    tensor = prepared["tensor"]
    assert tensor["shape"] == case["shape"]
    assert all(
        math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12)
        for a, b in zip(prepared["shape_list"], case["shape_list"])
    )
    values = array("f", tensor["values"])
    if sys.byteorder != "little":
        values.byteswap()
    assert hashlib.sha256(values.tobytes()).hexdigest() == case["fp32_le_sha256"], case["id"]


def compare(actual, expected, name):
    assert actual["tokens"] == expected["tokens"], (name, actual, expected)
    assert math.isclose(actual["confidence"], expected["confidence"], abs_tol=1e-5, rel_tol=1e-5), (
        name,
        actual,
        expected,
    )
    assert len(actual["cell_bboxes"]) == len(expected["cell_bboxes"]), (name, actual, expected)
    for a, b in zip(actual["cell_bboxes"], expected["cell_bboxes"]):
        assert len(a) == len(b)
        assert all(math.isclose(x, y, abs_tol=1e-5, rel_tol=1e-5) for x, y in zip(a, b)), (
            name,
            a,
            b,
        )


for case in fixture["decode_cases"]:
    value = json.loads(
        decode_table_structure_json(
            case["bbox_shape"],
            case["bbox_values"],
            case["prob_shape"],
            case["prob_values"],
            case["dictionary"],
            case["width"],
            case["height"],
            case["variant"] == "slanet-plus",
        )
    )
    compare(value, case, case["variant"])
for case in fixture["models"]:
    model = OnnxSession(
        root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        root / case["model"],
        case["sha256"],
    )
    for png in case["png_cases"]:
        compare(model.table_structure_png(png["input"], variant=case["variant"]), png, png["input"])
    try:
        model.table_structure_png(
            case["png_cases"][0]["input"],
            variant="slanet-1m" if case["variant"] == "slanet-plus" else "slanet-plus",
        )
    except RapidDocError as error:
        assert error.code == "unsupported"
    else:
        raise AssertionError("wrong model variant accepted")
case = fixture["models"][0]
command = [
    sys.executable,
    "-I",
    "-m",
    "rapid_doc_rust",
    "table-structure-png",
    case["png_cases"][0]["input"],
    "--runtime",
    str(root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
    "--model",
    str(root / case["model"]),
    "--sha256",
    case["sha256"],
    "--variant",
    case["variant"],
]
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
compare(json.loads(result.stdout), case["png_cases"][0], "CLI")
print(
    "Installed wheel: 7 table preprocessing cases bit-exact, 2 sequence contracts and 6 SLANet/plus PNG structures match"
)
