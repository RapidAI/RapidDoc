"""方向分类独立安装差分：张量 FP32 逐位，模型角度相同。"""

from array import array
from pathlib import Path
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import OnnxSession
from rapid_doc_rust._native import prepare_orientation_rgb_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads(
    (root / "build/migration/orientation-baseline.json").read_text(encoding="utf-8")
)
for case in fixture["cases"]:
    tensor = json.loads(
        prepare_orientation_rgb_json(bytes(case["rgb"]), case["width"], case["height"])
    )
    assert tensor["shape"] == case["shape"]
    assert array("f", tensor["values"]).tobytes() == array("f", case["values"]).tobytes(), case[
        "id"
    ]
model = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
for case in fixture["png_cases"]:
    result = model.classify_orientation_png(case["input"])
    assert result["rotation_degrees"] == case["rotation_degrees"], (result, case)
    assert math.isclose(result["score"], case["score"], rel_tol=1e-5, abs_tol=1e-5), (result, case)
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(
        pool.map(model.classify_orientation_png, [case["input"] for case in fixture["png_cases"]])
    )
assert [r["rotation_degrees"] for r in results] == [
    c["rotation_degrees"] for c in fixture["png_cases"]
]
print("Installed wheel: 5 orientation tensors bit-exact, 4 PNG labels/scores and concurrency pass")
