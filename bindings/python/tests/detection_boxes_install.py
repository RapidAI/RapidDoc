"""独立 wheel 的 DB 框和评分差分；不依赖 NumPy/OpenCV。"""

import json
import sys
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import detection_boxes_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/db-baseline.json").read_text(encoding="utf-8"))
failures = []
for case in fixture["cases"]:
    actual = json.loads(
        detection_boxes_json(
            case["values"],
            case["width"],
            case["height"],
            case["width"],
            case["height"],
            case["dilation"],
        )
    )
    if len(actual) != len(case["boxes"]):
        failures.append((case["id"], "count", len(actual), len(case["boxes"])))
        continue
    for index, (value, box, score) in enumerate(zip(actual, case["boxes"], case["scores"])):
        if value["points"] != box or abs(value["score"] - score) > 1e-5:
            failures.append((case["id"], index, value, dict(points=box, score=score)))
assert not failures, json.dumps(failures, ensure_ascii=False)
print(f"Installed wheel: {len(fixture['cases'])} DB boxes exact and scores within 1e-5")

session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
for case in fixture["png_cases"]:
    path = Path(case["input"])
    for source in (path, path.read_bytes()):
        actual = session.detect_png(source)
        assert [v["points"] for v in actual] == case["boxes"], (path, actual, case)
        assert all(
            math.isclose(v["score"], score, abs_tol=1e-5, rel_tol=1e-5)
            for v, score in zip(actual, case["scores"])
        )
with ThreadPoolExecutor(max_workers=3) as pool:
    outputs = list(pool.map(session.detect_png, [v["input"] for v in fixture["png_cases"]]))
assert [len(v) for v in outputs] == [len(v["boxes"]) for v in fixture["png_cases"]]
try:
    session.detect_png(b"broken")
except RapidDocError as error:
    assert error.code == "unsupported"
else:
    raise AssertionError("corrupt PNG accepted")
assert session.detect_png(fixture["png_cases"][-1]["input"]) == []
print("Installed wheel: 3 PNG detection cases, paths/bytes/concurrency/error recovery pass")
