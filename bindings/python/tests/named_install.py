"""命名输入/整型输出、真实版面模型与错误恢复的独立安装测试。"""

from pathlib import Path
import json
import math
import sys
from rapid_doc_rust import OnnxSession, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/named-baseline.json").read_text(encoding="utf-8"))
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"


def expect(code, callback):
    try:
        callback()
    except RapidDocError as error:
        assert error.code == code, (error.code, code)
    else:
        raise AssertionError(f"expected {code}")


for case in fixture["cases"]:
    model = OnnxSession(runtime, root / case["model"], case["sha256"])
    actual = model.run_named(case["inputs"])
    assert actual.keys() == case["outputs"].keys()
    for name, value in actual.items():
        expected = case["outputs"][name]
        assert value["dtype"] == expected["dtype"] and value["shape"] == expected["shape"]
        assert len(value["values"]) == len(expected["values"])
        if value["dtype"] == "float32":
            assert all(
                math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-5)
                for a, b in zip(value["values"], expected["values"])
            ), name
        else:
            assert value["values"] == expected["values"], name
    expect("invalid_input", lambda: model.run_named({}))
    expect("unsupported", lambda: model.run([1], [0.0]))
case = fixture["cases"][0]
for limit in [7, 9]:
    model = OnnxSession(runtime, root / case["model"], case["sha256"], max_elements=limit)
    expect("resource_limit", lambda: model.run_named(case["inputs"]))
model = OnnxSession(runtime, root / case["model"], case["sha256"])
expect(
    "invalid_input",
    lambda: model.run_named(
        dict(left=dict(shape=[2, 2], values=[1.0]), right=case["inputs"]["right"])
    ),
)
expect(
    "invalid_input",
    lambda: model.run_named(
        dict(
            left=dict(shape=[2, 2], values=[1.0, 2.0, 3.0, 4.0], extra=1),
            right=case["inputs"]["right"],
        )
    ),
)
assert model.run_named(case["inputs"])["int64"]["values"] == [9007199254740993, -9007199254740993]
print(
    "Installed wheel: named inputs, FP32/int32/int64 outputs, real layout model, exact large integers, aggregate budgets and recovery pass"
)
