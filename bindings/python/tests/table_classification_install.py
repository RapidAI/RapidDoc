"""隔离 wheel 的 Paddle 表格分类前处理逐位、解码和模型验证。"""

from array import array
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, sys, math, subprocess
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust import _merge_table_classifications
from rapid_doc_rust._native import (
    prepare_table_classifier_bgr_json,
    decode_table_classification_json,
)

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-classification.json").read_text())
for variant, case in [("paddle", c) for c in fixture["cases"]] + [
    ("qanything", c) for c in fixture["qcases"]
]:
    actual = json.loads(
        prepare_table_classifier_bgr_json(
            (root / case["bgr_file"]).read_bytes(), case["width"], case["height"], variant
        )
    )
    assert actual["shape"] == [1, 3, 224, 224]
    expected = array("f")
    expected.frombytes((root / case["tensor_file"]).read_bytes())
    values = array("f", actual["values"])
    if sys.byteorder != "little":
        expected.byteswap()
    failures = [(i, a, b) for i, (a, b) in enumerate(zip(values, expected)) if a != b]
    assert values.tobytes() == expected.tobytes(), (case["id"], len(failures), failures[:8])
for case in fixture["decode"]:
    actual = json.loads(decode_table_classification_json(case["values"], case["output_type"]))
    assert actual["label"] == case["label"] and math.isclose(
        actual["score"], case["score"], abs_tol=1e-6
    )
    if case["output_type"] == "probabilities":
        assert actual["score"] == case["score"]
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
model = OnnxSession(runtime, root / fixture["model"], fixture["sha256"])
qmodel = OnnxSession(runtime, root / fixture["qmodel"], fixture["qsha256"])
for session, variant, case in [(model, "paddle", c) for c in fixture["images"]] + [
    (qmodel, "qanything", c) for c in fixture["qimages"]
]:
    actual = session.classify_table_png(case["input"], variant=variant)
    assert actual["label"] == case["label"] and math.isclose(
        actual["score"], case["score"], abs_tol=1e-5
    ), (case, actual)
    assert session.classify_table_png(Path(case["input"]).read_bytes(), variant=variant) == actual
for case in fixture["consensus"]:
    actual = _merge_table_classifications(case["primary"], case["secondary"])
    assert actual["label"] == case["label"] and actual["score"] == case["score"]
with ThreadPoolExecutor(max_workers=3) as pool:
    case = fixture["images"][0]
    values = list(pool.map(lambda _: model.classify_table_png(case["input"]), range(3)))
    assert all(v == values[0] for v in values)
for values, kind in [
    ([2.0, -1.0], "probabilities"),
    ([0.2, 0.3], "probabilities"),
    ([0.5], "logits"),
    ([float("nan"), 1.0], "logits"),
    ([0.5, 0.5], "auto"),
]:
    try:
        decode_table_classification_json(values, kind)
    except RapidDocError as e:
        assert e.code == "invalid_input"
    else:
        raise AssertionError("invalid classification output accepted")
try:
    prepare_table_classifier_bgr_json(bytes(3 * 256), 1, 256)
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("extreme resized image bypassed budget")
assert (
    model.classify_table_png(fixture["images"][0]["input"])["label"]
    == fixture["images"][0]["label"]
)
result = subprocess.run(
    [
        sys.executable,
        "-I",
        "-m",
        "rapid_doc_rust",
        "classify-table-png",
        fixture["images"][0]["input"],
        "--runtime",
        str(runtime),
        "--model",
        str(root / fixture["model"]),
        "--sha256",
        fixture["sha256"],
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
assert result.returncode == 0, result.stderr
assert json.loads(result.stdout) == model.classify_table_png(fixture["images"][0]["input"])
result = subprocess.run(
    [
        sys.executable,
        "-I",
        "-m",
        "rapid_doc_rust",
        "classify-table-png",
        fixture["images"][0]["input"],
        "--runtime",
        str(runtime),
        "--model",
        str(root / fixture["qmodel"]),
        "--sha256",
        fixture["qsha256"],
        "--variant",
        "qanything",
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
assert result.returncode == 0, result.stderr
assert json.loads(result.stdout) == qmodel.classify_table_png(
    fixture["images"][0]["input"], variant="qanything"
)
try:
    prepare_table_classifier_bgr_json(bytes(3 * 20000), 1, 20000, "qanything")
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("Qanything intermediate image bypassed budget")
print(
    f"Installed wheel: {len(fixture['cases']) + len(fixture['qcases'])} Paddle/Qanything tensors bit-exact, {len(fixture['decode'])} decoders, 6 CPU PNG classifications, 4 consensus cases, CLI and concurrency/errors passed"
)
