"""UNET 原始 mask 验证，不将类别图宣称为完整网格恢复。"""

from pathlib import Path
from array import array
import json, sys, hashlib
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import prepare_unet_bgr_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/unet-baseline.json").read_text())
failures = []
for case in fixture["cases"]:
    result = json.loads(
        prepare_unet_bgr_json(Path(case["input"]).read_bytes(), case["width"], case["height"])
    )
    assert result["shape"] == case["shape"]
    values = array("f", result["values"])
    if sys.byteorder != "little":
        values.byteswap()
    actual = values.tobytes()
    if hashlib.sha256(actual).hexdigest() != case["fp32_le_sha256"]:
        expected = array("f")
        expected.frombytes(Path(case["expected"]).read_bytes())
        if sys.byteorder != "little":
            expected.byteswap()
        actual_values = array("f")
        actual_values.frombytes(actual)
        if sys.byteorder != "little":
            actual_values.byteswap()
        differences = [(i, a, b) for i, (a, b) in enumerate(zip(actual_values, expected)) if a != b]
        failures.append((case["input"], len(differences), differences[:10]))
assert not failures, failures
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
for case in fixture["png_cases"]:
    result = session.table_line_mask_png(case["input"])
    assert result["width"] == case["width"] and result["height"] == case["height"]
    assert hashlib.sha256(bytes(result["classes"])).hexdigest() == case["sha256"], case["input"]
case = fixture["png_cases"][0]
with ThreadPoolExecutor(max_workers=3) as pool:
    assert all(
        value == case["sha256"]
        for value in pool.map(
            lambda _: hashlib.sha256(
                bytes(session.table_line_mask_png(case["input"])["classes"])
            ).hexdigest(),
            range(3),
        )
    )
small = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
    max_elements=100,
)
for call, code in [
    (lambda: small.table_line_mask_png(case["input"]), "resource_limit"),
    (lambda: session.table_line_mask_png(b"invalid PNG"), "unsupported"),
]:
    try:
        call()
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid UNET request accepted")
assert (
    hashlib.sha256(bytes(session.table_line_mask_png(case["input"])["classes"])).hexdigest()
    == case["sha256"]
)
print(
    f"Installed wheel: {len(fixture['cases'])} UNET preprocessing tensors bit-exact and 3 raw line masks exact"
)
