"""在无 NumPy/Python ORT 的干净环境检验 wheel 中的 Rust ONNX 会话。"""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
from pathlib import Path
import sys

from rapid_doc_rust import OnnxSession, RapidDocError


def expect(code, operation):
    """检查稳定错误码。"""
    try:
        operation()
    except RapidDocError as error:
        assert error.code == code, (error.code, code)
    else:
        raise AssertionError(f"Expected {code}")


def main():
    """比较预先导出的 Python 基线，并验证复用、并发、错误后恢复。"""
    assert importlib.util.find_spec("numpy") is None
    assert importlib.util.find_spec("onnxruntime") is None
    root = Path(sys.argv[1]).resolve()
    runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
    files = sorted((root / "build/migration/onnx-fixtures").glob("*.json"))
    assert len(files) == 6
    sessions = {}
    for path in files:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        model = root / fixture["model"]
        if fixture["model"] not in sessions:
            sessions[fixture["model"]] = OnnxSession(runtime, model, fixture["sha256"])
        session = sessions[fixture["model"]]
        actual = session.run(fixture["shape"], fixture["values"])
        assert actual.keys() == fixture["outputs"].keys()
        for name, expected in fixture["outputs"].items():
            assert actual[name]["shape"] == expected["shape"]
            assert len(actual[name]["values"]) == len(expected["values"])
            assert all(
                math.isclose(x, y, abs_tol=1e-5, rel_tol=1e-5)
                for x, y in zip(actual[name]["values"], expected["values"])
            )
    expect("invalid_input", lambda: session.run([1, 2], [1.0]))
    expect("invalid_input", lambda: session.run([1], [float("nan")]))
    expect("resource_limit", lambda: session.run([16_000_001], []))
    expect("invalid_input", lambda: OnnxSession(runtime, model, "0" * 64))
    expect("model_missing", lambda: OnnxSession(runtime, root / "missing-model.onnx", "0" * 64))
    expect(
        "device_unavailable", lambda: OnnxSession(runtime, model, fixture["sha256"], device="cuda")
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: session.run(fixture["shape"], fixture["values"]), range(4))
        )
    assert all(result == actual for result in results)
    print(
        "Installed Rust ONNX session: 6 baselines, reuse, concurrency, hash/shape errors and explicit CUDA rejection pass"
    )


if __name__ == "__main__":
    main()
