"""对已有表格分类和方向模型执行 Python/Rust CPU 张量差分。"""

import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
from common import ROOT, digest, write_json


def main():
    """固定张量而不测试前处理；模型缺失、输出变更或超容差均失败。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/onnx-probe-report.json")
    args = parser.parse_args()
    models = ["rapid_doc/model/table/rapid_table_self/models/q_cls.onnx",
              "rapid_doc/model/table/rapid_table_self/models/paddle_cls.onnx",
              "rapid_doc/resources/rapid_orientation.onnx"]
    output = []
    for relative in models:
        model = ROOT / relative
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        for batch in (1, 2):
            shape = (batch, 3, 224, 224)
            tensor = np.random.default_rng(42).uniform(-1, 1, shape).astype(np.float32)
            path = ROOT / "build/migration/ort-input.json"
            write_json(path, {"shape": shape, "values": tensor.ravel().tolist()})
            python_outputs = session.run(None, {session.get_inputs()[0].name: tensor})
            result = subprocess.run([str(args.binary.resolve()), str(args.runtime.resolve()), str(model), str(path), digest(model.read_bytes())],
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=120)
            actual = json.loads(result.stdout)
            names = [value.name for value in session.get_outputs()]
            assert set(actual) == set(names), relative
            errors = {}
            for name, expected in zip(names, python_outputs):
                value = actual[name]
                assert value["shape"] == list(expected.shape), (relative, name)
                array = np.asarray(value["values"], dtype=np.float32).reshape(expected.shape)
                np.testing.assert_allclose(array, expected, atol=1e-5, rtol=1e-5, err_msg=relative)
                errors[name] = float(np.max(np.abs(array - expected)))
            write_json(ROOT / f"build/migration/onnx-fixtures/{model.stem}-batch{batch}.json", {
                "model": relative, "sha256": digest(model.read_bytes()), "shape": shape,
                "values": tensor.ravel().tolist(), "outputs": {name: {"shape": list(value.shape), "values": value.ravel().tolist()} for name, value in zip(names, python_outputs)}})
            output.append({"model": relative, "sha256": digest(model.read_bytes()), "shape": shape,
                "input_sha256": digest(tensor.tobytes()), "max_abs_error": errors,
                "python_providers": session.get_providers()})
    write_json(args.output, {"runtime_sha256": digest(args.runtime.read_bytes()), "python_ort": ort.__version__,
        "scope": "CPU single float input/output; synthetic tensors, not preprocessing or document quality", "cases": output})
    print(f"ONNX CPU differential: {len(output)} model/batch cases passed")


if __name__ == "__main__":
    main()
