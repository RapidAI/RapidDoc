"""检测张量逐 FP32 位模式比较；概率图使用固定数值容差。"""

from array import array
import json
import math
from pathlib import Path
import sys
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import prepare_detection_bgr_json


def main():
    """只验证前处理与模型，不将概率图等同于已完成检测框后处理。"""
    root = Path(sys.argv[1]).resolve()
    fixture = json.loads(
        (root / "build/migration/detection-preprocess.json").read_text(encoding="utf-8")
    )
    session = OnnxSession(
        root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        root / fixture["model"],
        fixture["sha256"],
    )
    count = 0
    for case in fixture["cases"]:
        try:
            actual = json.loads(
                prepare_detection_bgr_json(
                    bytes(case["bgr"]), case["width"], case["height"], case["side"], case["mode"]
                )
            )
        except RapidDocError as error:
            assert error.code == case.get("expected_error"), (case["id"], error.code)
            continue
        assert "expected_error" not in case, case["id"]
        assert actual["shape"] == case["shape"], case["id"]
        assert array("f", actual["values"]).tobytes() == array("f", case["values"]).tobytes(), case[
            "id"
        ]
        if "outputs" in case:
            outputs = session.run(actual["shape"], actual["values"])
            assert outputs.keys() == case["outputs"].keys()
            for name, value in outputs.items():
                expected = case["outputs"][name]
                assert value["shape"] == expected["shape"]
                assert len(value["values"]) == len(expected["values"])
                assert all(
                    math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(value["values"], expected["values"])
                ), case["id"]
            count += 1
    assert count == 2
    print(
        f"Installed wheel: {len(fixture['cases'])} detector preprocessing cases, bit-exact valid tensors, 2 CPU maps match"
    )


if __name__ == "__main__":
    main()
