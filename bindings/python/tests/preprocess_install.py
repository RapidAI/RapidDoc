"""无 NumPy/OpenCV 的安装环境，逐个 FP32 位模式比较前处理并验证 PNG 行识别。"""

from array import array
import json
import math
from pathlib import Path
import sys
from rapid_doc_rust import OnnxSession
from rapid_doc_rust._native import prepare_recognition_bgr_json


def main():
    """噪声、整数缩小、单行、边缘和 padding 均要求前处理 FP32 完全一致。"""
    root = Path(sys.argv[1]).resolve()
    fixture = json.loads(
        (root / "build/migration/recognition-preprocess.json").read_text(encoding="utf-8")
    )
    for case in fixture["tensor_cases"]:
        actual = json.loads(
            prepare_recognition_bgr_json(bytes(case["bgr"]), case["width"], case["height"])
        )
        assert actual["shape"] == case["shape"], case["id"]
        assert array("f", actual["values"]).tobytes() == array("f", case["values"]).tobytes(), case[
            "id"
        ]
    session = OnnxSession(
        root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        root / fixture["model"],
        fixture["sha256"],
    )
    for case in fixture["png_cases"]:
        result = session.recognize_png(case["input"])
        assert len(result) == 1
        assert result[0]["text"] == case["text"], (result, case)
        assert math.isclose(result[0]["confidence"], case["confidence"], rel_tol=0, abs_tol=1e-5), (
            case["input"]
        )
    print(
        f"Installed wheel: {len(fixture['tensor_cases'])} preprocessing tensors bit-exact; {len(fixture['png_cases'])} PNG line recognitions match Python"
    )


if __name__ == "__main__":
    main()
