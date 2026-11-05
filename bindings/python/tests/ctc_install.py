"""真实安装 wheel 的 OCR 识别张量/CTC 差分，无 NumPy 或 Python ORT。"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sys
from rapid_doc_rust import OnnxSession


def main():
    """文字、token、时间步精确一致；五位小数置信度使用固定 1e-5 容差。"""
    root = Path(sys.argv[1]).resolve()
    fixture = json.loads((root / "build/migration/ctc-fixtures.json").read_text(encoding="utf-8"))
    session = OnnxSession(
        root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
        root / fixture["model"],
        fixture["sha256"],
    )
    for case in fixture["cases"]:
        actual = session.recognize_ctc(case["shape"], case["values"])
        assert len(actual) == len(case["expected"]), case["id"]
        for value, expected in zip(actual, case["expected"]):
            assert value["text"] == expected["text"], (case["id"], value, expected)
            assert value["token_indices"] == expected["token_indices"], case["id"]
            assert value["timesteps"] == expected["timesteps"], case["id"]
            assert math.isclose(
                value["confidence"], expected["confidence"], abs_tol=1e-5, rel_tol=0
            ), case["id"]
    case = fixture["cases"][-1]
    with ThreadPoolExecutor(max_workers=2) as pool:
        repeated = list(
            pool.map(lambda _: session.recognize_ctc(case["shape"], case["values"]), range(4))
        )
    assert all(value == actual for value in repeated)
    print(
        "Installed wheel: 5 OCR/CTC fixtures, Chinese/English/blank/batch, tokens and concurrency pass"
    )


if __name__ == "__main__":
    main()
