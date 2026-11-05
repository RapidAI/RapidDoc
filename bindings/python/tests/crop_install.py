"""裁图像素差分，在不带 OpenCV 的 wheel 环境运行。"""

import json
import sys
import math
from pathlib import Path
from rapid_doc_rust import OnnxSession, RapidDocError, ocr_png, ocr_pdf_page
from concurrent.futures import ThreadPoolExecutor
import subprocess
from rapid_doc_rust._native import crop_bgr_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/crop-baseline.json").read_text(encoding="utf-8"))
for case in fixture["cases"]:
    actual = json.loads(
        crop_bgr_json(bytes(case["bgr"]), case["width"], case["height"], case["points"])
    )
    assert (actual["width"], actual["height"]) == (case["crop"]["width"], case["crop"]["height"]), (
        case["id"]
    )
    differences = [abs(a - b) for a, b in zip(actual["bgr"], case["crop"]["bgr"])]
    print(
        case["id"],
        "max channel error",
        max(differences),
        "different",
        sum(d != 0 for d in differences),
        "of",
        len(differences),
    )
    assert max(differences) == 0, case["id"]

session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
db = json.loads((root / "build/migration/db-baseline.json").read_text(encoding="utf-8"))
detector = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / db["model"],
    db["sha256"],
)
for case in fixture["png_cases"]:
    actual = session.recognize_regions_png(case["input"], case["boxes"])
    assert len(actual) == len(case["recognitions"])
    for value, expected in zip(actual, case["recognitions"]):
        assert value["text"] == expected["text"], (value, expected)
        assert math.isclose(value["confidence"], expected["confidence"], abs_tol=1e-5, rel_tol=0), (
            value,
            expected,
        )
    lines = ocr_png(case["input"], detector=detector, recognizer=session)
    assert len(lines) == len(actual)
    assert sorted(line["text"] for line in lines) == sorted(line["text"] for line in actual)
    assert ocr_png(case["input"], detector=detector, recognizer=session, min_confidence=1.0) == []
print("Installed wheel: 4 real region recognitions and 3 PNG OCR pipelines match baseline")

for index, case in enumerate(fixture["png_cases"]):
    page = ocr_pdf_page(fixture["pdf"], detector=detector, recognizer=session, page_index=index)
    assert (page.page_index, page.width, page.height) == (index, 640, 256)
    expected = ocr_png(case["input"], detector=detector, recognizer=session)
    assert page.lines == expected, (index, page.lines, expected)
    assert page.text == "\n".join(v["text"] for v in expected)
with ThreadPoolExecutor(max_workers=3) as pool:
    concurrent = list(
        pool.map(
            lambda p: ocr_png(p["input"], detector=detector, recognizer=session),
            fixture["png_cases"],
        )
    )
assert [len(v) for v in concurrent] == [len(p["boxes"]) for p in fixture["png_cases"]]
for points in ([[[0, 0], [0, 0], [0, 0], [0, 0]]], [[[0, 0], [20, 20], [20, 0], [0, 20]]]):
    try:
        session.recognize_regions_png(fixture["png_cases"][0]["input"], points)
    except RapidDocError as error:
        assert error.code == "invalid_input"
    else:
        raise AssertionError("invalid crop accepted")
command = [
    sys.executable,
    "-I",
    "-m",
    "rapid_doc_rust",
    "ocr-pdf-page",
    fixture["pdf"],
    "--runtime",
    str(root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
    "--det-model",
    str(root / db["model"]),
    "--det-sha256",
    db["sha256"],
    "--rec-model",
    str(root / fixture["model"]),
    "--rec-sha256",
    fixture["sha256"],
]
process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
assert json.loads(process.stdout)["text"] == "RapidDoc 123\n中文识别测试"
print("Installed wheel: 3 scanned PDF pages, concurrent PNG OCR, invalid regions and CLI pass")
