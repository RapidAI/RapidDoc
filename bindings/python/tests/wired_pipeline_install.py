from pathlib import Path
import json, math, sys, hashlib, subprocess, struct
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import OnnxSession, recognize_wired_table_png, match_wired_table, RapidDocError
from rapid_doc_rust._native import blank_cell_crop_json

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/wired-pipeline.json").read_text())
for case in fixture["crop_cases"]:
    actual = json.loads(
        blank_cell_crop_json(bytes(case["bgr"]), case["width"], case["height"], case["points"])
    )
    expected = case["expected"]
    assert (actual is None) == (expected is None), case["id"]
    if actual:
        assert (actual["width"], actual["height"]) == (expected["width"], expected["height"])
        assert hashlib.sha256(bytes(actual["bgr"])).hexdigest() == expected["sha256"]
        assert actual["contrast"] == expected["contrast"], (actual, expected)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
structure = OnnxSession(runtime, root / fixture["unet_model"], fixture["unet_sha256"])
detector = OnnxSession(runtime, root / fixture["det_model"], fixture["det_sha256"])
recognizer = OnnxSession(runtime, root / fixture["rec_model"], fixture["rec_sha256"])


def blank_compare(actual, expected):
    assert len(actual) == len(expected), (actual, expected)
    for a, b in zip(actual, expected):
        assert {k: v for k, v in a.items() if k != "confidence"} == {
            k: v for k, v in b.items() if k != "confidence"
        }, (a, b)
        assert math.isclose(a["confidence"], b["confidence"], abs_tol=1e-5, rel_tol=1e-5), (a, b)


for case in fixture["cases"]:
    blank_compare(
        recognizer.recognize_blank_table_cells_png(case["input"], case["polygons"]),
        case["all_blank_rec"],
    )
    actual = recognize_wired_table_png(
        case["input"], structure_model=structure, detector=detector, recognizer=recognizer
    )
    assert actual["matches"]["cell_lines"] == case["cell_lines"], (
        case["input"],
        actual["matches"]["cell_lines"],
        case["cell_lines"],
    )
    assert actual["matches"]["unmatched"] == case["match"]["unmatched"]
    assert actual["html"] == case["html"], (case["input"], actual["html"], case["html"])
    blank_compare(actual["blank_rec"], case["blank_rec"])
    disabled = recognize_wired_table_png(
        case["input"],
        structure_model=structure,
        detector=detector,
        recognizer=recognizer,
        blank_cell_rec=False,
    )
    assert (
        disabled["blank_rec"] == []
        and disabled["matches"]["cell_lines"] == case["match"]["cell_lines"]
    )
case = fixture["cases"][0]
expected = recognize_wired_table_png(
    case["input"], structure_model=structure, detector=detector, recognizer=recognizer
)
with ThreadPoolExecutor(max_workers=3) as pool:
    assert all(
        v == expected
        for v in pool.map(
            lambda _: recognize_wired_table_png(
                case["input"], structure_model=structure, detector=detector, recognizer=recognizer
            ),
            range(3),
        )
    )
try:
    recognizer.recognize_blank_table_cells_png(case["input"], case["polygons"], needed=[])
except RapidDocError as error:
    assert error.code == "invalid_input"
else:
    raise AssertionError("mismatched blank flags accepted")
blank = next(Path(v["input"]) for v in fixture["cases"] if Path(v["input"]).name == "blank.png")
w, h = struct.unpack(">II", blank.read_bytes()[16:24])
try:
    recognizer.recognize_blank_table_cells_png(blank, [[[0, 0], [0, h], [w, h], [w, 0]]] * 2000)
except RapidDocError as error:
    assert error.code == "resource_limit"
else:
    raise AssertionError("low-contrast crops bypassed cumulative work budget")
assert (
    recognizer.recognize_blank_table_cells_png(blank, [[[0, 0], [0, h], [w, h], [w, 0]]])[0][
        "reason"
    ]
    == "low_contrast"
)
command = [
    sys.executable,
    "-I",
    "-m",
    "rapid_doc_rust",
    "recognize-wired-table-png",
    case["input"],
    "--runtime",
    str(runtime),
    "--structure-model",
    str(root / fixture["unet_model"]),
    "--structure-sha256",
    fixture["unet_sha256"],
    "--det-model",
    str(root / fixture["det_model"]),
    "--det-sha256",
    fixture["det_sha256"],
    "--rec-model",
    str(root / fixture["rec_model"]),
    "--rec-sha256",
    fixture["rec_sha256"],
]
result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
assert result.returncode == 0, result.stderr
assert json.loads(result.stdout) == expected
result = subprocess.run(
    command + ["--no-blank-cell-rec", "--no-rotated-fix"],
    capture_output=True,
    text=True,
    encoding="utf-8",
)
assert result.returncode == 0, result.stderr
disabled = recognize_wired_table_png(
    case["input"],
    structure_model=structure,
    detector=detector,
    recognizer=recognizer,
    blank_cell_rec=False,
    rotated_fix=False,
)
assert json.loads(result.stdout) == disabled
print(
    f"Installed wheel: {len(fixture['crop_cases'])} color crop/contrast contracts, {len(fixture['cases'])} UNET→OCR→blank-rec→HTML pipelines, disabling, concurrency and CLI passed"
)
