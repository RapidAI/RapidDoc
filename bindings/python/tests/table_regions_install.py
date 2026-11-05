from pathlib import Path
import json, sys, hashlib, math
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._native import (
    rotate_table_image_bytes,
    table_cells_from_lines_json,
    table_rotation_angle,
)

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-regions.json").read_text())
for case in fixture["angles"]:
    result = table_rotation_angle(Path(case["input"]).read_bytes(), case["width"], case["height"])
    assert math.isclose(result, case["angle"], abs_tol=1e-5), (case["id"], result, case["angle"])
for case in fixture["rotations"]:
    result = rotate_table_image_bytes(
        Path(case["input"]).read_bytes(), case["width"], case["height"], case["angle"]
    )
    if hashlib.sha256(result).hexdigest() != case["sha256"]:
        expected = Path(case["expected"]).read_bytes()
        diff = [
            (i % case["width"], i // case["width"], a, b)
            for i, (a, b) in enumerate(zip(result, expected))
            if a != b
        ]
        raise AssertionError((case["input"], case["angle"], len(diff), diff[:12]))


def compare(result, case, topology=True):
    assert math.isclose(result["rotation_degrees"], case["rotation_degrees"], abs_tol=1e-5), case
    for key in ["polygons", "rotated_polygons"]:
        assert len(result[key]) == len(case[key]), (
            case.get("id", case.get("input")),
            key,
            result,
            case,
        )
        for a, b in zip(result[key], case[key]):
            for x, y in zip(a, b):
                assert all(math.isclose(u, v, abs_tol=1e-5, rel_tol=1e-6) for u, v in zip(x, y)), (
                    key,
                    a,
                    b,
                )
    if topology:
        assert (result["topology"]["logic_points"] if result["topology"] else None) == case[
            "logic_points"
        ], (result, case)


for case in fixture["cases"]:
    raw = json.loads(
        table_cells_from_lines_json(
            case["lines"], case["width"], case["height"], case["rotated_fix"], False
        )
    )
    compare(raw, case, False)
    try:
        result = json.loads(
            table_cells_from_lines_json(
                case["lines"], case["width"], case["height"], case["rotated_fix"]
            )
        )
    except RapidDocError as error:
        assert error.code == case.get("expected_error"), (case["id"], str(error))
        continue
    assert "expected_error" not in case
    compare(result, case)
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
)
for case in fixture["png_cases"]:
    compare(session.table_cells_png(case["input"]), case)
case = fixture["png_cases"][0]
expected = session.table_cells_png(case["input"])
with ThreadPoolExecutor(max_workers=3) as pool:
    assert all(
        value == expected
        for value in pool.map(lambda _: session.table_cells_png(case["input"]), range(3))
    )
for data, code in [(b"invalid PNG", "unsupported")]:
    try:
        session.table_cells_png(data)
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid UNET cells input accepted")
assert session.table_cells_png(case["input"]) == expected
print(
    f"Installed wheel: {len(fixture['angles'])} angles, {len(fixture['rotations'])} rotation rasters pixel-exact, {len(fixture['cases'])} region contracts and {len(fixture['png_cases'])} UNET cell pipelines matched"
)
