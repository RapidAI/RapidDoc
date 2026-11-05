from pathlib import Path
import json, sys, math
from rapid_doc_rust._native import wired_matches_json, render_wired_table_json
from rapid_doc_rust import RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/wired-match.json").read_text())
for case in fixture["cases"]:
    request = dict(polygons=case["polygons"], ocr=case["ocr"], logic_points=[])
    result = json.loads(wired_matches_json(json.dumps(request)))
    assert result["unmatched"] == case["expected"]["unmatched"], case["id"]
    assert result["cell_lines"] == case["expected"]["cell_lines"], (
        case["id"],
        result["cell_lines"],
        case["expected"]["cell_lines"],
    )
    assert len(result["cells"]) == len(case["expected"]["cells"])
    for actual, expected in zip(result["cells"], case["expected"]["cells"]):
        assert len(actual) == len(expected), (case["id"], actual, expected)
        for a, b in zip(actual, expected):
            assert {k: v for k, v in a.items() if k != "points"} == {
                k: v for k, v in b.items() if k != "points"
            }, (case["id"], a, b)
            assert all(
                math.isclose(x, y, abs_tol=1e-8, rel_tol=1e-8)
                for p, q in zip(a["points"], b["points"])
                for x, y in zip(p, q)
            ), (case["id"], a, b)
for case in fixture["html"]:
    actual = render_wired_table_json(case["points"], case["polygons"], case["texts"])
    assert actual == case["expected"], (case["id"], actual, case["expected"])
for request, code in [
    (
        dict(
            polygons=[],
            ocr=[dict(points=[[0, 0], [1, 0], [1, 1], [0, 1]], text="x", confidence=0.9)] * 1001,
            logic_points=[],
        ),
        "resource_limit",
    ),
    (
        dict(
            polygons=[],
            ocr=[dict(points=[[0, 0], [1, 0], [1, 1], [0, 1]], text="x", confidence=2)],
            logic_points=[],
        ),
        "invalid_input",
    ),
]:
    try:
        wired_matches_json(json.dumps(request))
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid wired matcher input accepted")
try:
    render_wired_table_json(
        [[0, 0, 0, 0], [0, 0, 0, 0]], [[[0, 0], [0, 10], [10, 10], [10, 0]]] * 2, [["A"], ["B"]]
    )
except RapidDocError as error:
    assert error.code == "invalid_input"
else:
    raise AssertionError("overlapping logical cells accepted")
print(
    f"Installed wheel: {len(fixture['cases'])} UNET OCR match/split/gather contracts and {len(fixture['html'])} safe HTML/trim contracts passed"
)
