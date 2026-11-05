"""UNET 格拓扑的源索引/合并格差分与输入边界。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import recover_table_polygons, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-recover.json").read_text())
for case in fixture["cases"]:
    if "expected_error" in case:
        try:
            recover_table_polygons(
                case["polygons"],
                row_threshold=case["row_threshold"],
                column_threshold=case["column_threshold"],
            )
        except RapidDocError as error:
            assert error.code == case["expected_error"]
        else:
            raise AssertionError("empty/reversed legacy logical span accepted")
        continue
    result = recover_table_polygons(
        case["polygons"],
        row_threshold=case["row_threshold"],
        column_threshold=case["column_threshold"],
    )
    assert result["logic_points"] == case["logic_points"], (case["id"], result, case)
    assert all(
        p[0] <= p[1] < result["rows"] and p[2] <= p[3] < result["columns"]
        for p in result["logic_points"]
    )
case = fixture["cases"][2]
expected = recover_table_polygons(case["polygons"])
with ThreadPoolExecutor(max_workers=4) as pool:
    assert all(
        value == expected
        for value in pool.map(lambda _: recover_table_polygons(case["polygons"]), range(16))
    )
for points, options, code in [
    ([], {}, "invalid_input"),
    ([[[0, 0]] * 4], {}, "invalid_input"),
    (case["polygons"], dict(row_threshold=float("nan")), "invalid_input"),
    (case["polygons"] * 2001, {}, "resource_limit"),
]:
    try:
        recover_table_polygons(points, **options)
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid polygon topology accepted")
assert recover_table_polygons(case["polygons"]) == expected
print(
    f"Installed wheel: {len(fixture['cases'])} polygon topology contracts, concurrency, limits and recovery passed"
)
