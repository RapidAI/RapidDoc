"""隔离 wheel 中验证补线顺序及每个 FP32 端点位模式。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from array import array
from pathlib import Path
import json, sys, hashlib
from rapid_doc_rust import connect_table_lines, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/table-lines.json").read_text())
for case in fixture["cases"]:
    result = connect_table_lines(case["rows"], case["columns"], **case["config"])
    for key in ("rows", "columns"):
        values = array("f", [v for line in result[key] for v in line])
        if sys.byteorder != "little":
            values.byteswap()
        assert hashlib.sha256(values.tobytes()).hexdigest() == case["expected"][key]["sha256"], (
            case["id"],
            key,
            result[key],
            case["expected"][key]["values"],
        )
for rows, columns, config, code in [
    ([[float("nan"), 0, 1, 1]], [], {}, "invalid_input"),
    ([[0, 0, 1, 1]], [], dict(max_work=0), "resource_limit"),
    ([[0, 0, 1, 1]], [[0, 0, 1, 1]], dict(max_lines=1), "resource_limit"),
    ([], [], dict(unknown=True), "invalid_input"),
]:
    try:
        connect_table_lines(rows, columns, **config)
    except RapidDocError as error:
        assert error.code == code
    else:
        raise AssertionError("invalid line connection accepted")
assert connect_table_lines([], []) == dict(rows=[], columns=[])
print(
    f"Installed wheel: {len(fixture['cases'])} line connection contracts bit-exact; invalid input and budgets passed"
)
