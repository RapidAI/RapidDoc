"""独立wheel旧引擎输入早期门与已有诊断守卫；不是完整候选引擎测试。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import inspect_native_table_engine_policy, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-table-engine-policy-stable.json").read_text(encoding="utf-8")
)
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def run(c):
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    result = inspect_native_table_engine_policy(**r)
    assert result == c["expected"], (r, result, c["expected"])
    assert (
        type(result["primitive_count"]) is int
        and type(result["minimum_columns"]) is int
        and type(result["column_consistent"]) is list
    )
    for key in [
        "alias_affected_physical_blank_row",
        "physical_row_undercount",
        "attempted_line_rect_grid_conflict",
    ]:
        assert type(result[key]) is bool
    assert r == before


for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::20]))


def fails(r):
    try:
        inspect_native_table_engine_policy(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid engine-policy input accepted")


r = copy.deepcopy(f["cases"][0]["request"])
r["vector_attempts"] = [{}] * 1001
fails(r)
r = copy.deepcopy(f["cases"][0]["request"])
r["table_input"]["drawing_lines"] = [dict(bbox=[0.0, 1.0, 2.0, 3.0])] * 20001
fails(r)
r = copy.deepcopy(f["cases"][0]["request"])
r["vector_attempts"] = [
    dict(
        evidence="line_grid",
        first_rejection_gate="empty_row",
        empty_rows=None,
        alias_affected_rows=[],
    )
]
fails(r)
assert (
    capabilities()["native_table_engine_policy"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(f['cases'])} old geometry/overlap count/alias blank/row undercount/column floor/attempted conflict/rect rescue guards; ownership/concurrency/budgets, no complete evaluator"
)
