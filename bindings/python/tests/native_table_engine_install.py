"""独立wheel：给定原始字符与页面原语的完整引擎，不注入预制文字或候选。"""

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
from rapid_doc_rust import (
    evaluate_native_table_primitives as evaluate,
    recover_native_table_primitives as recover,
    diagnose_native_table_primitives as diagnose,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-table-engine.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
for r in f["real"]:
    assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    same(evaluate(**r), c["expected"]["evaluation"], c["label"] + "/evaluate")
    same(recover(**r), c["expected"]["result"], c["label"] + "/recover")
    same(diagnose(**r), c["expected"]["diagnostics"], c["label"] + "/diagnose")
    same(r, before, "ownership")


for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::45]))
assert (
    len(f["cases"]) == 724 and sum(c["expected"]["result"] is not None for c in f["cases"]) == 547
)
assert sum(c["expected"]["evaluation"]["physical_topology_conflict"] for c in f["cases"]) == 9
assert {c["expected"]["result"]["source"] for c in f["cases"] if c["expected"]["result"]} == {
    "vector_grid",
    "sparse_hybrid",
    "sparse_grid",
    "text_grid",
    "key_value",
    "sparse_multiline",
}
good = next(c["request"] for c in f["cases"] if c["expected"]["result"] is not None)


def invalid(r):
    try:
        evaluate(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid engine input accepted")


r = copy.deepcopy(good)
r["table_input"]["chars"] = [{}] * 20001
invalid(r)
r = copy.deepcopy(good)
r["table_input"]["page_size"] = [float("nan"), 1.0]
invalid(r)
r = copy.deepcopy(good)
r["table_input"]["drawing_lines"] = [{}] * 20001
invalid(r)
# 稳定的大量可见字形在文字选择后的共享工作预算处失败，不能变成普通未接纳。
r = copy.deepcopy(good)
r["table_input"]["chars"] = [
    dict(char="A", char_idx=i, bbox=[1.0, 1.0, 2.0, 2.0]) for i in range(2300)
]
invalid(r)
lines = [
    dict(bbox=[0.0, y, 100.0, y + 0.1], width=0.1, orientation="horizontal")
    for y in (0.0, 30.0, 59.9)
] + [
    dict(bbox=[x, 0.0, x + 0.1, 60.0], width=0.1, orientation="vertical") for x in (0.0, 50.0, 99.9)
]
chars = [
    dict(
        char_idx=i,
        char="A",
        bbox=[
            10.0 + 50.0 * ((i // 250) % 2),
            10.0 + 30.0 * ((i // 250) // 2),
            18.0 + 50.0 * ((i // 250) % 2),
            18.0 + 30.0 * ((i // 250) // 2),
        ],
    )
    for i in range(1000)
]
budget = dict(
    table_bbox=[0.0, 0.0, 100.0, 60.0],
    page_size=[200.0, 200.0],
    angle=0,
    drawing_lines=lines,
    rectangles=[],
    chars=chars,
)
assert recover(budget) is not None
try:
    diagnose(budget)
except RapidDocError as e:
    assert e.code == "resource_limit", e
else:
    raise AssertionError("diagnostic replay reset shared work budget")
assert recover(budget) is not None
assert capabilities()["native_table_engine"] and not capabilities()["legacy_native_table_recovery"]
print(
    "Installed wheel: 724 old raw-input full engine/evaluation/recovery/diagnostic cases, 547 recoveries across all six sources, 9 physical conflicts, 16 actual old PDF/angle raw inputs; no mocked text/candidates, exact types/values/identity-based retention, ownership/concurrency/shared resource guards; Rust PDF primitive extraction and semantic adapter HTML not connected"
)
