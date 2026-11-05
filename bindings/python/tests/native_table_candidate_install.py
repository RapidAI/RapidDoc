"""独立wheel给定原生网格→落格/内容/质量分/普通HTML，实际旧函数冻结参考。"""

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
    inspect_native_table_candidate,
    build_native_table_candidate,
    render_native_table_candidate,
    select_native_table_candidates,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return f


f = load("native-table-candidate")
extra = load("native-table-candidate-extra")
cases = f["cases"] + extra["cases"]


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    r = c["request"]
    before = copy.deepcopy(r)
    options = {
        k: r[k]
        for k in [
            "rows",
            "cols",
            "specs",
            "allow_single_row",
            "allow_single_column",
            "use_grid_index",
        ]
    }
    result = inspect_native_table_candidate(r["text"], **options)
    for k in ["valid", "index", "assignments"]:
        same(result[k], c[k], k)
    result = build_native_table_candidate(**r)
    same(result, c["expected"], "candidate")
    same(before, r, "ownership")
    if result["candidate"] is not None:
        same(render_native_table_candidate(result["candidate"]), result["html"], "render")
    return result


for c in cases:
    run(c)
for c in f["html"]:
    same(
        render_native_table_candidate(dict(rows=c["rows"], cells=c["cells"])),
        c["expected"],
        "standalone html",
    )
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::20]))
# 来源仲裁接已有候选，低分仍由后段拒绝，builder不会替上游编造置信证据。
good = next(c for c in cases if c["expected"]["candidate"] is not None)
result = build_native_table_candidate(**good["request"])
chosen = select_native_table_candidates([result["candidate"]])
assert chosen["selected"] == 0
low = copy.deepcopy(good["request"])
low["structure_support"] = 0.5
result = build_native_table_candidate(**low)
assert (
    result["candidate"]["score"] == 0.5
    and select_native_table_candidates([result["candidate"]])["selected"] is None
)
seed = {"keep": {"value": 1}, "candidate_rejection_gate": "old"}
before = copy.deepcopy(seed)
result = build_native_table_candidate(**good["request"], diagnostics=seed)
assert (
    result["diagnostics"]["keep"] == seed["keep"]
    and result["diagnostics"]["candidate_rejection_gate"] is None
    and seed == before
)


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("bad candidate accepted")


bad = copy.deepcopy(good["request"])
bad["rows"] = 1001
fails(lambda: build_native_table_candidate(**bad))
bad = copy.deepcopy(good["request"])
bad["structure_support"] = float("nan")
fails(lambda: build_native_table_candidate(**bad))
bad = copy.deepcopy(good["request"])
bad["specs"] = [bad["specs"][0]] * 2001
fails(lambda: build_native_table_candidate(**bad))
assert (
    capabilities()["native_table_candidate_builder"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    "Installed wheel: 327 old candidates +22 edge cases, exact grid/index/ownership, diagnostics/content/provenance/scores/plain HTML/types; 4 HTML cases; arbitration integration, ownership/concurrency/budgets passed; no physical grid discovery or semantic markup"
)
