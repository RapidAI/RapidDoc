"""独立wheel旧拓扑/物理证据/候选物化，未使用Python生产算法。"""

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
    build_native_table_vector_candidate,
    select_native_table_candidates,
    render_native_table_candidate,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f["cases"]


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


cases = load("native-table-vector-grid") + load("native-table-vector-grid-edges")


def run(c):
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    result = build_native_table_vector_candidate(**r)
    same(result, c["expected"])
    same(r, before, "ownership")
    if result["candidate"] is not None:
        same(render_native_table_candidate(result["candidate"]), result["html"])


for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::15]))
good = next(
    c
    for c in cases
    if c["expected"]["candidate"] is not None
    and c["expected"]["candidate"]["score"] == 1.0
    and c["request"]["tracks"]["rows"] >= 2
    and c["request"]["tracks"]["cols"] >= 2
)
r = copy.deepcopy(good["request"])
r["diagnostics"] = {"unknown": {"keep": True}, "first_rejection_gate": "old"}
before = copy.deepcopy(r)
result = build_native_table_vector_candidate(**r)
assert (
    result["diagnostics"]["unknown"] == r["diagnostics"]["unknown"]
    and result["diagnostics"]["first_rejection_gate"] is None
    and r == before
)
assert select_native_table_candidates([result["candidate"]])["selected"] == 0


def fails(r):
    try:
        build_native_table_vector_candidate(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid vector grid accepted")


r = copy.deepcopy(good["request"])
r["tracks"]["rules"] = [r["tracks"]["rules"][0]] * 5001
fails(r)
r = copy.deepcopy(good["request"])
r["tracks"]["canonical_x_tracks"][0]["aliases"] = []
fails(r)
r = copy.deepcopy(good["request"])
r["tracks"]["x_tracks"][1] = r["tracks"]["x_tracks"][0]
fails(r)
r = copy.deepcopy(good["request"])
r["tracks"]["snap_tolerance"] = -1.0
fails(r)
r = copy.deepcopy(good["request"])
r["tracks"]["rows"] = 200
fails(r)
r = copy.deepcopy(good["request"])
r["tracks"]["is_single_row_shape"] = True
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["rows"][0]["bbox"][0] = float("nan")
fails(r)
r = copy.deepcopy(good["request"])
t = r["tracks"]
t.update(
    rows=6, cols=6, x_tracks=[50.0 * i for i in range(7)], y_tracks=[30.0 * i for i in range(7)]
)
t["canonical_x_tracks"] = [
    dict(coordinate=v, aliases=[v + i * 0.00001 for i in range(200)]) for v in t["x_tracks"]
]
t["canonical_y_tracks"] = [
    dict(coordinate=v, aliases=[v + i * 0.00001 for i in range(200)]) for v in t["y_tracks"]
]
t["rules"] = (t["rules"] * 1000)[:5000]
fails(r)
assert (
    capabilities()["native_table_vector_topology"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(cases)} old vector topology/materialization cases; exact diagnostics, physical scores, content/provenance/HTML and types; ownership/concurrency/budgets/arbitration passed; given canonical tracks, not full PDF/vector recovery"
)
