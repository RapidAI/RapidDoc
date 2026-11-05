"""Installed wheel: full old text-lane routing, identity, ordering and budgets."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import infer_native_text_lanes as infer, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-text-lanes.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
assert len(f["cases"]) == 1128
assert sum(c["events"].get("nested", False) for c in f["cases"]) == 182
assert sum(c["events"].get("repeated_tail", False) for c in f["cases"]) == 24
assert sum(c["events"].get("cross_tail", False) for c in f["cases"]) == 47


def exact(a, b, p=""):
    assert type(a) is type(b), (p, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), p
        for k in a:
            exact(a[k], b[k], p + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), p
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, p + "/" + str(i))
    else:
        assert a == b, (p, a, b)


def case(c):
    v = copy.deepcopy(c["request"])
    before = copy.deepcopy(v)
    r = infer(**v)
    exact(r, c["expected"], c["label"])
    exact(v, before)
    assert Counter(i for l in r for i in l["line_indices"]) == Counter(
        range(len(v["line_geometry"]))
    )
    # Same references can be consumed repeatedly without storing mutable membership state.
    exact(infer(**v), r, c["label"])


for c in f["cases"]:
    case(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(case, f["cases"][::47]))


def rejects(v, code=None):
    try:
        infer(**v)
    except (ValueError, RapidDocError) as e:
        if code:
            assert isinstance(e, RapidDocError) and e.code == code
    else:
        raise AssertionError("invalid or over-budget lanes accepted")


base = copy.deepcopy(f["cases"][0]["request"])
for key, value in [
    ("median_height", 0),
    ("local_page_width", -1),
    ("median_height", float("nan")),
    ("recalculate_intervals", 1),
    ("line_geometry", []),
    ("line_geometry", [base["line_geometry"][0]] * 513),
]:
    rejects(dict(base, **{key: value}))
v = copy.deepcopy(base)
v["line_geometry"][1][0]["source_index"] = v["line_geometry"][0][0]["source_index"]
rejects(v)
v = copy.deepcopy(base)
v["line_geometry"][0][1] = [0.0, 0.0, 0.0, 1.0]
rejects(v)
v = copy.deepcopy(base)
v["line_geometry"][0][1] = [0.0, 0.0, 1e10, 1.0]
rejects(v)
# A valid <=512-line geometry requires many cumulative span-tail visits; forbid reset budgets.
rows = []


def add(b):
    i = len(rows)
    rows.append([dict(text="body", bbox=b, angle=0, source_index=i, effective_height=10.0), b])


for y in (0.0, 10.0, 20.0):
    add([0.0, y, 300.0, y + 10.0])
for y in (10000.0 + i * 20.0 for i in range(10)):
    add([0.0, y, 140.0, y + 10.0])
    add([160.0, y, 300.0, y + 10.0])
for i in range(480):
    y = 30.0 + i * 10.0
    add([0.0, y, 40.0, y + 10.0])
assert len(rows) == 503
rejects(dict(line_geometry=rows, local_page_width=300.0, median_height=10.0), "resource_limit")
case(f["cases"][0])
case(f["cases"][-1])
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").read_bytes() == (
    root / "licenses/MIGRATION-NOTICES.txt"
).read_bytes()
assert (
    capabilities()["native_text_lanes"] is True
    and capabilities()["legacy_native_table_recovery"] is False
)
assert "rapid_doc" not in sys.modules
print(
    "Passed 1128 complete old lane inferences: 182 nested bands, 24 repeated-indent state changes, 47 cross-tail migrations; exact membership/order/bounds/types, repeated calls, ownership, concurrency, shared 10M budget, limits and attribution; 2D merging pending"
)
