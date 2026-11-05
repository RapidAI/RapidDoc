"""Installed old graphic-caption preservation and owned semantic mutations."""

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
from rapid_doc_rust import protect_native_graphic_captions as protect, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-graphic-captions.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


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


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = protect(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["lines"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 151
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::7]))


def error(q, code):
    try:
        protect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(lines=[], candidate_bboxes=[], median_height=10.0)
error(dict(base, lines=[{}] * 4097), "resource_limit")
error(dict(base, candidate_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
seed = dict(text="Fig 1: Title", bbox=[50.0, 50.0, 180.0, 60.0], angle=0, source_index=0)
error(
    dict(base, lines=[seed] * 1500, candidate_bboxes=[[40.0, 90.0, 220.0, 200.0]] * 4000),
    "resource_limit",
)
error(dict(base, lines=[dict(seed, text="x" * 20001)]), "resource_limit")
for q in [
    dict(base, median_height=0),
    dict(base, lines=[dict(seed, font_signature=[])]),
    dict(base, candidate_bboxes=[[0, 1, 2]]),
]:
    error(q, "invalid_input")
run(f["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_graphic_captions"] and not capabilities()["legacy_native_table_recovery"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "151 original caption references (146 protected/107 semantic mutations), exact fields/types/order/Unicode seed+strong-tail/font-family/identity/ownership/concurrency/budgets/recovery passed; original English seed domain retained"
)
