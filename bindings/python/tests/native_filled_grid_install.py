"""Installed enclosing-fill grid detector; actual PDF path chain, no old table decisions injected."""

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
    detect_native_filled_grid_tables as detect,
    read_pdf_vector_geometry as vectors,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-filled-grid.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-filled-grid.json").read_text(encoding="utf-8"))
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
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
    out = detect(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["candidates"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 165
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::9]))
q = copy.deepcopy(f["cases"][0]["request"])
q["excluded_bboxes"] = None
same(detect(**q), f["cases"][0]["expected"], "None exclusion")
assert len(pdf["cases"]) == 28


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    q = dict(path_infos=v["path_infos"], page_size=v["page_size"], excluded_bboxes=[])
    same(q, c["request"], "live source")
    before = copy.deepcopy(q)
    same(detect(**q), c["expected"], c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::4]))


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(path_infos=[], page_size=[400, 600], excluded_bboxes=[])
error(dict(base, path_infos=[{}] * 5001), "resource_limit")
error(dict(base, excluded_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(
    dict(
        base,
        path_infos=[
            dict(bbox=[40, 100, 240, 220], form_depth=0, fill_visible=True, segment_count=5)
        ]
        * 3500,
    ),
    "resource_limit",
)
for q in [
    dict(base, page_size=[1]),
    dict(base, excluded_bboxes=[[0, 1, 2]]),
    dict(base, path_infos=[dict(bbox=[0, 1, 2], fill_visible=True, segment_count=5)]),
]:
    error(q, "invalid_input")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_filled_grid_tables"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "165 old complete enclosing-fill references/101 positive; 28 live PDF path chains/4 positive; exact candidate fields/types/order/ownership/concurrency/budgets/recovery passed; original horizontal-band orientation support retained; general closed-grid/diagram/default PDF pending"
)
