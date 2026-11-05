"""Installed old component order/distance/candidates and real PDF line/path inputs."""

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
    detect_native_drawing_graphics as detect,
    read_pdf_vector_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-drawing-graphics.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-drawing-graphics.json").read_text(encoding="utf-8"))
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
    out["components"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 239
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::13]))
live = [c for c in f["cases"] if "pdf_ref" in c]
assert len(live) == 41
for c in live:
    ref = c["pdf_ref"]
    data = (root / ref["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["pdf_sha256"]
    g = read(data, page_index=ref["page_index"])
    projected = [{k: l[k] for k in ["bbox", "width", "orientation"]} for l in g["drawing_lines"]]
    same(projected, c["request"]["drawing_lines"], "live line input")
    same(g["path_infos"], c["request"]["path_infos"], "live path input")
    same(
        detect(
            g["drawing_lines"],
            g["path_infos"],
            page_size=g["page_size"],
            median_height=c["request"]["median_height"],
        ),
        c["expected"],
        "live " + c["label"],
    )
assert len(pdf["cases"]) == 8


def positive(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = read(data, page_index=c["page_index"])
    same(g["drawing_lines"], c["drawing_lines"])
    same(g["path_infos"], c["path_infos"])
    same(
        detect(g["drawing_lines"], g["path_infos"], page_size=g["page_size"], median_height=10.0),
        c["expected"],
        c["label"],
    )


for c in pdf["cases"]:
    positive(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(positive, pdf["cases"][::3]))


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(drawing_lines=[], path_infos=[], page_size=[400.0, 600.0], median_height=10.0)
error(
    dict(base, drawing_lines=[dict(bbox=[0, 0, 1, 1], orientation="horizontal")] * 5001),
    "resource_limit",
)
error(dict(base, path_infos=[dict(bbox=[0, 0, 1, 1])] * 5001), "resource_limit")
error(
    dict(base, drawing_lines=[dict(bbox=[0, 0, 1, 1], orientation="horizontal")] * 4600),
    "resource_limit",
)
for q in [
    dict(base, page_size=[400]),
    dict(base, median_height=None),
    dict(base, drawing_lines=[dict(bbox=[0, 0, 1], orientation="horizontal")]),
    dict(base, drawing_lines=[dict(bbox=[0, 0, 1, 1], orientation="diagonal")]),
]:
    error(q, "invalid_input")
run(f["cases"][0])
positive(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_drawing_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "239 old component facts/134 graphic-positive, 41 live existing PDF inputs, 8 dedicated real PDF chains/2 positives; exact component positions/order/geometry/types/ownership/concurrency/budgets/recovery passed; full strong-graphic/table arbitration/cross-page/default PDF pending"
)
