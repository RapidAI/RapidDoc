"""Installed complex-container/axis stage and actual positive PDF path-read chains."""

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
    detect_native_path_graphics as detect,
    read_pdf_vector_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-path-graphics.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-path-graphics.json").read_text(encoding="utf-8"))
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
    out["axis_bboxes"].append([0, 0, 1, 1])
    same(q, before, "output independence")


assert len(f["cases"]) == 263
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
    same(g["path_infos"], c["request"]["path_infos"], "live input")
    same(
        detect(
            g["path_infos"], page_size=g["page_size"], median_height=c["request"]["median_height"]
        ),
        c["expected"],
        "live " + c["label"],
    )
assert len(pdf["cases"]) == 8


def positive(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = read(data, page_index=c["page_index"])
    same(g["path_infos"], c["path_infos"], "positive input")
    same(g["page_size"], c["page_size"])
    same(
        detect(g["path_infos"], page_size=g["page_size"], median_height=10.0),
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


base = dict(path_infos=[], page_size=[400.0, 600.0], median_height=10.0)
error(
    dict(base, path_infos=[dict(bbox=[0, 0, 100, 100], fill_visible=True)] * 5001), "resource_limit"
)
error(
    dict(
        base,
        path_infos=[
            dict(bbox=[0, 0, 100, 100], fill_visible=True, source_index=i) for i in range(4000)
        ],
    ),
    "resource_limit",
)
# Output bound independently prevents large duplicate axis arrays before serialization.
paths = (
    [dict(bbox=[50, 180, 200, 180], stroke_visible=True)] * 80
    + [dict(bbox=[50, 50, 50, 180], stroke_visible=True)] * 80
    + [dict(bbox=[60, 70, 180, 160], segment_count=6)]
)
error(dict(base, path_infos=paths), "resource_limit")
for q in [
    dict(base, page_size=[400]),
    dict(base, median_height=None),
    dict(base, path_infos=[dict(bbox=[0, 0, 1])]),
    dict(base, path_infos=[dict(bbox=[0, 0, 1, 1], stroke_visible=1)]),
]:
    error(q, "invalid_input")
run(f["cases"][0])
positive(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_path_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "263 old path facts (60 container/73 axis positive), 41 live existing PDF inputs and 8 actual positive/rotated PDF chains (4 container/2 axis positive), exact types/order/source IDs/ownership/concurrency/work+output bounds/recovery passed; full graphic/table arbitration/default PDF pending"
)
