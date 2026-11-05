"""Installed original bar geometry preparation, explicit table inputs and live PDF axes."""

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
    prepare_native_bar_geometry as prepare,
    read_pdf_vector_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-bar-geometry.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-bar-geometry.json").read_text(encoding="utf-8"))
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
    out = prepare(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["admitted_bars"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 170
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::11]))
assert len(pdf["cases"]) == 8


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = read(data, page_index=c["page_index"])
    same(v["path_infos"], c["request"]["path_infos"])
    same(v["drawing_lines"], c["request"]["drawing_lines"])
    assert c["request"]["table_bboxes"] == []
    q = dict(
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        em=10.0,
        table_bboxes=[],
        axis_queries=[],
    )
    pre = prepare(**q)
    queries = []
    for h in [True, False]:
        bars = [b["bbox"] for b in pre["admitted_bars"] if b["horizontal" if h else "vertical"]]
        if not bars:
            continue
        bounds = [
            min(b[0] for b in bars),
            min(b[1] for b in bars),
            max(b[2] for b in bars),
            max(b[3] for b in bars),
        ]
        for edge in [0, 2] if h else [3, 1]:
            queries.append(dict(bounds=bounds, horizontal=h, baseline=bars[0][edge]))
    q["axis_queries"] = queries
    same(q, c["request"], "live regenerated queries")
    before = copy.deepcopy(q)
    same(prepare(**q), c["expected"], c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::3]))


def error(q, code):
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(path_infos=[], drawing_lines=[], em=10.0, table_bboxes=[])
error(dict(base, path_infos=[dict(bbox=[0, 0, 1, 1])] * 5001), "resource_limit")
error(dict(base, table_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(
    dict(base, axis_queries=[dict(bounds=[0, 0, 1, 1], horizontal=True, baseline=0)] * 5001),
    "resource_limit",
)
compound = dict(bbox=[0, 0, 100, 100], fill_visible=True, rectangle_bboxes=[[0, 0, 100, 10]] * 1600)
error(dict(base, path_infos=[compound]), "resource_limit")
q = copy.deepcopy(f["cases"][1]["request"])
q["axis_queries"] = [q["axis_queries"][0]] * 5000
q["drawing_lines"] = [q["drawing_lines"][0]] * 5
error(q, "resource_limit")
for q in [
    dict(base, em=None),
    dict(base, table_bboxes=[[0, 0, 1]]),
    dict(base, axis_queries=[dict(bounds=[0, 0, 1, 1], horizontal=1, baseline=0)]),
]:
    error(q, "invalid_input")
try:
    prepare([], [], em=10.0)
except TypeError:
    pass
else:
    raise AssertionError("upstream table evidence must be explicit")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_bar_geometry"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "170 original bar-preparation facts/123 admission positive, 8 live PDF path+rule+regenerated zero-axis chains/8 positive; exact flags/order/types/ownership/concurrency/work+output bounds/recovery passed; table bounds explicit, complete tiled-fill/bar detection/default PDF pending"
)
