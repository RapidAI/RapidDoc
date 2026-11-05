"""Installed tiled-table detector and fully live PDF/table/bar exclusion chain."""

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
    detect_native_tiled_tables as detect,
    prepare_native_bar_geometry as bars,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-tiled-tables.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-tiled-tables.json").read_text(encoding="utf-8"))
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


assert len(f["cases"]) == 168
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::9]))
assert len(pdf["cases"]) == 12


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    lines = [{k: l[k] for k in ["text", "bbox", "source_index", "chars"]} for l in rows]
    s = dict(
        page_size=g["page_size"],
        chars=g["chars"],
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        lines=lines,
    )
    same(s, c["request"]["source"], "live source")
    before = copy.deepcopy(s)
    out = detect(s)
    same(out, c["expected"], c["label"])
    same(s, before, "live ownership")
    if out["candidates"]:
        assert bars(v["path_infos"], v["drawing_lines"], em=10.0, table_bboxes=[])["admitted_bars"]
        assert not bars(
            v["path_infos"],
            v["drawing_lines"],
            em=10.0,
            table_bboxes=[t["bbox"] for t in out["candidates"]],
        )["admitted_bars"]


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::3]))


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source=dict(page_size=[400, 600]))
for key, n, value in [
    ("chars", 20001, {}),
    ("lines", 4097, {}),
    ("path_infos", 5001, {}),
    ("drawing_lines", 5001, {}),
]:
    error(dict(source=dict(page_size=[400, 600], **{key: [value] * n})), "resource_limit")
error(dict(base, excluded_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
p = q["source"]["path_infos"][0]
q["source"]["path_infos"] = [
    dict(p, bbox=[i * 1000.0, 100.0, i * 1000.0 + 100.0, 130.0]) for i in range(3200)
]
q["source"]["drawing_lines"] = []
q["source"]["lines"] = []
error(q, "resource_limit")
for q in [
    dict(source=dict(page_size=[1])),
    dict(
        source=dict(
            page_size=[400, 600], chars=[dict(char="中", bbox=[0, 0, 1], tight_bbox=[0, 0, 10, 10])]
        )
    ),
    dict(f["cases"][0]["request"], excluded_bboxes=[[0, 0, 1]]),
]:
    error(q, "invalid_input")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_tiled_tables"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules
    and "pypdfium2" not in sys.modules
    and "fontTools" not in sys.modules
    and "reportlab" not in sys.modules
)
print(
    "168 old full tiled-table references/140 positives/2 caption transfers; 12 fully live PDF char+vector+visual-line chains/4 table+bar-exclusion positives/1 inferred-grid positive; exact fields/types/order/ownership/concurrency/work bounds/recovery passed; broader filled-grid family/full bar grouping/default PDF pending"
)
