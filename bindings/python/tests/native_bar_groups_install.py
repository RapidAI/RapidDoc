"""Installed real tiled-table/bar preparation to common-axis groups and numeric evidence."""

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
    prepare_native_bar_groups as prepare,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
corpora = [
    json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-bar-groups", "native-bar-group-boundaries", "pdf-bar-groups"]
]
f, boundary, pdf = corpora
for corpus in corpora:
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
    out["groups"].append({})
    out["admitted_bars"].append({})
    same(q, before, "output independence")


cases = f["cases"] + boundary["cases"]
assert len(cases) == 187
for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::9]))
assert len(pdf["cases"]) == 20


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
    same(prepare(s, em=10.0), c["expected"], c["label"])
    same(s, before, "live ownership")


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


base = dict(source=dict(page_size=[400, 600]), em=10.0)
for key, n in [("chars", 20001), ("lines", 4097), ("path_infos", 5001), ("drawing_lines", 5001)]:
    error(dict(base, source=dict(page_size=[400, 600], **{key: [{}] * n})), "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
p = q["source"]["path_infos"][0]
q["source"]["path_infos"] = [
    dict(p, bbox=[i * 1000.0, 100.0, i * 1000.0 + 80.0, 110.0]) for i in range(3200)
]
q["source"]["drawing_lines"] = []
q["source"]["lines"] = []
error(q, "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["path_infos"] = [
    dict(q["source"]["path_infos"][0], rectangle_bboxes=[[50.0, 100.0, 130.0, 110.0]] * 1600)
]
error(q, "resource_limit")
for em in [0.0, -1.0, None]:
    error(dict(base, em=em), "invalid_input")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["lines"][0]["text"] = None
error(q, "invalid_input")
run(cases[0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_bar_groups"]
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
    "187 old bar-group prefix/threshold references (152 positive, 2 table exclusions), 20 fully live PDF chars+vectors+visual-line chains (6 positive pages/10 groups/4 table exclusions); exact fields/types/order/ownership/concurrency/work/recovery passed; caption protection/body expansion/final occupied-grid/default PDF pending"
)
