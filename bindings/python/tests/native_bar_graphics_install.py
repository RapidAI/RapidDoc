"""Installed complete original bar detector with live PDF extraction and caption side effects."""

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
    detect_native_bar_graphics as detect,
    prepare_native_bar_groups as groups,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    read_pdf_image_infos as images,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-bar-graphics.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-bar-graphics.json").read_text(encoding="utf-8"))
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
    out["source"]["lines"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 122
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
grid = next(c for c in f["cases"] if c["label"] == "occupied-grid")
assert groups(**grid["request"])["groups"] and not detect(**grid["request"])["bar_bboxes"]
assert len(pdf["cases"]) == 40
fields = "text bbox angle source_index chars font_signature dominant_font_weight effective_height em_height style_scale_repaired semantic_type title_suppressed paragraph_group visual_row_id".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    im = images(data, page_index=c["page_index"])
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    s = dict(
        page_size=g["page_size"],
        chars=g["chars"],
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        lines=[{k: l[k] for k in fields} for l in rows],
        image_bboxes=[x["bbox"] for x in im["image_infos"]],
        form_bboxes=[],
    )
    same(s, c["request"]["source"], "live source")
    before = copy.deepcopy(s)
    out = detect(s, em=10.0)
    same(out, c["expected"], c["label"])
    same(s, before, "live ownership")
    if c["grid_rejected"]:
        assert groups(s, em=10.0)["groups"] and not out["bar_bboxes"]


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::7]))


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source=dict(page_size=[400, 600]), em=10.0)
for key in ["image_bboxes", "form_bboxes"]:
    error(
        dict(base, source=dict(page_size=[400, 600], **{key: [[0, 0, 1, 1]] * 5001})),
        "resource_limit",
    )
error(dict(base, source=dict(page_size=[400, 600], lines=[{}] * 4097)), "resource_limit")
for em in [0.0, -1.0, None]:
    error(dict(base, em=em), "invalid_input")
q = copy.deepcopy(f["cases"][0]["request"])
p = q["source"]["path_infos"][0]
q["source"]["path_infos"] = [
    dict(p, bbox=[i * 1000.0, 100.0, i * 1000.0 + 80.0, 110.0]) for i in range(3200)
]
q["source"]["lines"] = []
q["source"]["drawing_lines"] = []
error(q, "resource_limit")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_bar_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules and "PIL" not in sys.modules
)
print(
    "122 complete old bar references (121 positive/1 source mutation/1 occupied-grid rejection), 40 fully live PDF chars+images+vectors+visual-lines/full detector chains (10 positive/1 caption mutation/1 grid rejection); exact fields/types/order/ownership/concurrency/budgets/recovery passed; complete default PDF/other strong-graphic producers pending"
)
