"""Installed old concentric evidence and fully read PDF candidate/arbitration chains."""

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
    detect_native_concentric_graphics as detect,
    arbitrate_native_graphic_candidates as select,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as chars,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-concentric-graphics.json").read_text(encoding="utf-8")
)
pdf = json.loads(
    (root / "build/migration/pdf-concentric-graphics.json").read_text(encoding="utf-8")
)
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
    out["concentric_bboxes"].append([0, 0, 1, 1])
    same(q, before, "output independence")


assert len(f["cases"]) == 196
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::11]))
assert len(pdf["cases"]) == 8


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    page = c["page_index"]
    v = vectors(data, page_index=page)
    g = chars(data, page_index=page)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    source = dict(
        page_size=g["page_size"],
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        form_bboxes=[],
        lines=[
            {k: l[k] for k in ["text", "bbox", "angle", "source_index", "visual_row_id"]}
            for l in rows
        ],
    )
    same(source, c["source"], "live input")
    before = copy.deepcopy(source)
    out = detect(source, em=10.0)
    same(out, c["expected"], c["label"])
    same(
        select(source, out["concentric_bboxes"], median_height=10.0),
        c["arbitration"],
        "arbitration " + c["label"],
    )
    same(source, before, "live ownership")


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


base = dict(source=dict(page_size=[400.0, 600.0]), em=10.0)
error(
    dict(
        base,
        source=dict(page_size=[400.0, 600.0], lines=[dict(text="x", bbox=[0, 0, 1, 1])] * 4097),
    ),
    "resource_limit",
)
error(
    dict(base, source=dict(page_size=[400.0, 600.0], path_infos=[dict(bbox=[0, 0, 1, 1])] * 5001)),
    "resource_limit",
)
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["path_infos"] = [q["source"]["path_infos"][0]] * 4600
error(q, "resource_limit")
for q in [
    dict(base, em=None),
    dict(base, source=dict(page_size=[400.0])),
    dict(base, source=dict(page_size=[400.0, 600.0], lines=[dict(text=1, bbox=[0, 0, 1, 1])])),
    dict(
        base,
        source=dict(
            page_size=[400.0, 600.0], path_infos=[dict(bbox=[0, 0, 1], stroke_visible=True)]
        ),
    ),
]:
    error(q, "invalid_input")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_concentric_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "196 old concentric facts/145 positives, 8 fully live real PDF path+text+candidate+grid arbitration chains/1 positive; exact Unicode syntax/geometry/types/order/ownership/concurrency/budgets/recovery passed; complete strong-graphic/default PDF producers pending"
)
