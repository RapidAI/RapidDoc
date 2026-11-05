"""Installed exact isolated vectors/small rasters and real PDF source reads."""

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
    detect_native_isolated_graphics as detect,
    read_pdf_vector_geometry as read,
    read_pdf_image_infos as images,
    read_pdf_native_text_geometry as chars,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-isolated-graphics.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-isolated-graphics.json").read_text(encoding="utf-8"))
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
    out["vector_bboxes"].append([0, 0, 1, 1])
    same(q, before, "output independence")


assert len(f["cases"]) == 229
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::13]))


def source(data, page):
    v = read(data, page_index=page)
    g = chars(data, page_index=page)
    im = images(data, page_index=page)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    keys = [
        "text",
        "bbox",
        "source_index",
        "angle",
        "effective_height",
        "em_height",
        "style_scale_repaired",
    ]
    return dict(
        page_size=g["page_size"],
        path_infos=v["path_infos"],
        image_bboxes=[i["bbox"] for i in im["image_infos"]],
        lines=[{k: l[k] for k in keys} for l in rows],
    )


live = [c for c in f["cases"] if "pdf_ref" in c]
assert len(live) == 41
for c in live:
    ref = c["pdf_ref"]
    data = (root / ref["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["pdf_sha256"]
    s = source(data, ref["page_index"])
    same(s, c["request"]["source"], "live source")
    same(detect(s), c["expected"], "live " + c["label"])
assert len(pdf["cases"]) == 8


def positive(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    s = source(data, c["page_index"])
    same(s, c["source"], c["label"] + "/input")
    same(detect(s), c["expected"], c["label"])


for c in pdf["cases"]:
    positive(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(positive, pdf["cases"][::3]))


def error(s, code):
    try:
        detect(s)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(page_size=[400.0, 600.0])
error(dict(base, path_infos=[dict(bbox=[0, 0, 10, 10])] * 5001), "resource_limit")
error(dict(base, lines=[dict(bbox=[0, 0, 10, 10])] * 4097), "resource_limit")
error(dict(base, image_bboxes=[[0, 0, 10, 10]] * 5001), "resource_limit")
paths = [
    dict(bbox=[i * 100.0, 50.0, i * 100.0 + 40.0, 90.0], segment_count=30, fill_visible=True)
    for i in range(4600)
]
error(dict(base, path_infos=paths), "resource_limit")
error(
    dict(
        base,
        path_infos=[dict(bbox=[50.0, 50.0, 90.0, 90.0], segment_count=30, fill_visible=True)]
        * 4600,
    ),
    "resource_limit",
)
for s in [
    dict(page_size=[400]),
    dict(base, lines=[dict(bbox=[0, 0, 10])]),
    dict(
        base,
        path_infos=[
            dict(
                bbox=[50, 50, 90, 90], fill_visible=True, segment_count=30, fill_rgba=[0, 0, 0, 256]
            )
        ],
    ),
]:
    error(s, "invalid_input")
run(f["cases"][0])
positive(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_isolated_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules and "PIL" not in sys.modules
)
print(
    "229 old isolated facts/137 vector+7 small-raster positive, 41 live existing PDF source inputs and 8 dedicated real chains/3 vector+1 small-raster positive; exact merge order/fields/types/ownership/concurrency/limits/recovery passed; full strong-graphic/table arbitration/default PDF pending"
)
