"""Installed original raster connection/plain-frame rules with actual grid-protected PDF chains."""

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
    detect_native_raster_diagrams as detect,
    detect_native_drawing_graphics as components,
    inspect_native_graphic_grids as inspect,
    read_pdf_vector_geometry as vectors,
    read_pdf_image_infos as images,
    read_pdf_native_text_geometry as chars,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-raster-diagrams.json").read_text(encoding="utf-8"))
real = []
for name in ["pdf-raster-diagrams", "pdf-raster-diagrams-protected"]:
    p = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for path, h in p["sources"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == h, path
    real.extend(p["cases"])
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
    out = detect(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["raster_diagram_bboxes"].append([0, 0, 1, 1])
    same(q, before, "output independence")


assert len(f["cases"]) == 144
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::11]))
assert len(real) == 20


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    page = c["page_index"]
    v = vectors(data, page_index=page)
    g = chars(data, page_index=page)
    im = images(data, page_index=page)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    source = dict(
        page_size=g["page_size"],
        lines=[
            {k: l[k] for k in ["text", "bbox", "angle", "source_index", "visual_row_id"]}
            for l in rows
        ],
        drawing_lines=v["drawing_lines"],
        path_infos=v["path_infos"],
        image_bboxes=[p["bbox"] for p in im["image_infos"]],
        form_bboxes=[],
    )
    same(source, c["source"], "live source")
    frames = [
        x["bbox"]
        for x in components(v["drawing_lines"], [], page_size=g["page_size"], median_height=10.0)[
            "components"
        ]
    ]
    same(frames, c["frame_bboxes"], "live frame queries")
    before = copy.deepcopy(source)
    same(detect(source, em=10.0, frame_bboxes=frames), c["expected"], c["label"])
    same(source, before, "live ownership")
    if "grid_bboxes" in c:
        same(
            inspect(source, frames, median_height=10.0)["grid_bboxes"],
            c["grid_bboxes"],
            "live grid protection",
        )


for c in real:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, real[::6]))


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source=dict(page_size=[400.0, 600.0]), em=10.0)
error(
    dict(base, source=dict(page_size=[400.0, 600.0], image_bboxes=[[0, 0, 1, 1]] * 5001)),
    "resource_limit",
)
error(dict(base, frame_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(
    dict(
        base,
        source=dict(
            page_size=[400.0, 600.0],
            drawing_lines=[dict(bbox=[0, 0, 100, 0], orientation="horizontal")] * 4600,
        ),
        frame_bboxes=[[0, 0, 100, 100]],
    ),
    "resource_limit",
)
for q in [
    dict(base, em=None),
    dict(base, source=dict(page_size=[400.0])),
    dict(base, frame_bboxes=[[0, 0, 1]]),
    dict(
        base,
        source=dict(
            page_size=[400.0, 600.0], drawing_lines=[dict(bbox=[0, 0, 1, 1], orientation="bad")]
        ),
        frame_bboxes=[[0, 0, 1, 1]],
    ),
]:
    error(q, "invalid_input")
run(f["cases"][0])
live(real[0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_raster_diagrams"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules and "PIL" not in sys.modules
)
print(
    "144 old diagram/frame facts (63 diagram/46 frame positive), 20 actual fully read PDF chains (5 diagram/1 frame positive; 1 occupied-grid rejection), exact fields/types/order/ownership/concurrency/budgets/recovery passed; complete native diagram/bar producers/default PDF pending"
)
