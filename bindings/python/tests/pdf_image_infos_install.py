"""Independent wheel: live raster facts and live chars/vectors/images -> source/filter chain."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    read_pdf_image_infos as read,
    read_pdf_native_text_geometry as chars,
    read_pdf_vector_geometry as vectors,
    build_native_visual_lines as visual,
    prepare_native_source_lines as source,
    prepare_native_image_regions as regions,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/pdf-image-infos.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    same(read(data, page_index=c["page_index"]), c["expected"], c.get("label", c["pdf"]))


assert len(f["cases"]) == 85
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::11]))
chain = json.loads((root / "build/migration/pdf-image-sources.json").read_text(encoding="utf-8"))
assert len(chain["pages"]) == 10
pages = []
for c in chain["pages"]:
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    index = c["page_index"]
    g = chars(data, page_index=index)
    images = read(data, page_index=index)
    v = vectors(data, page_index=index)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    prepared = source(
        rows,
        page_size=g["page_size"],
        drawing_lines=v["drawing_lines"],
        image_infos=images["image_infos"],
    )
    same(prepared, c["expected"], "live source " + c["label"])
    # Only the fields consumed by the global image stage; no frozen raster/style facts.
    lines = [
        {k: l[k] for k in ("text", "bbox", "angle", "effective_height", "dominant_font_weight")}
        for l in prepared["lines"]
    ]
    pages.append(dict(page_size=g["page_size"], lines=lines, image_infos=images["image_infos"]))
before = copy.deepcopy(pages)
same(regions(pages), chain["expected"], "live image decisions")
same(pages, before, "ownership")
data = (root / f["cases"][0]["pdf"]).read_bytes()
same(read(io.BytesIO(data)), read(data), "file-like")
for payload, kwargs, code in [
    (b"bad", {}, "corrupt_document"),
    (data, dict(page_index=999), "invalid_input"),
    (data, dict(password="a\0b"), "invalid_input"),
]:
    try:
        read(payload, **kwargs)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)
for index in [True, -1, 1.5]:
    try:
        read(data, page_index=index)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid page index")
run(f["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["pdf_image_infos"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules and "PIL" not in sys.modules
)
print(
    "85 actual old PDF pages/57 images/13 smooth/20 blank tops; 10 fully live chars+vectors+images -> source lines -> raster decisions passed; exact types/order/fingerprints/ownership/concurrency/input errors/recovery; default PDF orchestration pending"
)
