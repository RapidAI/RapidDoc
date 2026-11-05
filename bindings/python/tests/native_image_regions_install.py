"""Installed image decision stage; raster evidence is explicitly frozen, not read by this API."""

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
    prepare_native_image_regions as prepare,
    prepare_native_source_lines as source_lines,
    build_native_visual_lines as visual,
    read_pdf_character_primitives as raw,
    read_pdf_native_text_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-image-regions.json").read_text(encoding="utf-8"))
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
    got = prepare(**q)
    same(got, c["expected"], c["label"])
    same(q, before, "ownership")
    if got["image_bboxes"]:
        got["image_bboxes"][0].append([1, 2, 3, 4])
        same(q, before, "output independence")


assert len(f["cases"]) == 328
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::13]))
source = json.loads((root / "build/migration/native-source-lines.json").read_text(encoding="utf-8"))
corpora = {
    name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-visual-lines", "native-visual-lines-extra"]
}
chains = [c for c in f["cases"] if "pdf_pages" in c]
assert len(chains) == 64
for c in chains:
    q = copy.deepcopy(c["request"])
    for p, ref in zip(q["pages"], c["pdf_pages"], strict=True):
        data = (root / ref["pdf"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == ref["pdf_sha256"]
        src = source["cases"][ref["source_case"]]
        vr = src["visual_case"]
        v = corpora[vr["corpus"]]["cases"][vr["index"]]
        dedup = v.get("reader") == "deduplicated"
        g = (read if dedup else raw)(data, page_index=ref["page_index"])
        rows = visual(
            g["chars" if dedup else "restored_chars"],
            page_size=g["page_size"],
            page_rotation=g["rotation"],
        )
        sq = copy.deepcopy(src["request"])
        sq["lines"] = rows
        p["lines"] = source_lines(**sq)["lines"]
    # Image fingerprints/pixel evidence remain explicit frozen inputs here.
    same(prepare(**q), c["expected"], "live text + frozen raster " + c["label"])


def error(q, code):
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(page_size=[400, 600], lines=[], image_infos=[])
error(dict(pages=[base] * 10001), "resource_limit")
error(dict(pages=[dict(base, lines=[dict(text="x", bbox=[0, 0, 1, 1])] * 4097)]), "resource_limit")
error(dict(pages=[dict(base, image_infos=[dict(bbox=[0, 0, 1, 1])] * 5001)]), "resource_limit")
for p in [
    dict(base, page_size=[400]),
    dict(base, image_infos=[dict(bbox=[0, 0, 1, 1], fingerprint=12)]),
    dict(base, image_infos=[dict(bbox=[0, 0, 1, 1], smooth_background=1)]),
    dict(base, lines=[dict(text="x", bbox=[0, 0, 1, 1], effective_height=None)]),
]:
    error(dict(pages=[p]), "invalid_input")
many = dict(
    base,
    lines=[dict(text="x", bbox=[0, 0, 1, 1], effective_height=10)] * 2500,
    image_infos=[dict(bbox=[0, 0, 400, 600], blank_top_bbox=[0, 50, 400, 600])] * 4500,
)
error(dict(pages=[many]), "resource_limit")
run(f["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_image_regions"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "328 old image-region decisions, 33 watermark positives, 64 live-text/frozen-image PDF source chains, types/order/ownership/concurrency/limits/recovery passed; raster evidence extraction and default PDF integration remain pending"
)
