"""Installed exact old raster chart stage and full live PDF source/image/noise chains."""

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
    group_native_raster_chart_descriptions as group,
    read_pdf_native_text_geometry as read,
    read_pdf_image_infos as images,
    build_native_visual_lines as visual,
    prepare_native_source_lines as prepare,
    prepare_native_image_regions as filter_images,
    filter_pdf_native_text_noise as noise,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
cases = []
for name in ["native-raster-charts", "native-raster-charts-edges"]:
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    cases.extend(f["cases"])


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
    got = group(**q)
    same(got, c["expected"], c["label"])
    same(q, before, "ownership")
    got["source"]["lines"].append({})
    same(q, before, "output independence")


assert len(cases) == 272
for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::13]))
f = json.loads((root / "build/migration/pdf-raster-charts.json").read_text(encoding="utf-8"))
assert len(f["cases"]) == 16
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
facts = []
image_pages = []
for c in f["cases"]:
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    page = c["page_index"]
    g = read(data, page_index=page)
    im = images(data, page_index=page)
    same(im["image_infos"], c["image_infos"], "actual images")
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    prepared = prepare(rows, page_size=g["page_size"], image_infos=im["image_infos"])
    same(prepared["lines"], c["prepared_lines"], "live lines")
    minimal = [
        {k: l[k] for k in ["text", "bbox", "angle", "effective_height", "dominant_font_weight"]}
        for l in prepared["lines"]
    ]
    image_pages.append(dict(lines=minimal, page_size=g["page_size"], image_infos=im["image_infos"]))
    facts.append((c, g, prepared, data))
filtered = filter_images(image_pages)
same(filtered["watermark_fingerprints"], f["watermark_fingerprints"])
for i, (c, g, p, data) in enumerate(facts):
    s = dict(
        lines=p["lines"],
        chars=g["chars"],
        image_bboxes=filtered["image_bboxes"][i],
        page_index=c["page_index"],
    )
    clean = noise(s, data, page_index=c["page_index"])["source"]
    same(clean, c["source"], "live source")
    before = copy.deepcopy(clean)
    same(group(clean), c["expected"], "live grouping " + c["label"])
    same(clean, before, "live ownership")


def error(source, code):
    try:
        group(source)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


error(dict(lines=[dict(text="x", bbox=[0, 0, 1, 1])] * 4097), "resource_limit")
error(dict(lines=[], image_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(
    dict(
        lines=[
            dict(text=str(i), bbox=[0, i * 20, 10, i * 20 + 10], effective_height=10)
            for i in range(1000)
        ]
    ),
    "resource_limit",
)
error(dict(lines=[dict(text="x", bbox=[0, 0, 1])]), "invalid_input")
error(dict(lines=[dict(text="x", bbox=[0, 0, 1, 1], paragraph_group=1.5)]), "invalid_input")
error(dict(lines=[dict(text="x", bbox=[0, 0, 1, 1], font_signature=["x", {}])]), "invalid_input")
run(cases[0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_raster_charts"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert (
    "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules and "PIL" not in sys.modules
)
print(
    "272 old raster chart facts (223 core/151 grouping positive), 16 live PDF chars/images/filter/noise/chart chains (10 core/2 grouping positive), exact fields/types/order/ownership/concurrency/budgets/recovery passed; raster helper regression scope; cross-page/default PDF orchestration pending"
)
