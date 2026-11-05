"""Installed complete page node: reference values, ownership, aliases and live PDF."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b), a, b)
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
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    if "pages" in q:
        out = sdk.prepare_native_document_pages(q["pages"], complete_page=True)
    else:
        out = sdk.finish_native_geometry_page(
            q["prepared"],
            q["plan"],
            page_index=q["page_index"],
            style_lines=q.get("style_lines", []),
            link_lines=q.get("link_lines", []),
            tight_bboxes=q.get("tight_bboxes"),
            origins=q.get("origins"),
            complete_page=True,
        )
    same(q, before, "ownership")
    same(out, c["expected"], c["label"])
    return out


f = load("native-prepared-pages")
selected = [
    c
    for i, c in enumerate(f["cases"])
    if "expected" in c
    and (
        i % 37 == 0
        or c["label"].startswith(("compact/", "merge/"))
        or c["label"] == "live-footnote-page"
    )
]
for c in selected:
    run(c)
aliases = load("native-prepared-identity")["cases"]
for c in aliases:
    run(c)
for c in load("native-prepared-font-contracts")["cases"]:
    try:
        run(c)
    except sdk.RapidDocError as e:
        assert e.code == "invalid_input"
    else:
        raise AssertionError(c["label"])
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, selected[::17]))
for bad in [None, 1, "yes"]:
    try:
        sdk.prepare_native_document_pages([], complete_page=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("complete_page type gate")
# Live evidence is re-read through this installed wheel, never a frozen-result injection.
live = load("native-footnote-runtime")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
g = sdk.read_pdf_native_text_geometry(data, pdfium_library=library)
v = sdk.read_pdf_vector_geometry(data, pdfium_library=library)
forms = sdk.read_pdf_form_geometry(data, pdfium_library=library)
ls = sdk.build_native_visual_lines(
    g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
)
source = dict(
    page_size=g["page_size"],
    page_index=0,
    lines=ls,
    chars=g["chars"],
    drawing_lines=v["drawing_lines"],
    path_infos=v["path_infos"],
    form_infos=forms["form_infos"],
    form_bboxes=forms["form_bboxes"],
    image_bboxes=[],
    signature_bboxes=[],
)
geometry = dict(chars=g["chars"])
for key, value in [("tight_bboxes", "bbox"), ("origins", "origin"), ("loose_bboxes", "bbox")]:
    geometry[key] = [
        dict(char_idx=int(i), **{value: b})
        for i, b in sorted(g[key].items(), key=lambda e: int(e[0]))
    ]
same(source, live["request"]["pages"][0]["source"], "live-source")
same(geometry, live["request"]["pages"][0]["geometry"], "live-geometry")
actual = sdk.prepare_native_document_pages(
    [dict(source=source, geometry=geometry)], complete_page=True
)
expected = next(c["expected"] for c in f["cases"] if c["label"] == "live-footnote-page")
same(actual, expected, "live-prepared")
assert actual["pages"][0]["page_footnote_groups"] == [[6, 7]]
assert all(not l["chars"] for l in actual["pages"][0]["remaining_lines"])
assert sdk.capabilities()["native_complete_page_preparation"]
print(
    f"{len(selected)} complete documents, {len(aliases)} alias/canonical cases, 3 old font errors, ownership/concurrency and installed live PDF passed"
)
