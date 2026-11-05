"""Installed reference-region/group/compact sequence and raw-source PDF chain."""

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


def run(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    if kind == "document":
        out = sdk.classify_native_document_references(q["pages"])
    elif kind == "group":
        out = sdk.group_native_reference_lines(q["page"])
    else:
        out = sdk.prepare_native_document_references(q["pages"])
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


selected = [
    c
    for i, c in enumerate(load("native-reference-document-canonical")["cases"])
    if i % 47 == 0 or not c["label"].startswith("titles/")
]
for c in selected:
    run(c, "document")
groups = load("native-reference-groups")["cases"]
for c in groups:
    run(c, "group")
chains = load("native-reference-chain")["cases"]
for c in chains:
    run(c, "chain")
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, "chain"), chains[-4:]))
# Fresh three-page PDFium extraction; no old Python parser in this process.
live = load("native-reference-runtime")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
pages = []
for i in range(3):
    g = sdk.read_pdf_native_text_geometry(data, page_index=i, pdfium_library=library)
    v = sdk.read_pdf_vector_geometry(data, page_index=i, pdfium_library=library)
    forms = sdk.read_pdf_form_geometry(data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    source = dict(
        page_size=g["page_size"],
        page_index=i,
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
            dict(char_idx=int(j), **{value: b})
            for j, b in sorted(g[key].items(), key=lambda e: int(e[0]))
        ]
    pages.append(dict(source=source, geometry=geometry))
same(pages, live["request"]["pages"], "live-raw-pages")
actual = sdk.prepare_native_document_references(pages)
same(actual, live["expected"], "live-document")
assert all(p["numbered_references"] and p["reference_regions"] for p in actual["pages"][:2])
assert not actual["pages"][2].get("reference_regions")
for fn, arg in [
    (
        sdk.classify_native_document_references,
        [dict(page_size=[400.0, 600.0], remaining_lines="invalid")],
    ),
    (sdk.group_native_reference_lines, dict(remaining_lines="invalid")),
]:
    try:
        fn(arg)
    except sdk.RapidDocError as e:
        assert e.code in ("invalid_input", "resource_limit")
    else:
        raise AssertionError("invalid reference lines")
assert sdk.capabilities()["native_document_reference_preparation"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{len(selected)} document reference/compact cases, {len(groups)} group cases, {len(chains)} raw-source chains, strict types/ownership/concurrency and 3 fresh PDFium pages passed"
)
