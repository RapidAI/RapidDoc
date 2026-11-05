"""Installed final-page formula prefix and raw-source document continuation."""

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


def run(c, document=False):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = (
        sdk.prepare_native_document_formulas(q["pages"])
        if document
        else sdk.prepare_native_page_formulas(q["page"])
    )
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


selected = [
    c
    for i, c in enumerate(load("native-final-formula")["cases"])
    if i % 53 == 0
    or c["label"].startswith(
        ("final/", "live-", "input/sequence/", "input/continuation/", "input/variation/")
    )
]
for c in selected:
    run(c)
proofs = load("native-final-formula-proofs")["cases"]
for c in proofs:
    run(c)
chains = load("native-final-formula-chain")["cases"]
for c in chains:
    run(c, True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, True), chains[-3:]))
# Re-read the actual formula PDF through this installed SDK before the checkpoint.
live = load("native-formula-runtime")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
finals = {c["label"]: c for c in load("native-final-formula")["cases"]}
for i, c in enumerate(live["cases"]):
    g = sdk.read_pdf_native_text_geometry(data, page_index=i, pdfium_library=library)
    v = sdk.read_pdf_vector_geometry(data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    p = dict(c["request"], remaining_lines=ls, lines=ls, drawing_lines=v["drawing_lines"])
    same(p, finals[c["label"]]["request"]["page"], "live-page")
    same(sdk.prepare_native_page_formulas(p), finals[c["label"]]["expected"], "live-final-formula")
for p in [
    dict(page_size=[400.0, 600.0], remaining_lines="invalid"),
    dict(page_size=[400.0, 600.0], remaining_lines=[None]),
]:
    try:
        sdk.prepare_native_page_formulas(p)
    except sdk.RapidDocError as e:
        assert e.code in ("invalid_input", "resource_limit")
    else:
        raise AssertionError("invalid prepared formula page")
assert sdk.capabilities()["native_final_page_formula_preparation"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{len(selected)} final-page prefixes, {len(proofs)} positive canonical/index/ink proofs, {len(chains)} raw-source document chains, ownership/concurrency/invalid inputs and 2 fresh PDFium pages passed"
)
