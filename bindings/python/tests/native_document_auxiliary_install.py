"""Installed sequential document preparation and contiguous auxiliary/body sequence."""

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
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))[
    "cases"
]


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


def run(c, chain=False):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    fn = sdk.prepare_native_document_auxiliary if chain else sdk.classify_native_document_auxiliary
    if "error" in c:
        try:
            fn(q["pages"])
        except sdk.RapidDocError as e:
            assert e.code == "invalid_input"
        else:
            raise AssertionError(c["label"])
    else:
        same(fn(q["pages"]), c["expected"], c["label"])
    same(q, before, "ownership")


sequences = load("native-document-auxiliary-canonical")
selected = [
    c for i, c in enumerate(sequences) if i % 41 == 0 or not c["label"].startswith("prepared/")
]
for c in selected:
    run(c)
proofs = load("native-document-auxiliary-proofs")
for c in proofs:
    run(c)
chains = load("native-document-auxiliary-chain")
for c in chains:
    run(c, True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, True), chains[-6:]))
for fn in [sdk.prepare_native_document_auxiliary, sdk.classify_native_document_auxiliary]:
    try:
        fn([{}])
    except sdk.RapidDocError as e:
        assert e.code in ("invalid_input", "resource_limit")
    else:
        raise AssertionError("invalid page")
assert sdk.capabilities()["native_document_auxiliary_preparation"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
# Fresh PDFium data enters the complete new chain; old Python is never imported here.
live = json.loads(
    (root / "build/migration/native-footnote-runtime.json").read_text(encoding="utf-8")
)
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
expected = next(c["expected"] for c in chains if c["label"] == "prepared/live-footnote-page")
same(
    sdk.prepare_native_document_auxiliary([dict(source=source, geometry=geometry)]),
    expected,
    "live-document",
)
print(
    f"{len(selected)} auxiliary/body sequences, {len(proofs)} directed proofs, {len(chains)} complete chains, ownership/concurrency, input errors and fresh installed PDFium document passed"
)
