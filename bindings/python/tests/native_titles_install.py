"""Installed full page/document titles and raw-document text chain; strict values/ownership."""

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


def run(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    if kind == "page":
        out = dict(
            lines=sdk.classify_native_page_titles(
                q["lines"],
                q["page_size"],
                page_index=q["page_index"],
                container_bboxes=q["container_bboxes"],
                caption_container_bboxes=q.get("caption_container_bboxes"),
                body_profile=q.get("document_body_profile"),
                title_profile=q.get("document_title_profile"),
            )
        )
    elif kind == "profile":
        out = sdk.infer_native_document_title_profile(q["pages"], q["body_profile"])
    elif kind == "document":
        out = sdk.classify_native_document_titles(
            q["pages"], q["body_profile"], q["canonical_body_profile"]
        )
    else:
        out = sdk.prepare_native_document_text(q["pages"])
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


counts = []
for c in load("native-title-contracts"):
    if "expected" in c:
        run(c, "page")
    else:
        try:
            run(c, "page")
        except sdk.RapidDocError as e:
            assert e.code == "invalid_input"
        else:
            raise AssertionError(c["label"])
for name, kind in [
    ("native-page-titles", "page"),
    ("native-title-profiles", "profile"),
    ("native-document-titles", "document"),
]:
    selected = [
        c
        for i, c in enumerate(load(name))
        if "expected" in c
        and (
            i % 43 == 0
            or not c["label"].startswith(("prepared/", "classifier/", "native-", "mixed/"))
        )
    ]
    for c in selected:
        run(c, kind)
    counts.append(len(selected))
chains = load("native-document-text-chain")
for c in chains:
    run(c, "chain")
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, "chain"), chains[-6:]))
for bad in [True, -1, 1.5, "1"]:
    try:
        sdk.classify_native_page_titles([], [400.0, 600.0], page_index=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("page_index validation")
# Fresh installed PDFium -> raw source -> complete document-text sequence.
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
    sdk.prepare_native_document_text([dict(source=source, geometry=geometry)]),
    expected,
    "live-document-text",
)
assert sdk.capabilities()["native_document_text_classification"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{counts} page/profile/document title cases, {len(chains)} complete raw-source chains, strict types, ownership/concurrency and fresh installed PDFium passed"
)
