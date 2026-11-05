"""Installed complete original final-page output and visual/grouped-order contracts."""

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


def same(a, b, p=""):
    assert type(a) is type(b), (p, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (p, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], p + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (p, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, p + "/" + str(i))
    else:
        assert a == b, (p, a, b)


def invoke(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    try:
        if kind == "page":
            out = sdk.prepare_native_final_page_blocks(
                q["page"],
                page_index=q.get("page_index", 0),
                body_profile=q.get("document_body_profile"),
                canonical_body_profile=q.get("canonical_body_profile"),
                title_profile=q.get("document_title_profile"),
                refine_blocks=True,
                finalize_output=True,
            )
        elif kind == "document":
            out = sdk.prepare_native_final_document_blocks(
                q["pages"], refine_blocks=True, finalize_output=True
            )
        else:
            out = json.loads(sdk.legacy_pipeline_json(kind, json.dumps(q, allow_nan=False)))
    except sdk.RapidDocError as e:
        assert "error" in c, (c["label"], e)
        assert e.code in ["invalid_input", "resource_limit"]
        return
    assert "error" not in c, c["label"]
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


pages = load("native-final-page-output")["cases"]
chains = load("native-final-page-output-chain")["cases"]
for c in pages:
    invoke(c, "page")
for c in chains:
    invoke(c, "document")
for c in pages[::37]:
    if "expected" not in c:
        continue
    q = c["request"]
    same(
        sdk.prepare_native_final_page_blocks(
            q["page"],
            page_index=q.get("page_index", 0),
            body_profile=q.get("document_body_profile"),
            canonical_body_profile=q.get("canonical_body_profile"),
            title_profile=q.get("document_title_profile"),
            finalize_output=True,
        ),
        c["expected"],
        "implicit-refinement",
    )
count = 0
for name, kind in [
    ("native-annotation-bands", "recover_native_annotation_bands"),
    ("native-grouped-order", "native_grouped_order"),
]:
    for c in load(name)["cases"]:
        invoke(c, kind)
        count += 1
for c in load("native-visual-annotations")["cases"]:
    invoke(
        c, "bind_native_visual_annotations" if c["kind"] == "bind" else "native_visual_annotations"
    )
    count += 1
for c in load("native-visual-order-contracts")["cases"]:
    invoke(c, "bind_native_visual_annotations" if c["kind"] == "bind" else "native_grouped_order")
    count += 1
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: invoke(c, "document"), chains[-3:]))
live = load("native-reference-runtime")
expected = next(c for c in chains if c["label"] == "live-reference-document")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
raw = []
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
    raw.append(dict(source=source, geometry=geometry))
same(raw, expected["request"]["pages"], "live-source")
same(
    sdk.prepare_native_final_document_blocks(raw, refine_blocks=True, finalize_output=True),
    expected["expected"],
    "live-final-output",
)
for fn in [
    lambda: sdk.prepare_native_final_page_blocks({}, finalize_output=1),
    lambda: sdk.prepare_native_final_document_blocks([], finalize_output=None),
]:
    try:
        fn()
    except ValueError:
        pass
    else:
        raise AssertionError("strict finalization flag")
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{len(pages)} final pages, {len(chains)} complete raw-source chains, {count} visual/order rules and 3 live PDFium pages; strict types, ownership, concurrent calls and flags passed"
)
