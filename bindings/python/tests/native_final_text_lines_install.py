"""Installed complete final-title and text-line sequence; no Python parser imports."""

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


def invoke(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    try:
        if kind == "page":
            out = sdk.prepare_native_final_page_lines(
                q["page"],
                page_index=q.get("page_index", 0),
                body_profile=q.get("document_body_profile"),
                canonical_body_profile=q.get("canonical_body_profile"),
                title_profile=q.get("document_title_profile"),
            )
        elif kind == "document":
            out = sdk.prepare_native_final_document_lines(q["pages"])
        else:
            out = json.loads(sdk.legacy_pipeline_json(kind, json.dumps(q, allow_nan=False)))
    except sdk.RapidDocError as e:
        assert "error" in c, (c["label"], str(e))
        assert e.code in ("invalid_input", "resource_limit")
        return
    assert "error" not in c, c["label"]
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


pages = load("native-final-text-lines")["cases"]
for c in pages:
    invoke(c, "page")
chains = load("native-final-text-lines-chain-canonical")["cases"]
for c in chains:
    invoke(c, "document")
component_count = 0
for name, kind in [
    ("native-page-structure", "classify_native_page_structure"),
    ("native-final-title-rules-typed", "refine_native_final_titles"),
    ("native-semantic-merges", "merge_native_final_text"),
]:
    for i, c in enumerate(load(name)["cases"]):
        if i % 29 == 0 or not c["label"].startswith(("structure/", "final/", "native-")):
            invoke(c, kind)
            component_count += 1
contracts = load("native-final-line-contracts")["cases"]
for c in contracts:
    invoke(
        c,
        "classify_native_page_structure"
        if c["kind"] == "structure"
        else "refine_native_final_titles",
    )
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: invoke(c, "document"), chains[-3:]))
# Re-extract the three actual reference/appendix pages with this installed SDK.
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
same(sdk.prepare_native_final_document_lines(raw), expected["expected"], "live-final-lines")
formula_live = load("native-formula-runtime")
formula_data = (root / formula_live["pdf"]).read_bytes()
page_cases = {c["label"]: c for c in pages}
for i, c in enumerate(formula_live["cases"]):
    g = sdk.read_pdf_native_text_geometry(formula_data, page_index=i, pdfium_library=library)
    v = sdk.read_pdf_vector_geometry(formula_data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    old = page_cases["formula/" + c["label"]]
    p = copy.deepcopy(old["request"]["page"])
    same(ls, p["remaining_lines"], "live-formula-lines")
    same(v["drawing_lines"], p["drawing_lines"], "live-formula-rules")
    p["remaining_lines"] = ls
    out = sdk.prepare_native_final_page_lines(
        p, body_profile=old["request"].get("document_body_profile")
    )
    same(out, old["expected"], "live-formula-final-lines")
    assert out["formula_blocks"]
for p in [
    dict(page_size=[400.0, 600.0], remaining_lines="invalid"),
    dict(page_size=[400.0, 600.0], remaining_lines=[None]),
]:
    try:
        sdk.prepare_native_final_page_lines(p)
    except sdk.RapidDocError as e:
        assert e.code in ("invalid_input", "resource_limit")
    else:
        raise AssertionError("invalid final-page lines")
assert sdk.capabilities()["native_final_text_line_preparation"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{len(pages)} continuous final-line cases, {len(chains)} raw-source chains, {component_count} component cases, {len(contracts)} Unicode/label contracts; strict types/ownership/concurrency/errors and 5 fresh PDFium pages (3 reference/appendix, 2 formula) passed"
)
