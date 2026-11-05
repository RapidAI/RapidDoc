"""Installed original spacing, complete document inline and nested-media orchestration."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import ast, base64, copy, hashlib, json, re, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))


def same(a, b, p=""):
    assert type(a) is type(b), (p, type(a), type(b), str(a)[:100], str(b)[:100])
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


def run(c, kind):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    try:
        if kind == "document":
            out = sdk.prepare_native_final_document_blocks(
                q["pages"],
                refine_blocks=q.get("refine_blocks", False),
                finalize_output=q.get("finalize_output", False),
                materialize_inline=True,
            )
        elif kind == "inline":
            out = sdk.materialize_native_document_inline(
                q["finalized_pages"],
                q["prepared_pages"],
                page_sizes=q["page_sizes"],
                page_style_lines=q["page_style_lines"],
                page_link_lines=q["page_link_lines"],
                page_spacing_lines=q.get("page_spacing_lines", []),
                script_diagnostics=q.get("script_diagnostics"),
            )
        elif c["kind"] == "prepare":
            out = sdk.prepare_native_spacing_lines(q["lines"], space_before=q.get("space_before"))
        else:
            out = sdk.apply_native_spacing_lines(q["blocks"], q["spacing_lines"], q["page_size"])
    except sdk.RapidDocError as e:
        assert "error" in c, (c["label"], e)
        assert e.code in ["invalid_input", "resource_limit"]
        return
    assert "error" not in c, c["label"]
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


spacing = load("native-spacing")["cases"] + load("native-spacing-contracts")["cases"]
direct = load("native-document-inline")["cases"]
chains = load("native-final-document-inline-chain")["cases"]
for c in spacing:
    run(c, "spacing")
for c in direct:
    run(c, "inline")
for c in chains:
    run(c, "document")
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, "document"), chains[-3:]))
# Reuse the previously verified PNG decoder/projection functions without rerunning its suite.
tree = ast.parse(
    (root / "bindings/python/tests/native_table_media_png_install.py").read_text(encoding="utf-8")
)
module = ast.Module(
    body=[
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ["pixels", "project"]
    ],
    type_ignores=[],
)
exec(compile(module, "verified_png_projection", "exec"), globals())
media_count = images = 0
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
for name in [
    "native-document-media-chain",
    "native-document-media-chain-positive-canonical-source",
]:
    f = load(name)
    data = (root / f["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == f["pdf_sha256"]
    for c in f["cases"]:
        run(c, "document")
        q = copy.deepcopy(c["request"]["pages"])
        before = copy.deepcopy(q)
        out = sdk.finalize_native_document(q, data, pdfium_library=library)
        same(project(out), c["media_expected"], c["label"] + "/media")
        same(q, before, "media-ownership")
        bad = sdk.finalize_native_document(q, b"invalid-pdf", pdfium_library=library)
        same(project(bad), c["failure_expected"], c["label"] + "/render-fallback")
        media_count += 1
        images += len(c["media_expected"]["images"])
assert images == 3
# Regenerate three PDFium pages rather than using frozen character inputs.
live = load("native-reference-runtime")
expected = next(c for c in chains if c["label"] == "live-reference-document")
data = (root / live["pdf"]).read_bytes()
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
    sdk.finalize_native_document(raw, data, pdfium_library=library),
    expected["expected"]["finalized_pages"],
    "live-document-inline-media",
)
for bad in [None, 1, "yes"]:
    try:
        sdk.prepare_native_final_document_blocks([], materialize_inline=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("strict inline flag")
assert sdk.capabilities()["compatibility"]["python_parser"] is False
assert not any(n.startswith("rapid_doc._vendor") for n in sys.modules)
print(
    f"{len(spacing)} spacing, {len(direct)} full inline, {len(chains)} raw-source chains, {media_count} media chains/{images} decoded images and 3 live PDFium pages; strict types, ownership, concurrency, old render fallback and gates passed"
)
