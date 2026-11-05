"""Installed complete file/PdfModel entry: reference text, structures and decoded media."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import ast, base64, copy, hashlib, io, json, re, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
library = str(Path(sdk.__file__).parent / "runtime/rapiddoc_pdfium.dll")
assert (
    hashlib.sha256(Path(library).read_bytes()).digest()
    == hashlib.sha256(
        (root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll").read_bytes()
    ).digest()
)


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


tree = ast.parse(
    (root / "bindings/python/tests/native_table_media_png_install.py").read_text(encoding="utf-8")
)
exec(
    compile(
        ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in ["pixels", "project"]
            ],
            type_ignores=[],
        ),
        "verified_png_projection",
        "exec",
    ),
    globals(),
)
glyph = load("native-glyph-weight")["cases"]
for c in glyph:
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    same(
        sdk.plan_native_glyph_weight(q["lines"], tight_bboxes=q.get("tight_bboxes")),
        c["plan"],
        c["label"] + "/plan",
    )
    png = None if c.get("png") is None else (root / c["png"]).read_bytes()
    if png is not None:
        assert hashlib.sha256(png).hexdigest() == c["png_sha256"]
    same(
        sdk.detect_native_glyph_weight(
            q["lines"], png, tight_bboxes=q.get("tight_bboxes"), scale=q.get("scale", 1.0)
        ),
        c["expected"],
        c["label"],
    )
    same(q, before, "glyph ownership")
same(sdk.detect_native_glyph_weight([], b"invalid-png"), [], "no candidates skip PNG")
normal = load("native-model-normalization")["cases"]
for c in normal:
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    got = sdk.normalize_native_pdf_model_text(q)
    same(got, c["expected"], c["label"])
    same(q, before, "normalization ownership")
    same(sdk.normalize_native_pdf_model_text(got), got, "idempotence")
visible = load("native-visible-text")
data = (root / visible["pdf"]).read_bytes()
assert hashlib.sha256(data).hexdigest() == visible["pdf_sha256"]
for c in visible["cases"]:
    got = sdk.read_pdf_visible_text_geometry(
        data, page_index=c["page_index"], pdfium_library=library
    )
    same({k: got[k] for k in c["expected"]}, c["expected"], c["label"])
print(
    f"{len(glyph)} glyph, {len(normal)} normalization, {len(visible['cases'])} actual visible PDF cases passed",
    flush=True,
)
sources = load("native-pdf-document")["cases"]
predictions = load("native-pdf-predictions")["cases"]
for c, expected in zip(sources, predictions, strict=True):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    same(
        sdk.collect_native_pdf_sources(data, pdfium_library=library),
        c["collected"],
        c["label"] + "/collected",
    )
    got = sdk.analyze_native_pdf(data, pdfium_library=library)
    same(got, expected["expected"], c["label"] + "/predict")
print("4 complete PDF files / 20 pages / 12 visible signature widgets passed", flush=True)
acceptance = load("native-pdf-acceptance")["cases"]
for c in acceptance:
    path = root / c["pdf"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == c["pdf_sha256"]
    got = sdk.analyze_native_pdf(path)
    same(project(got["pages"]), c["expected"], c["label"])
    assert got["source_page_count"] == len(c["expected"]["pages"])
    preview = root / "build/native-pdf-model-preview"
    preview.mkdir(parents=True, exist_ok=True)
    (preview / (path.stem + ".json")).write_text(
        json.dumps(got, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        c["label"],
        got["source_page_count"],
        "complete pages/text/structure/decoded PNG passed",
        flush=True,
    )
# File, bytes and caller-owned stream agree; no Python parser in this process.
c = sources[0]
path = root / c["pdf"]
stream = io.BytesIO(path.read_bytes())
same(sdk.collect_native_pdf_sources(stream, pdfium_library=library), c["collected"], "stream")
assert not stream.closed


def concurrent(_):
    return sdk.collect_native_pdf_sources(path, pdfium_library=library)


with ThreadPoolExecutor(max_workers=2) as pool:
    for got in pool.map(concurrent, range(2)):
        same(got, c["collected"], "concurrent")
for options in [dict(max_pixels=True), dict(max_pixels=0), dict(max_pixels=1.5)]:
    try:
        sdk.analyze_native_pdf(path, pdfium_library=library, **options)
    except ValueError:
        pass
    else:
        raise AssertionError("strict pixel limit")
for options in [dict(page_index=True), dict(page_index=-1), dict(page_index=1.5)]:
    try:
        sdk.read_pdf_visible_text_geometry(path, pdfium_library=library, **options)
    except ValueError:
        pass
    else:
        raise AssertionError("strict page index")
for raw in [b"", b"not-pdf"]:
    try:
        sdk.analyze_native_pdf(raw, pdfium_library=library)
    except (ValueError, sdk.RapidDocError):
        pass
    else:
        raise AssertionError("invalid PDF accepted")
for password in [None, "wrong"]:
    try:
        sdk.analyze_native_pdf(
            root / "build/migration/encrypted-native.pdf", pdfium_library=library, password=password
        )
    except sdk.RapidDocError as e:
        assert e.code == "encrypted"
    else:
        raise AssertionError("encrypted PDF accepted")
same(concurrent(0), c["collected"], "recovery")
assert sdk.capabilities()["native_pdf_file_collection"]
assert sdk.capabilities()["native_pdf_model_analysis"]
assert not any(sdk.compatibility_report()["claims"].values())
assert not any(
    n == "rapid_doc" or n.startswith("rapid_doc.") or n == "pypdfium2" for n in sys.modules
)
print(
    "Strict types/ownership/concurrency/error recovery and default gates passed; no full suite/CUDA or performance claims"
)
