"""Original public native-layout variants; forbid all Python model scheduling."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import hashlib, io, json, sys
from pathlib import Path
from PIL import Image
import rapid_doc_rust as sdk
from rapid_doc_rust._runtime import LazySession

root = Path(sys.argv[1]).resolve()
selected = set(sys.argv[2:])


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for key in a:
            same(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, str(a)[:160], str(b)[:160])


def forbidden(*args, **kwargs):
    raise AssertionError("Production called a Python algorithm or model scheduler")


sdk.parse_pdf_models = forbidden
LazySession.session = forbidden
cases = []
for filename in ["native-api-variants.json", "native-api-range.json"]:
    for case in json.loads((root / "build/migration" / filename).read_text(encoding="utf-8"))[
        "cases"
    ]:
        case["_frozen_single_thread"] = filename == "native-api-variants.json"
        cases.append(case)
for c in cases:
    if selected and Path(c["source"]).name not in selected:
        continue
    source = root / c["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c["sha256"]
    options = dict(c["options"])
    ocr = {"seal_enable": False}
    if c["_frozen_single_thread"]:
        # The original variant exporter supplied one thread to every model.
        for key in ["layout_config", "formula_config", "table_config"]:
            options[key] = dict(options.get(key) or {}, engine_cfg={"intra_op_num_threads": 1})
        ocr["EngineConfig.onnxruntime.intra_op_num_threads"] = 1
    result = sdk.RapidDoc(ocr_config=ocr, **options)(
        source, start_page_id=c.get("start_page_id", 0), end_page_id=c["end_page_id"]
    )
    actual = dict(
        markdown=result.markdown,
        middle_json=result.middle_json,
        content_list_json=result.content_list_json,
    )
    (root / "build/native-pipeline-variant-actual.json").write_text(
        json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    same(actual, c["expected"], c["source"])
    assets = []
    for name, png in result.images.items():
        with Image.open(io.BytesIO(png)) as im:
            assets.append(
                dict(
                    name=name,
                    mode=im.mode,
                    width=im.width,
                    height=im.height,
                    pixels_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                )
            )
    same(assets, c["assets"], c["source"] + "/assets")
    print(
        "Original native API variant passed:",
        c["source"],
        c["options"],
        len(result.markdown),
        "characters",
        flush=True,
    )
from rapid_doc_rust._native import PdfEngine

library, data = sdk._prepare_pdf(root / "demo/pdfs/示例1-论文模板.pdf", None, 256 * 1024 * 1024)
native = PdfEngine()


def cancel(event):
    raise LookupError("progress cancellation marker")


try:
    native.parse(library, data, '{"end_page":1}', forbidden, cancel)
except LookupError as exc:
    assert str(exc) == "progress cancellation marker"
else:
    raise AssertionError("Progress cancellation lost its exception type")


def missing(role):
    raise FileNotFoundError("resource descriptor marker")


try:
    native.parse(library, data, '{"end_page":1}', missing, lambda event: None)
except FileNotFoundError as exc:
    assert str(exc) == "resource descriptor marker"
else:
    raise AssertionError("Resource callback lost its exception type")
assert not any(
    n == "rapid_doc"
    or n.startswith("rapid_doc.")
    or n in ("pypdfium2", "pypdfium2_raw", "numpy", "cv2", "shapely")
    for n in sys.modules
)
print("Native public API variants and Rust-only scheduling passed; no full suite/CUDA")
