"""Installed API: original auto classification, actual default CPU PDFs, IO contracts."""

import ast, base64, copy, hashlib, inspect, io, json, sys, tempfile
from pathlib import Path
import rapid_doc_rust as sdk
from rapid_doc_rust import RapidDoc, RapidDocOutput
from rapid_doc_rust._classification import classify, collect_samples, get_sample_page_indices
from PIL import Image

root = Path(sys.argv[1]).resolve()
selected_sources = set(sys.argv[2:])


def load(name):
    return json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))


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
        assert a == b, (path, str(a)[:250], str(b)[:250])


tree = ast.parse((root / "rapid_doc/main.py").read_text(encoding="utf-8"))
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RapidDoc")
for name in ["__init__", "__call__", "warmup"]:
    node = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert list(inspect.signature(getattr(RapidDoc, name)).parameters) == [
        a.arg for a in node.args.args
    ], name
    for arg, value in zip(node.args.args[-len(node.args.defaults) :], node.args.defaults):
        expected = "mm_markdown" if isinstance(value, ast.Attribute) else ast.literal_eval(value)
        assert inspect.signature(getattr(RapidDoc, name)).parameters[arg.arg].default == expected, (
            name,
            arg.arg,
        )
assert list(RapidDocOutput.__dataclass_fields__) == [
    "markdown",
    "images",
    "middle_json",
    "content_list_json",
]
for c in [] if selected_sources else load("api-classification")["cases"]:
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["sha256"]
    assert classify(data) == c["expected"], c["source"]
    count = sdk.parse_pdf_text(data, start_page=0, end_page=0, allow_partial=True).document[
        "source_page_count"
    ]
    samples = collect_samples(data, get_sample_page_indices(count))
    same(samples, c["samples"], c["source"] + "/classification samples")
    print("Original classification/statistics passed:", c["source"], flush=True)

f = load("api-document")
fixed_threads = {"engine_cfg": {"intra_op_num_threads": 1}}
# The frozen exporter explicitly used one CPU thread for every model.
engine = RapidDoc(
    layout_config=fixed_threads,
    formula_config=fixed_threads,
    table_config=fixed_threads,
    ocr_config={"seal_enable": False, "EngineConfig.onnxruntime.intra_op_num_threads": 1},
)
from rapid_doc_rust._runtime import load_models

defaults = load_models({}, {}, {}, {})
assert defaults["formula_model"].threads == 0 and defaults["table_model"].threads == 4
assert (
    load_models({}, {}, {"engine_cfg": {"intra_op_num_threads": 1}}, {})["formula_model"].threads
    == 1
)
events = []


class ObservedEngine:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def parse(self, data, options, progress, device):
        def observe(event):
            events.append(event)
            return progress(event)

        return self.native.parse(data, options, observe, device)


engine._native_engine = ObservedEngine(engine._get_native_engine())


def forbidden_python_pipeline(*args, **kwargs):
    raise AssertionError("The production API called the Python inference pipeline")


# This regression must exercise the shared native engine, not the preview host.
sdk.parse_pdf_models = forbidden_python_pipeline
from rapid_doc_rust._runtime import LazySession

LazySession.session = forbidden_python_pipeline
assert selected_sources <= {Path(c["source"]).name for c in f["cases"]}, selected_sources
for index, c in enumerate(f["cases"]):
    if selected_sources and Path(c["source"]).name not in selected_sources:
        continue
    source = root / c["source"]
    out = engine(source, end_page_id=c["end_page_id"])
    destination = root / "build/rust-api-preview" / source.stem
    destination.mkdir(parents=True, exist_ok=True)
    actual = dict(
        markdown=out.markdown, middle_json=out.middle_json, content_list_json=out.content_list_json
    )
    (destination / "actual.json").write_text(
        json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "document.md").write_text(out.markdown, encoding="utf-8")
    same(actual, c["expected"], source.name)
    assets = []
    for key, blob in out.images.items():
        (destination / "images").mkdir(exist_ok=True)
        (destination / "images" / key).write_bytes(blob)
        with Image.open(io.BytesIO(blob)) as im:
            assets.append(
                dict(
                    name=key,
                    mode=im.mode,
                    width=im.width,
                    height=im.height,
                    pixels_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
                )
            )
    same(assets, c["assets"], source.name + "/assets")
    assert tuple(out) == (out.markdown, out.images)
    print(
        "Actual default CPU API output passed:",
        source.name,
        len(out.markdown),
        "characters",
        flush=True,
    )
assert not any(
    n == "rapid_doc" or n.startswith("rapid_doc.") or n in ("pypdfium2", "pypdfium2_raw")
    for n in sys.modules
)
assert not any(name in sys.modules for name in ["numpy", "cv2", "shapely"]), (
    "Production layout/classification still loaded Python algorithms"
)
groups = []
for event in events:
    if event["stage"] == "batch":
        groups.append([])
    if groups:
        groups[-1].append(event["stage"])
for stages in groups:
    assert max(i for i, s in enumerate(stages) if s == "layout") < min(
        i for i, s in enumerate(stages) if s == "formula"
    )
    assert max(i for i, s in enumerate(stages) if s == "formula") < min(
        i for i, s in enumerate(stages) if s == "ocr_detection"
    )
    assert max(i for i, s in enumerate(stages) if s == "ocr_detection") < min(
        i for i, s in enumerate(stages) if s == "ocr_recognition"
    )
print("Native phase ordering, automatic formula threads and explicit thread override passed")
print(
    "Original signatures/defaults and selected actual default API documents passed; no full suite/CUDA"
    if selected_sources
    else "Original signatures/defaults, 6 files auto classification/statistics and 5 actual default API documents passed; no full suite/CUDA"
)
