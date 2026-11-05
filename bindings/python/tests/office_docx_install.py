"""Installed native Office: original exports, IO, lazy runtime and PDF boundary."""

import base64, gzip, hashlib, io, json, sys, tempfile, zipfile
from pathlib import Path
from PIL import Image
from loguru import logger
from rapid_doc_rust import RapidDoc
from rapid_doc_rust._native import parse_docx_bytes, RapidDocError

ROOT = Path(sys.argv[1]).resolve()
logger.remove()
base = ROOT / "tests/fixtures/office_rust_reference_068"
load = lambda name: json.loads(gzip.decompress((base / name).read_bytes()))
references = load("office_output.json.gz")
fixtures = {k: load(k)["cases"] for k in ["docx_body.json.gz", "docx_resources.json.gz"]}


def rename(value, aliases):
    if isinstance(value, str):
        for a, b in aliases.items():
            value = value.replace(a, b)
        return value
    if isinstance(value, list):
        return [rename(v, aliases) for v in value]
    if isinstance(value, dict):
        return {k: rename(v, aliases) for k, v in value.items()}
    return value


def compare(result, case):
    aliases = {}
    used = set()
    for name, data in result.images.items():
        if name.endswith(".svg"):
            # Original exact-svg structure/pixels are covered by Rust stage tests.
            matches = [n for n in case["assets"] if n.endswith(".svg") and n not in used]
        else:
            with Image.open(io.BytesIO(data)) as image:
                summary = dict(
                    size=list(image.size),
                    sha256=hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest(),
                )
            matches = [n for n, v in case["assets"].items() if v == summary and n not in used]
        assert matches, (case["name"], name)
        aliases[name] = matches[0]
        used.add(matches[0])
    assert len(used) == len(case["assets"])
    assert rename(result.middle_json, aliases) == case["middle"], case["name"] + "/middle"
    assert rename(result.markdown, aliases) == case["exports"]["mm_markdown"], (
        case["name"] + "/markdown"
    )
    assert rename(result.content_list_json, aliases) == case["exports"]["content_list"], (
        case["name"] + "/content"
    )


engine = RapidDoc()
engine._get_models = lambda: (_ for _ in ()).throw(AssertionError("DOCX must not load models"))
selected = [
    c
    for c in references["cases"]
    if c["name"].startswith(("real ", "chart cells ", "notes ")) and "error" not in c
]
for case in selected:
    source = next(c for c in fixtures[case["fixture"]] if c["name"] == case["name"])
    compare(engine(bytes.fromhex(source["package_hex"])), case)
source = ROOT / "demo/docx/demo1-docx.docx"
raw = source.read_bytes()
baseline = engine(source)
stream = io.BytesIO(raw)
stream.seek(len(raw))
for item in [str(source), raw, bytearray(raw), stream]:
    result = engine(item)
    assert result.middle_json == baseline.middle_json
    assert result.markdown == baseline.markdown
assert len(engine([source, raw])) == 2
assert engine._models is None and engine._native_engine is None
assert not any(
    name.startswith(("docx", "mammoth", "openpyxl", "rapid_doc.")) for name in sys.modules
), "Python Office parser imported"


class Writer:
    def __init__(self):
        self.data = {}

    def write(self, name, data):
        self.data[name] = data


iw = Writer()
mw = Writer()
written = engine(source, image_writer=iw, md_writer=mw, image_dir_name="assets")
assert iw.data == written.images and "demo1-docx.md" in mw.data
assert "assets/" in written.markdown and "images/" not in written.markdown
uri = engine(source, image_output_mode="data_uri")
assert "data:image/" in uri.markdown
assert uri.images == baseline.images and uri.middle_json == baseline.middle_json
with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)
    result = engine(source, output_dir=folder, f_dump_middle_json=False)
    target = folder / source.stem / "office"
    assert (target / (source.stem + ".md")).read_text(encoding="utf-8") == result.markdown
    assert not (target / (source.stem + "_middle.json")).exists()
    assert (
        json.loads((target / (source.stem + "_content_list.json")).read_text(encoding="utf-8"))
        == result.content_list_json
    )
    empty = folder / "empty.docx"
    empty.write_bytes(b"")
    assert engine(empty).markdown == ""
    # The high-level original Office route ignores PDF page selection options.
    assert engine(source, start_page_id=100, end_page_id=101).middle_json == baseline.middle_json
    for mode in ["nlp_markdown", "content_list", "content_list_v2"]:
        metadata, _ = parse_docx_bytes(raw, make_mode=mode)
        case = next(c for c in selected if c["name"] == "real demo/docx/demo1-docx.docx")
        assert json.loads(metadata)["markdown"] is not None
    metadata, _ = parse_docx_bytes(
        (ROOT / "demo/docx/equations.docx").read_bytes(),
        delimiters=json.dumps(
            dict(inline_left="\\(", inline_right="\\)", display_left="\\[", display_right="\\]")
        ),
    )
    assert "\\[" in json.loads(metadata)["markdown"] or "\\(" in json.loads(metadata)["markdown"]
try:
    engine(b"PK\x03\x04broken")
except RapidDocError as e:
    assert e.code
else:
    raise AssertionError("broken package accepted")
# An unrelated PDF text read still works after Office resource processing.
import rapid_doc_rust

pdf = ROOT / "demo/pdfs/示例1-论文模板.pdf"
text = rapid_doc_rust.parse_pdf_text(pdf, start_page=0, end_page=1, allow_partial=True)
assert text.document["source_page_count"] > 0
print(
    json.dumps(
        dict(
            installed=rapid_doc_rust.__file__,
            reference_documents=len(selected),
            io_variants=5,
            batch=True,
            writers=True,
            data_uri=True,
            empty_docx=True,
            no_python_office_imports=True,
            no_model_load=True,
            pdf_text_smoke=True,
        ),
        ensure_ascii=False,
    )
)
