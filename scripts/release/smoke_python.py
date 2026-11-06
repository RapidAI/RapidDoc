"""Run from the installed wheel, including all seven supported formats."""

import hashlib
import json
import os
from pathlib import Path
from rapid_doc_rust import RapidDoc, OnnxSession
from rapid_doc_rust._runtime import runtime_path

root = Path(os.environ["RAPIDDOC_TEST_ROOT"])
model = root / "tests/fixtures/table_partition/encoder.onnx"
session = OnnxSession(str(runtime_path()), str(model), hashlib.sha256(model.read_bytes()).hexdigest())
assert session.run([1, 1], [3.0]) == {"encoded": {"shape": [1, 1], "values": [6.0]}}
print("Installed CPU ONNX inference: PASS")
engine = RapidDoc(
    layout_config={"model_type": "docvortex_rule"},
    formula_enable=False,
    table_enable=False,
    table_config={"use_word_box": False},
)
reports = {}
for name in [
    "pdf_native.pdf",
    "synthetic.doc",
    "synthetic.docx",
    "synthetic.xls",
    "synthetic.xlsx",
    "synthetic.ppt",
    "synthetic.pptx",
]:
    r = engine(root / "tests/migration/inputs" / name, image_dir_name="figures")
    assert r.markdown, name
    reports[name] = {
        "markdown": r.markdown,
        "middle_json": r.middle_json,
        "content_list_json": r.content_list_json,
        "images": {n: hashlib.sha256(b).hexdigest() for n, b in r.images.items()},
    }
Path(os.environ["RAPIDDOC_SMOKE_REPORT"]).write_text(
    json.dumps(reports), encoding="utf8"
)
