"""Run from the installed wheel, including all seven supported formats."""

import hashlib
import json
import os
from pathlib import Path
from rapid_doc_rust import RapidDoc

root = Path(os.environ["RAPIDDOC_TEST_ROOT"])
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
