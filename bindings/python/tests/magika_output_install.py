"""Original Python Magika labels, features and complete algorithm output parity."""

import json
from pathlib import Path
import sys
from rapid_doc_rust import legacy_pipeline_json, render_model_pages
from rapid_doc_rust._runtime import runtime_path

reference = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
runtime = str(runtime_path())
request = dict(texts=[case["text"] for case in reference["languages"]], runtime=runtime)
labels = json.loads(legacy_pipeline_json("guess_code_language", json.dumps(request)))
assert labels == [case["label"] for case in reference["languages"]]
for case in reference["cases"]:
    actual = render_model_pages(case["pages"], make_mode=case["mode"], image_path_prefix="images")
    assert actual == case["expected"], (
        case["name"],
        str(actual)[:200],
        str(case["expected"])[:200],
    )
assert "rapid_doc" not in sys.modules
print(
    "Native Magika exact:",
    len(labels),
    "language cases;",
    len(reference["cases"]),
    "complete Markdown/content-list cases; no Python parser import",
)
