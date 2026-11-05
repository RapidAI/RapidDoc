"""Installed production classifier is native, including the image paint probe."""

import hashlib
import json
from pathlib import Path
import sys
import rapid_doc_rust
from rapid_doc_rust._classification import classify

root = Path(sys.argv[1]).resolve()
reference = json.loads(
    (root / "build/migration/native-classification-documents.json").read_text(encoding="utf-8")
)
for case in reference["cases"]:
    data = (root / case["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == case["sha256"]
    assert classify(data) == case["expected"], case["label"]
package = Path(rapid_doc_rust.__file__).parent
for name in [
    "_classification_rules.py",
    "_classification_probe.py",
    "_classifier_bridge.py",
    "_layout_rules.py",
]:
    assert not (package / name).exists(), name
assert not any(
    name in sys.modules for name in ["numpy", "cv2", "shapely", "pypdf", "pypdfium2", "rapid_doc"]
)
print(
    "36 original classification documents passed; retired Python algorithms absent; classifier imports no NumPy/OpenCV/Shapely/pypdf/original parser"
)
