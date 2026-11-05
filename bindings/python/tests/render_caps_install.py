"""Installed native public-API render scale and RGB pixels versus original PDFium."""

import hashlib
import io
import json
from pathlib import Path
import sys
from PIL import Image
import rapid_doc_rust as sdk
from rapid_doc_rust._native import render_pdf_model_page_bytes

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/api-render-caps.json").read_text(encoding="utf-8"))
for path, expected in fixture["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
for case in fixture["cases"]:
    library, data = sdk._prepare_pdf(root / case["source"], None, 256 * 1024 * 1024)
    assert hashlib.sha256(data).hexdigest() == case["source_sha256"]
    w, h, png, scale = render_pdf_model_page_bytes(library, data, case["page_index"], case["dpi"])
    assert (w, h, scale) == (case["width"], case["height"], case["scale"]), case["source"]
    with Image.open(io.BytesIO(png)) as image:
        assert hashlib.sha256(image.tobytes()).hexdigest() == case["pixels_sha256"], case["source"]
print("Original public-API render scale, long-edge cap and exact RGB pixels passed: 4 pages")
