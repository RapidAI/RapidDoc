"""Borrowed layout masks preserve the pre-change public PNG outputs, including colors."""

import hashlib
import io
import json
from pathlib import Path
import sys
from PIL import Image
from rapid_doc_rust import OnnxSession, RapidDocError
from rapid_doc_rust._runtime import geos_runtime

root = Path(sys.argv[1]).resolve()
reference = root / "build/layout-borrow-original-output-20261029.json"
fixture = json.loads((root / "build/migration/layout-post.json").read_text(encoding="utf-8"))
model = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["sha256"],
    threads=1,
    max_elements=32_000_000,
)
cases = []
for case in fixture["png_cases"]:
    data = Path(case["input"]).read_bytes()
    gray = Image.open(io.BytesIO(data)).convert("L")
    colored = Image.merge(
        "RGB", (gray, gray.point(lambda x: x // 2), gray.point(lambda x: 255 - x))
    )
    buffer = io.BytesIO()
    colored.save(buffer, format="PNG")
    for variant, png in [("original", data), ("colored", buffer.getvalue())]:
        for mode in ["rect", "auto", "poly", "quad"]:
            value = json.loads(
                model._native.layout_polygons_png_json(png, 0.5, mode, geos_runtime())
            )
            cases.append(
                dict(
                    input=Path(case["input"]).name,
                    variant=variant,
                    mode=mode,
                    png_sha256=hashlib.sha256(png).hexdigest(),
                    output=value,
                )
            )
result = dict(model_sha256=fixture["sha256"], cases=cases)
if "--record" in sys.argv:
    with reference.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
else:
    expected = json.loads(reference.read_text(encoding="utf-8"))
    for actual, baseline in zip(cases, expected["cases"]):
        assert actual == baseline, (actual["input"], actual["variant"], actual["mode"])
    assert result == expected
    for mode, threshold in [("auto", float("nan")), ("auto", -1.0), ("bad", 0.5)]:
        try:
            model._native.layout_polygons_png_json(data, threshold, mode, geos_runtime())
        except RapidDocError as error:
            assert error.code == "invalid_input"
        else:
            raise AssertionError("invalid layout settings accepted")
    assert json.loads(
        model._native.layout_polygons_png_json(data, 0.5, "auto", geos_runtime())
    ) == next(
        c["output"]
        for c in cases
        if c["input"] == Path(case["input"]).name
        and c["variant"] == "original"
        and c["mode"] == "auto"
    )
print(
    f"Layout mask outputs: {len(cases)} original/colored PNG and shape-mode cases match the preserved wheel"
)
