"""Formula permutation/order contract and cross-page plan assignment."""

from pathlib import Path
import io
import json
import sys
from PIL import Image, ImageDraw, ImageFont
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
plans = [
    dict(
        model=dict(layout_dets=[dict(category_id=14, latex="keep"), dict(category_id=1)]),
        formulas=[dict(layout_index=0, source="page-0")],
    ),
    dict(model=dict(layout_dets=[]), formulas=[]),
    dict(
        model=dict(layout_dets=[dict(category_id=8), dict(category_id=13)]),
        formulas=[dict(layout_index=0, source="page-2-a"), dict(layout_index=1, source="page-2-b")],
    ),
]
before = json.dumps(plans, sort_keys=True)
jobs = json.loads(sdk.legacy_pipeline_json("plan_model_formulas", json.dumps(dict(plans=plans))))[
    "jobs"
]
assert [(j["page"], j["layout_index"], j["source"]) for j in jobs] == [
    (0, 0, "page-0"),
    (2, 0, "page-2-a"),
    (2, 1, "page-2-b"),
]
out = json.loads(
    sdk.legacy_pipeline_json(
        "apply_model_formulas", json.dumps(dict(plans=plans, results=["", None, "x+y"]))
    )
)
assert out[0]["model"]["layout_dets"][0]["latex"] == "keep"
assert "latex" not in out[2]["model"]["layout_dets"][0]
assert out[2]["model"]["layout_dets"][1]["latex"] == "x+y"
assert json.dumps(plans, sort_keys=True) == before

from rapid_doc_rust._runtime import MANIFEST

manifest = MANIFEST["formula"]
model = sdk.OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / "rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx",
    manifest["sha256"],
    threads=0,
    max_elements=32_000_000,
)
images = []
for size, text, height in [
    ((160, 48), "x+1", 24),
    ((260, 72), "a=b+c", 32),
    ((380, 96), "1+2=3", 44),
]:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).text((8, 6), text, fill="black", font=ImageFont.load_default(size=height))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    images.append(encoded.getvalue())
expected = model.recognize_formula_batch_png(images, batch_size=1)
assert len(set(expected)) >= 2, expected
for indices in [[2, 0, 1], [2, 0, 2, 1]]:
    assert model.recognize_formula_batch_png([images[i] for i in indices], batch_size=2) == [
        expected[i] for i in indices
    ]
print(
    "Formula grouping: cross-page assignments, empty results, duplicate crops and original output order passed"
)
