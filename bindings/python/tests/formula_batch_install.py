"""Actual FP32 formula model: batching preserves the original decoded formulas."""

from pathlib import Path
import json, sys
from rapid_doc_rust import OnnxSession, RapidDoc

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/formula-decode.json").read_text(encoding="utf-8"))
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / f["model"]["path"],
    f["model"]["sha256"],
    threads=0,
)
count = 0
for case in f["real"]:
    sources = [(root / path).read_bytes() for path in case["sources"]]
    # Mixed lengths, repeated images and a non-divisible last batch.
    images = sources * 2 + sources[:1]
    baseline = session.recognize_formula_batch_png(images, batch_size=1)
    for batch in [2, 4]:
        assert session.recognize_formula_batch_png(images, batch_size=batch) == baseline, (
            case,
            batch,
        )
        count += len(images)
print("Actual formula outputs match single-image FP32 reference:", count, "batched images")

for invalid in [True, 0, 65, None, "4"]:
    try:
        RapidDoc(formula_config={"batch_num": invalid})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid formula batch setting accepted")
