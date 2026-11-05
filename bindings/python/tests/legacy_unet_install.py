"""独立wheel旧UNET已转义OCR→模型→原HTML/最终HTML及实际全图OCR回退。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    OnnxSession,
    RapidDocError,
    match_legacy_unet_table,
    recognize_legacy_unet_table_png,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/legacy-unet.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
model = OnnxSession(runtime, root / f["model"], f["sha256"])
m = f["models"]
det = OnnxSession(runtime, root / m["det_model"], m["det_sha256"])
rec = OnnxSession(runtime, root / m["rec_model"], m["rec_sha256"])


def run(c, source=None):
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["source_sha256"]
    before = copy.deepcopy(c["ocr_result"])
    r = recognize_legacy_unet_table_png(source or data, c["ocr_result"], structure_model=model)
    assert r["raw_html"] == c["raw_html"] and r["html"] == c["html"], (
        c["source"],
        r["raw_html"],
        c["raw_html"],
    )
    assert c["ocr_result"] == before
    return r


for c in f["real"]:
    run(c)
for c in f["pipelines"]:
    r = recognize_legacy_unet_table_png(
        root / c["source"], None, structure_model=model, detector=det, recognizer=rec
    )
    assert r["html"] == c["html"], (c["source"], r["html"], c["html"])
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, f["real"][:2]))


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


c = f["real"][0]
s = Once((root / c["source"]).read_bytes())
run(c, s)
assert s.count == 1 and not s.closed


class Empty:
    def detect_table_png(self, *a):
        return []


assert recognize_legacy_unet_table_png(root / c["source"], None, detector=Empty())["html"] is None
try:
    match_legacy_unet_table({}, [[], ["x"], []])
except ValueError:
    pass
else:
    raise AssertionError("OCR length mismatch truncated")


class Broken:
    def table_cells_png(self, *a, **kw):
        raise RuntimeError("UNET failed")


try:
    recognize_legacy_unet_table_png(root / c["source"], c["ocr_result"], structure_model=Broken())
except RuntimeError as e:
    assert str(e) == "UNET failed"
else:
    raise AssertionError("UNET error swallowed")
print(
    "Installed wheel: 5 actual old UNET raw/final HTML and 5 CPU OCR→UNET outputs passed; default blank-cell rec disabled, ownership/concurrency/errors passed"
)
