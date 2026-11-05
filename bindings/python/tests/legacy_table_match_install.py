"""独立wheel旧SLANet raw matcher/逻辑点，真实结构网络及OCR→网络串联。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, math, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    match_legacy_slanet_table,
    recognize_legacy_slanet_raw_png,
    OnnxSession,
    RapidDocError,
    capabilities,
)
from rapid_doc_rust._native import legacy_pipeline_json

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/legacy-table-match.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b), a, b)
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for section, operation in [
    ("geometry", "legacy_table_match_geometry"),
    ("assembled", "legacy_table_match_assemble"),
]:
    for c in f[section]:
        before = copy.deepcopy(c["request"])
        same(json.loads(legacy_pipeline_json(operation, json.dumps(c["request"]))), c["expected"])
        same(c["request"], before)
for c in f["runs"]:
    before = copy.deepcopy(c["request"])
    same(match_legacy_slanet_table(**c["request"]), c["expected"])
    same(c["request"], before)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
models = {
    c["variant"]: OnnxSession(runtime, root / c["model"], c["sha256"], threads=1)
    for c in f["real"][::3]
}
det = OnnxSession(runtime, root / f["det_model"], f["det_sha256"], threads=1)
rec = OnnxSession(runtime, root / f["rec_model"], f["rec_sha256"], threads=1)


def structure_same(a, b):
    same(a["tokens"], b["tokens"])
    assert abs(a["confidence"] - b["confidence"]) <= 1e-5
    assert len(a["cell_bboxes"]) == len(b["cell_bboxes"])
    same(a["cell_bboxes"], b["cell_bboxes"])


def run(c, source=None):
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["source_sha256"]
    before = copy.deepcopy(c["ocr_result"])
    r = recognize_legacy_slanet_raw_png(
        source or data, c["ocr_result"], structure_model=models[c["variant"]], variant=c["variant"]
    )
    structure_same(r["structure"], c["structure"])
    same(r["match"], c["expected"])
    same(r["raw_html"], c["expected"]["raw_html"])
    assert not r["full_ocr_used"]
    same(c["ocr_result"], before)
    return r


for c in f["real"]:
    run(c)
for c in f["pipelines"]:
    r = recognize_legacy_slanet_raw_png(
        root / c["source"],
        None,
        structure_model=models[c["variant"]],
        variant=c["variant"],
        detector=det,
        recognizer=rec,
    )
    assert r["full_ocr_used"]
    if c["expected"] is None:
        same(r["structure"], None)
        same(r["match"], None)
        same(r["raw_html"], None)
    else:
        structure_same(r["structure"], c["structure"])
        same(r["match"], c["expected"])
        same(r["raw_html"], c["expected"]["raw_html"])
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, f["real"][:4]))


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


c = f["real"][0]
stream = Once((root / c["source"]).read_bytes())
run(c, stream)
assert stream.count == 1 and not stream.closed


class EmptyDetector:
    def detect_table_png(self, data):
        return []


same(
    recognize_legacy_slanet_raw_png(root / c["source"], None, detector=EmptyDetector()),
    dict(raw_html=None, structure=None, match=None, full_ocr_used=True),
)


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid matcher input accepted")


bad = copy.deepcopy(f["runs"][0]["request"])
bad["ocr_result"][2] = []
fails(lambda: match_legacy_slanet_table(**bad))
fails(lambda: recognize_legacy_slanet_raw_png(root / c["source"], c["ocr_result"]))
fails(lambda: recognize_legacy_slanet_raw_png(root / c["source"], c["ocr_result"], variant="unet"))


class BrokenModel:
    def table_structure_legacy_png(self, *a, **k):
        raise RuntimeError("structure failed")


before = copy.deepcopy(c["ocr_result"])
try:
    recognize_legacy_slanet_raw_png(
        root / c["source"], c["ocr_result"], structure_model=BrokenModel()
    )
except RuntimeError as e:
    assert str(e) == "structure failed"
else:
    raise AssertionError("structure error swallowed")
same(c["ocr_result"], before)
assert (
    capabilities()["legacy_slanet_matcher"]
    and capabilities()["legacy_slanet_raw_inference"]
    and not capabilities()["legacy_table_execution"]
)
print(
    "Installed wheel: 54 old geometry, 156 raw HTML/logic, 96 format→match cases; 6 actual plus/1m structures and 6 CPU OCR compositions (2 empty skip) match exact raw HTML/assignments/types; ownership/concurrency/errors passed; final BeautifulSoup normalization not executed"
)
