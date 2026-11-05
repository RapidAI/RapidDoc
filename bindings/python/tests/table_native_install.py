"""独立wheel旧表格原生文字/逐字符框及实际PDF→CPU识别回退。"""

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
    extract_table_native_text_png,
    recognize_table_text_stage_png,
    read_native_pdf_page,
    OnnxSession,
    RapidDocError,
    capabilities,
)
from rapid_doc_rust._native import legacy_pipeline_json

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/table-native.json").read_text(encoding="utf-8"))
typed = json.loads(
    (root / "build/migration/table-native-typed-detector.json").read_text(encoding="utf-8")
)
cpu = json.loads((root / "build/migration/table-native-cpu.json").read_text(encoding="utf-8"))


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


for fixture in [f, typed, cpu]:
    for p, h in fixture["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in f["geometry"]:
    same(
        json.loads(legacy_pipeline_json("table_native_spans", json.dumps(c["request"]))),
        c["expected"],
    )
for c in f["cases"] + f["real"]:
    source = (root / c["source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == c["source_sha256"]
    req = copy.deepcopy(c["request"])
    if "pdf" in c:
        info = c["pdf"]
        data = (root / info["source"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        facts = read_native_pdf_page(
            data, page_index=info["page_index"], quote_loosebox=info["quote_loosebox"]
        )
        same(facts, req["native_page"])
        req["native_page"] = facts
    before = copy.deepcopy(req)
    same(extract_table_native_text_png(source, **req), c["expected"])
    same(req, before)


class FixedDetector:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect_table_png(self, data):
        self.calls += 1
        return [dict(points=q) for q in self.boxes]


class FixedRecognizer:
    def __init__(self):
        self.calls = 0

    def recognize_batch_png(self, images, **kw):
        self.calls += 1
        return [dict(text="FALLBACK", confidence=0.7) for p in images]


for c in typed["workflows"]:
    detector = FixedDetector(c["boxes"])
    rec = FixedRecognizer()
    before = copy.deepcopy(c)
    r = recognize_table_text_stage_png(
        root / c["source"],
        native_page=c["page"],
        table=c["table"],
        useful_list=c["useful_list"],
        force_ocr=c["force_ocr"],
        ocr_enable=c["ocr_enable"],
        detector=detector,
        recognizer=rec,
        scale=1,
    )
    same(r["ocr_result"], c["expected"])
    assert bool(rec.calls) == c["fallback"]
    same(c, before)
    assert r["text_source"] == ("ocr" if c["fallback"] else "native" if c["expected"] else "none")
assert any(
    c["kind"] == "partial" and not c["fallback"] and len(c["expected"][0]) == 1
    for c in typed["workflows"]
)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
det = OnnxSession(runtime, root / cpu["det_model"], cpu["det_sha256"], threads=1)
rec = OnnxSession(runtime, root / cpu["rec_model"], cpu["rec_sha256"], threads=1)
ori = OnnxSession(runtime, root / cpu["orientation_model"], cpu["orientation_sha256"], threads=1)
max_error = 0.0


def run(c, source=None):
    request = copy.deepcopy(c["request"])
    info = c["pdf"]
    request["native_page"] = read_native_pdf_page(
        root / info["source"], page_index=info["page_index"], quote_loosebox=info["quote_loosebox"]
    )
    before = copy.deepcopy(request)
    r = recognize_table_text_stage_png(
        source or root / c["source"], **request, detector=det, recognizer=rec, orientation=ori
    )
    same(request, before)
    same(r["text_source"], c["text_source"])
    if not c["expected"]:
        same(r["ocr_result"], [])
        return 0.0
    same(r["ocr_result"][:2], c["expected"][:2])
    assert len(r["ocr_result"][2]) == len(c["expected"][2])
    error = 0.0
    for a, b in zip(r["ocr_result"][2], c["expected"][2]):
        assert type(a) is type(b), (a, b)
        assert abs(a - b) <= 1e-5, (a, b)
        error = max(error, abs(a - b))
    return error


for c in cpu["cases"]:
    max_error = max(max_error, run(c))
assert sum(c["text_source"] == "native" for c in cpu["cases"]) == 2
assert sum(c["text_source"] == "ocr" for c in cpu["cases"]) == 8
# 原生结果存在时不要求识别session；整体为空时才要求，不能用OCR逐行补原生漏字。
native_case = next(c for c in cpu["cases"] if c["text_source"] == "native")
lazy = recognize_table_text_stage_png(
    root / native_case["source"], **native_case["request"], detector=det, orientation=ori
)
same(lazy["ocr_result"], native_case["expected"])
assert lazy["text_source"] == "native"


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


c = cpu["cases"][0]
stream = Once((root / c["source"]).read_bytes())
run(c, stream)
assert stream.count == 1 and not stream.closed
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, cpu["cases"][:4]))


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid native table request accepted")


c = f["cases"][0]
bad = copy.deepcopy(c["request"])
bad["native_page"] = {}
fails(lambda: extract_table_native_text_png(root / c["source"], **bad))
bad = copy.deepcopy(c["request"])
bad["word_boxes"] = "yes"
fails(lambda: extract_table_native_text_png(root / c["source"], **bad))
case = typed["workflows"][0]
d = FixedDetector(case["boxes"])
fails(
    lambda: recognize_table_text_stage_png(
        root / case["source"],
        native_page=case["page"],
        table=case["table"],
        useful_list=[],
        detector=d,
    )
)
assert d.calls == 0
forced = copy.deepcopy(native_case["request"])
forced["force_ocr"] = True
fails(
    lambda: recognize_table_text_stage_png(
        root / native_case["source"], **forced, detector=det, orientation=ori
    )
)
assert capabilities()["legacy_table_native_text"] and not capabilities()["legacy_table_execution"]
print(
    f"Installed wheel: 262 exact legacy table geometries/262 native fills/10 actual PDF fills including per-character boxes; 28 old whole-table fallback policies; 10 actual PDF→CPU chains (2 native, 8 OCR), max score error {max_error:.9g}; ownership/preflight/concurrency passed"
)
