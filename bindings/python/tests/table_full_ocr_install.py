"""独立wheel旧全图OCR回退：真实直接TextRecognizer、0.5过滤、无白边/预遮罩及模型输入。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, struct, sys, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    recognize_table_full_ocr_png,
    prepare_table_model_input_png,
    OnnxSession,
    RapidDocError,
    capabilities,
)
from rapid_doc_rust._native import legacy_pipeline_json

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/table-full-ocr.json").read_text(encoding="utf-8"))


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


def png(data):
    offset = 8
    packed = bytearray()
    assert data.startswith(b"\x89PNG")
    while offset + 12 <= len(data):
        n = int.from_bytes(data[offset : offset + 4], "big")
        k = data[offset + 4 : offset + 8]
        v = data[offset + 8 : offset + 8 + n]
        offset += n + 12
        if k == b"IHDR":
            w, h, depth, color, compression, filter_, interlace = struct.unpack(">IIBBBBB", v)
        if k == b"IDAT":
            packed.extend(v)
    assert (depth, color, compression, filter_, interlace) == (8, 2, 0, 0, 0)
    raw = zlib.decompress(packed)
    stride = w * 3
    out = bytearray()
    prev = bytearray(stride)
    for y in range(h):
        p = y * (stride + 1)
        method = raw[p]
        row = bytearray(raw[p + 1 : p + 1 + stride])
        for i in range(stride):
            a = row[i - 3] if i >= 3 else 0
            b = prev[i]
            c = prev[i - 3] if i >= 3 else 0
            if method == 1:
                add = a
            elif method == 2:
                add = b
            elif method == 3:
                add = (a + b) // 2
            elif method == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                add = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else:
                assert method == 0
                add = 0
            row[i] = (row[i] + add) & 255
        out.extend(row)
        prev = row
    return dict(width=w, height=h, pixels_sha256=hashlib.sha256(out).hexdigest())


for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in f["applies"]:
    same(
        json.loads(legacy_pipeline_json("apply_table_full_ocr", json.dumps(c["request"]))),
        c["expected"],
    )


class FixedDetector:
    def __init__(self, c):
        self.case = c
        self.calls = []

    def detect_table_png(self, data):
        self.calls.append(png(data))
        return [dict(points=q) for q in self.case["boxes"]]


class FixedRecognizer:
    def __init__(self, c):
        self.case = c
        self.calls = []

    def recognize_batch_png(self, images, **kw):
        self.calls.append([png(p) for p in images])
        return [dict(text=self.case["text"], confidence=self.case["score"]) for _ in images]


for c in f["cases"]:
    source = (root / c["source"]).read_bytes()
    d = FixedDetector(c)
    r = FixedRecognizer(c)
    before = copy.deepcopy(c)
    if c["raises"]:
        try:
            recognize_table_full_ocr_png(source, detector=d, recognizer=r, masks=c["masks"])
        except ValueError:
            pass
        else:
            raise AssertionError("legacy empty recognition list silently accepted")
        assert not r.calls
    else:
        result = recognize_table_full_ocr_png(source, detector=d, recognizer=r, masks=c["masks"])
        same(result["ocr_result"], c["expected"])
        same(result["boxes"], c["expected"][0] if c["expected"] else [])
        same(r.calls, [c["crops"]])
    same(d.calls, [png(source)])
    same(c, before)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
det = OnnxSession(runtime, root / f["det_model"], f["det_sha256"], threads=1)
rec = OnnxSession(runtime, root / f["rec_model"], f["rec_sha256"], threads=1)


def rows_same(a, b):
    if not b:
        same(a, b)
        return 0.0
    same(a[:2], b[:2])
    assert len(a[2]) == len(b[2])
    error = 0.0
    for x, y in zip(a[2], b[2]):
        assert type(x) is type(y) and abs(x - y) <= 1e-5, (x, y)
        error = max(error, abs(x - y))
    return error


def full(c, source=None):
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["source_sha256"]
    result = recognize_table_full_ocr_png(
        source or data, detector=det, recognizer=rec, masks=c["masks"]
    )
    same(result["boxes"], c["expected"]["boxes"])
    return rows_same(result["ocr_result"], c["expected"]["ocr_result"])


max_error = max(full(c) for c in f["real"])
for c in f["resolved"]:
    source = (root / c["source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == c["source_sha256"]
    before = copy.deepcopy(c["request"])
    provided = bool(c["request"]["ocr_result"])
    r = prepare_table_model_input_png(
        source,
        **c["request"],
        detector=None if provided else det,
        recognizer=None if provided else rec,
    )
    image = r.pop("png")
    max_error = max(max_error, rows_same(r.pop("ocr_result"), c["expected"]["ocr_result"]))
    expected = {k: v for k, v in c["expected"].items() if k != "ocr_result"}
    same(r, expected)
    same(png(image), c["image"])
    same(c["request"], before)
assert any(c["expected"]["full_ocr_used"] and not c["expected"]["can_infer"] for c in f["resolved"])


class EmptyDetector:
    calls = 0

    def detect_table_png(self, data):
        self.calls += 1
        return []


empty = EmptyDetector()
r = prepare_table_model_input_png(
    (root / f["real"][0]["source"]).read_bytes(),
    None,
    detector=empty,
    fill_images=[dict(uuid="ignored")],
    formulas=[],
)
assert empty.calls == 1 and not r["can_infer"] and r["ocr_result"] is None


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


c = f["real"][0]
data = (root / c["source"]).read_bytes()
stream = Once(data)
full(c, stream)
assert stream.count == 1 and not stream.closed
stream = Once(data)
prepare_table_model_input_png(stream, None, detector=det, recognizer=rec)
assert stream.count == 1 and not stream.closed
chunk = struct.pack(">I", 6) + b"tRNS" + struct.pack(">HHH", 0, 0, 0)
chunk += struct.pack(">I", zlib.crc32(chunk[4:]) & 0xFFFFFFFF)
full(c, data[:33] + chunk + data[33:])
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(full, f["real"][:4]))


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid full OCR accepted")


fails(lambda: recognize_table_full_ocr_png(data, detector=det, batch_size=0))
fails(lambda: recognize_table_full_ocr_png(data, detector=det, masks=[dict(bbox=[1, 2])]))
fails(lambda: prepare_table_model_input_png(data, None))
assert capabilities()["legacy_table_full_ocr"] and not capabilities()["legacy_table_execution"]
print(
    f"Installed wheel: 243 actual old full-OCR policies/81 score boundaries, exact pre-filter crops and original-image detection; 8 real direct-rec CPU references/16 model-input resolutions, exact boxes/text/RGB, max score error {max_error:.9g}; no-retry/empty-skip/ownership/tRNS/concurrency passed; structure network not executed"
)
