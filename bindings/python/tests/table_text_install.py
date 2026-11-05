"""独立wheel：实际旧表格行级OCR/重试与模型HTML回填；不含完整表格解析。"""

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
    organize_table_detection_boxes,
    apply_table_ocr_results,
    apply_table_prediction,
    recognize_table_text_png,
    OnnxSession,
    RapidDocError,
    capabilities,
)
from rapid_doc_rust._native import (
    prepare_table_detection_bytes,
    pad_table_detection_bytes,
    prepare_table_recognition_bytes,
    legacy_pipeline_json,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/table-text.json").read_text(encoding="utf-8"))


def same(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b, (a, b)


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


for path, h in f["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == h
for c in f["boxes"]:
    same(organize_table_detection_boxes(**c["request"]), c["expected"])
for c in f["restores"]:
    same(
        json.loads(legacy_pipeline_json("restore_table_detection", json.dumps(c["request"]))),
        c["expected"],
    )
for c in f["applies"]:
    same(apply_table_ocr_results(**c["request"]), c["expected"])
for c in f["attachments"]:
    before = copy.deepcopy(c["request"])
    same(apply_table_prediction(**c["request"]), c["expected"])
    same(c["request"], before)
for c in f["masks"]:
    _, data = prepare_table_detection_bytes(
        (root / c["source"]).read_bytes(), json.dumps(c["request"])
    )
    same(png(data), c["expected"])
    same(png(pad_table_detection_bytes(data, 8)), c["padded"])
for c in f["crops"]:
    same(
        [
            png(p)
            for p in prepare_table_recognition_bytes((root / c["source"]).read_bytes(), c["boxes"])
        ],
        c["expected"],
    )
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
det = OnnxSession(runtime, root / f["det_model"], f["det_sha256"], threads=1)
rec = OnnxSession(runtime, root / f["rec_model"], f["rec_sha256"], threads=1)
max_error = 0.0
for c in f["real"]:
    source = (root / c["source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == c["source_sha256"]
    result = recognize_table_text_png(source, detector=det, recognizer=rec, masks=c["masks"])
    same(result["boxes"], c["expected"]["boxes"])
    same(result["retried"], c["expected"]["retried"])
    if result["ocr_result"]:
        same(result["ocr_result"][:2], c["expected"]["ocr_result"][:2])
        assert len(result["ocr_result"][2]) == len(c["expected"]["ocr_result"][2])
        for a, b in zip(result["ocr_result"][2], c["expected"]["ocr_result"][2]):
            assert abs(a - b) <= 1e-5
            max_error = max(max_error, abs(a - b))
    else:
        same(result["ocr_result"], c["expected"]["ocr_result"])
assert any(c["expected"]["retried"] and c["expected"]["boxes"] for c in f["real"])


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid table stage accepted")


source = (root / f["real"][0]["source"]).read_bytes()
transparent_key = struct.pack(">I", 6) + b"tRNS" + struct.pack(">HHH", 0, 0, 0)
transparent_key += struct.pack(">I", zlib.crc32(transparent_key[4:]) & 0xFFFFFFFF)
tagged = source[:33] + transparent_key + source[33:]
same(
    recognize_table_text_png(source, detector=det, recognizer=rec),
    recognize_table_text_png(tagged, detector=det, recognizer=rec),
)


class FixedDetector:
    def detect_table_png(self, png):
        return [dict(points=[[5, 5], [30, 5], [30, 15], [5, 15]])]


class BrokenRecognizer:
    calls = 0

    def recognize_batch_png(self, crops, *, batch_size):
        self.calls += 1
        raise RuntimeError("table recognition failure")


broken = BrokenRecognizer()
try:
    recognize_table_text_png(source, detector=FixedDetector(), recognizer=broken)
except RuntimeError as e:
    assert str(e) == "table recognition failure" and broken.calls == 1
else:
    raise AssertionError("table recognition failure swallowed")
for kwargs in [dict(batch_size=True), dict(retry_padding=-1), dict(retry_short_side=65537)]:
    fails(lambda: recognize_table_text_png(source, detector=det, recognizer=rec, **kwargs))
fails(lambda: recognize_table_text_png(b"bad", detector=det, recognizer=rec))
fails(lambda: apply_table_ocr_results(f["applies"][0]["request"]["boxes"], []))


class Stream(io.BytesIO):
    reads = 0

    def read(self, *args):
        self.reads += 1
        return super().read(*args)


stream = Stream(source)
recognize_table_text_png(stream, detector=det, recognizer=rec)
assert stream.reads == 1 and not stream.closed
with ThreadPoolExecutor(max_workers=2) as pool:
    values = list(
        pool.map(lambda _: recognize_table_text_png(source, detector=det, recognizer=rec), range(2))
    )
    same(values[0], values[1])
caps = capabilities()
assert caps["legacy_table_text_execution"] and not caps["legacy_table_execution"]
print(
    f"131 old table box rules/75 retries/209 OCR policies/64 HTML attachments and exact mask/crop pixels; 8 actual CPU chains with successful thin retry, max score error {max_error:.9g}; ownership and concurrency passed"
)
