"""独立wheel：旧框规则/识别裁图/回填与真实CPU det→rec→model链。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, sys, zlib
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    organize_model_detection_boxes,
    prepare_model_recognition,
    apply_model_ocr_recognition,
    execute_model_ocr,
    OnnxSession,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/model-ocr-execution.json").read_text(encoding="utf-8"))


def equal(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            equal(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def png(data):
    offset = 8
    packed = bytearray()
    assert data.startswith(b"\x89PNG")
    while offset < len(data):
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
for i, c in enumerate(f["cases"]):
    equal(organize_model_detection_boxes(**c["request"]), c["expected"], "boxes/" + str(i))


def plan_for(c):
    plan = copy.deepcopy(c["request"]["plan"])
    data = (root / c["source"]).read_bytes()
    plan["images"] = dict(src=data, det=data)
    return plan


for i, c in enumerate(f["preparations"]):
    plan = plan_for(c)
    before = copy.deepcopy(plan)
    r = c["request"]
    result = prepare_model_recognition(plan, r["detections"], lang=r["lang"], page_index=4)
    equal(
        {k: result[k] for k in ["model", "processed", "recognition"]},
        c["expected"],
        "prep/" + str(i),
    )
    equal(plan, before)
    equal(
        [dict(key=e["key"], **png(result["images"][e["key"]])) for e in c["images"]],
        c["images"],
        "crop/" + str(i),
    )
for i, c in enumerate(f["applies"]):
    equal(apply_model_ocr_recognition(**c["request"]), c["expected"], "apply/" + str(i))
detector = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / f["det_model"],
    f["det_sha256"],
)
recognizer = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / f["rec_model"],
    f["rec_sha256"],
)
for c in f["real"]:
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["source_sha256"]
    actual = detector.detect_model_png(data)
    equal([d["points"] for d in actual], c["detected"], "real detection")
    plan = plan_for(c)
    before = copy.deepcopy(plan)
    result = execute_model_ocr(plan, detector=detector, recognizer=recognizer, page_index=4)
    equal(result["model"], c["after_recognition"], "real CPU chain")
    equal(plan, before)
    equal(
        [dict(key=e["key"], **png(result["images"][e["key"]])) for e in c["images"]],
        c["images"],
        "real rec crop",
    )
data = (root / f["real"][0]["source"]).read_bytes()
kind = b"tRNS"
payload = struct.pack(">HHH", 0, 0, 0)
transparent = (
    data[:33]
    + len(payload).to_bytes(4, "big")
    + kind
    + payload
    + zlib.crc32(kind + payload).to_bytes(4, "big")
    + data[33:]
)
equal(
    [d["points"] for d in detector.detect_model_png(transparent)],
    f["real"][0]["detected"],
    "RGB tRNS raw pixels",
)
plan = plan_for(f["real"][0])
plan["images"] = dict(src=transparent, det=transparent)
equal(
    execute_model_ocr(plan, detector=detector, recognizer=recognizer, page_index=4)["model"],
    f["real"][0]["after_recognition"],
    "RGB tRNS chain",
)
samples = f["cases"][::29]
with ThreadPoolExecutor(max_workers=3) as pool:
    for actual, c in zip(
        pool.map(lambda c: organize_model_detection_boxes(**c["request"]), samples), samples
    ):
        equal(actual, c["expected"])


class NoCall:
    def detect_model_png(self, *args):
        raise AssertionError("called before validation")

    def recognize_batch_png(self, *args, **kwargs):
        raise AssertionError("called before validation")


plan = plan_for(f["real"][0])
plan["images"].pop("src")
before = copy.deepcopy(plan)
try:
    execute_model_ocr(plan, detector=NoCall(), recognizer=NoCall())
except KeyError:
    pass
else:
    raise AssertionError("missing source accepted")
equal(plan, before)
try:
    execute_model_ocr(
        plan_for(f["real"][0]), detector=NoCall(), recognizer=NoCall(), det_batch_size=2
    )
except ValueError:
    pass
else:
    raise AssertionError("det batch2 silently substituted")
try:
    apply_model_ocr_recognition(f["applies"][0]["request"]["model"], [])
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("missing recognition accepted")


class Stream(BytesIO):
    def __init__(self, v):
        super().__init__(v)
        self.reads = 0

    def read(self, *a):
        self.reads += 1
        return super().read(*a)


plan = plan_for(f["real"][0])
stream = Stream(plan["images"]["src"])
plan["images"] = dict(src=stream, det=stream)
result = execute_model_ocr(plan, detector=detector, recognizer=recognizer, page_index=4)
equal(result["model"], f["real"][0]["after_recognition"])
assert stream.reads == 1 and not stream.closed
print(
    "Installed model OCR execution: 131 old box rules, 32 recognition plans/16 exact RGB crops, 112 old rec policies, 5 actual CPU det1→rec6→model chains including 1280/2048 large inputs; concurrency, atomic input and shared-stream guards passed"
)
