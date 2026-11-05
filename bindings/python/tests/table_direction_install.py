"""独立wheel旧表格方向策略、像素和真实CPU检测/方向模型链。"""

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
    plan_table_direction,
    prepare_oriented_table_text_png,
    OnnxSession,
    RapidDocError,
    capabilities,
)
from rapid_doc_rust._native import prepare_table_direction_bytes

root = Path(sys.argv[1]).resolve()
fixtures = [
    json.loads((root / "build/migration" / name).read_text(encoding="utf-8"))
    for name in [
        "table-direction.json",
        "table-direction-portrait.json",
        "table-direction-empty-portrait.json",
    ]
]


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


class FixedDetector:
    def __init__(self, case):
        self.case = case
        self.calls = []

    def detect_table_png(self, data):
        self.calls.append(dict(image=png(data), masks=None))
        return [dict(points=q) for q in self.case["initial" if len(self.calls) == 1 else "second"]]


class FixedOrientation:
    def __init__(self, case):
        self.case = case
        self.calls = []

    def classify_orientation_png(self, data):
        self.calls.append(png(data))
        return dict(rotation_degrees=int(self.case["classifier_label"]))


for f in fixtures:
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    for c in f["plans"]:
        before = copy.deepcopy(c["request"])
        assert plan_table_direction(**c["request"]) == c["expected"], c
        assert c["request"] == before
    for c in f["pixels"]:
        _, data = prepare_table_direction_bytes(
            (root / c["source"]).read_bytes(), c["angle"], c["classifier_bgr"]
        )
        assert png(data) == c["expected"]
    for c in f["workflows"]:
        detector = FixedDetector(c)
        ori = FixedOrientation(c)
        result = prepare_oriented_table_text_png(
            root / c["source"], page=c["page"], table=c["table"], detector=detector, orientation=ori
        )
        assert detector.calls == c["trace"], (detector.calls, c["trace"])
        assert ori.calls == c["classifier_inputs"]
        assert result["boxes"] == c["final"][0]["boxes"]
        assert png(result["png"]) == c["final"][0]["image"]
        assert result["classifier_used"] == bool(ori.calls)
        assert len(result["retries"]) == len(c["trace"])
f = fixtures[0]
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
masked = json.loads(
    (root / "build/migration/table-direction-masks.json").read_text(encoding="utf-8")
)
for p, h in masked["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in masked["cases"]:
    calls = []

    class EmptyDetector:
        def detect_table_png(self, data):
            calls.append(png(data))
            return []

    r = prepare_oriented_table_text_png(
        root / c["source"],
        page=c["page"],
        table=c["table"],
        masks=c["masks"],
        detector=EmptyDetector(),
    )
    assert calls == [t["image"] for t in c["trace"]]
    assert all(t["masks"] == c["masks"] for t in c["trace"])
    assert r["rotate_label"] == c["rotate_label"]
    assert png(r["png"]) == c["image"] and not r["boxes"]
det = OnnxSession(runtime, root / f["det_model"], f["det_sha256"], threads=1)
ori = OnnxSession(runtime, root / f["orientation_model"], f["orientation_sha256"], threads=1)
real = [c for f in fixtures for c in f["real"]]


def execute(c, source=None):
    data = (root / c["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["source_sha256"]
    calls = []
    orientation_calls = []

    class Detector:
        def detect_table_png(self, png_bytes):
            r = det.detect_table_png(png_bytes)
            calls.append(dict(image=png(png_bytes), boxes=[p["points"] for p in r]))
            return r

    class Orientation:
        def classify_orientation_png(self, png_bytes):
            orientation_calls.append(png(png_bytes))
            return ori.classify_orientation_png(png_bytes)

    r = prepare_oriented_table_text_png(
        source or data,
        page=c["page"],
        table=c["table"],
        detector=Detector(),
        orientation=Orientation(),
    )
    image = r.pop("png")
    assert r == c["expected"], (r, c["expected"])
    assert png(image) == c["image"]
    assert calls == c["trace"]
    assert orientation_calls == c["classifier_inputs"]
    return r


for c in real:
    execute(c)
assert sum(c["expected"]["classifier_used"] for c in real) >= 2


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


stream = Once((root / real[0]["source"]).read_bytes())
execute(real[0], stream)
assert stream.count == 1 and not stream.closed
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(execute, real[-4:]))


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid direction accepted")


c = fixtures[1]["workflows"][0]
fails(
    lambda: prepare_oriented_table_text_png(
        root / c["source"], page=c["page"], table=c["table"], detector=FixedDetector(c)
    )
)


class BrokenOrientation:
    count = 0

    def classify_orientation_png(self, data):
        self.count += 1
        raise RuntimeError("orientation failed")


broken = BrokenOrientation()
d = FixedDetector(c)
page_before = copy.deepcopy(c["page"])
table_before = copy.deepcopy(c["table"])
try:
    prepare_oriented_table_text_png(
        root / c["source"], page=c["page"], table=c["table"], detector=d, orientation=broken
    )
except RuntimeError as e:
    assert str(e) == "orientation failed"
else:
    raise AssertionError("orientation failure swallowed")
assert (
    broken.count == 1
    and len(d.calls) == 1
    and c["page"] == page_before
    and c["table"] == table_before
)


class InvalidOrientation:
    def classify_orientation_png(self, data):
        return dict(rotation_degrees=True)


fails(
    lambda: prepare_oriented_table_text_png(
        root / c["source"],
        page=c["page"],
        table=c["table"],
        detector=FixedDetector(c),
        orientation=InvalidOrientation(),
    )
)
bad = copy.deepcopy(fixtures[0]["plans"][0]["request"])
bad["scale"] = 0
fails(lambda: plan_table_direction(**bad))
bad["scale"] = 1
bad["page"]["rotation"] = "１"
fails(lambda: plan_table_direction(**bad))
# Invalid page preflight must not run detection or mutate the caller.
d = FixedDetector(c)
before = copy.deepcopy(c["table"])
fails(
    lambda: prepare_oriented_table_text_png(
        root / c["source"], page=[], table=c["table"], detector=d
    )
)
assert not d.calls and c["table"] == before
assert capabilities()["legacy_table_direction"] and not capabilities()["legacy_table_execution"]
print(
    f"Installed wheel: 320 unique old native angle/gating cases across 3 image profiles, 18 exact pixel cases, 76 actual legacy fixed-response workflows (4 rotated masks), {len(real)} real CPU detection/orientation chains (2 classifier calls); ownership, preflight and concurrency passed"
)
