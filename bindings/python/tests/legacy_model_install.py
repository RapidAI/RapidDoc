"""独立wheel：实际旧布局转换/过滤/区域任务与RGB裁图，真实CPU布局推理另测。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, math, struct, sys, zlib
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    convert_model_layout,
    filter_model_layout,
    predict_model_layout,
    prepare_model_crops,
    prepare_pdf_model_inference,
    OnnxSession,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/legacy-model.json").read_text(encoding="utf-8"))


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


for path, h in fixture["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == h
for i, c in enumerate(fixture["cases"]):
    equal(convert_model_layout(**c["request"]), c["expected"], str(i))
for i, c in enumerate(fixture["filters"]):
    equal(filter_model_layout(**c["request"]), c["expected"], "filter/" + str(i))
for i, c in enumerate(fixture["crops"]):
    before = copy.deepcopy(c["request"])
    output = prepare_model_crops(root / c["source"], **c["request"])
    images = output.pop("images")
    equal(output, c["expected"], "plan/" + str(i))
    equal(c["request"], before)
    equal([dict(key=k, **png(v)) for k, v in images.items()], c["images"], "pixels/" + str(i))
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / fixture["model"],
    fixture["model_sha256"],
)
for c in fixture["real"]:
    assert hashlib.sha256(Path(c["source"]).read_bytes()).hexdigest() == c["sha256"]
    equal(
        predict_model_layout(c["source"], layout_model=session, layout_shape_mode="rect"),
        c["expected"],
        "actual CPU layout",
    )
    plan = prepare_model_crops(c["source"], c["expected"])
    assert isinstance(plan["images"], dict)
stream = BytesIO((root / fixture["crops"][1]["source"]).read_bytes())
prepare_model_crops(stream, **fixture["crops"][1]["request"])
assert not stream.closed
samples = fixture["cases"][::41]
with ThreadPoolExecutor(max_workers=4) as pool:
    for actual, c in zip(
        pool.map(lambda c: convert_model_layout(**c["request"]), samples), samples
    ):
        equal(actual, c["expected"])


class Unreadable:
    def read(self, *args):
        raise AssertionError("read before validation")


try:
    predict_model_layout(Unreadable(), layout_model=object(), layout_shape_mode="auto")
except RapidDocError as e:
    assert e.code == "unsupported"
else:
    raise AssertionError("auto shape accepted")
bad = copy.deepcopy(fixture["cases"][0]["request"])
bad["regions"][0]["order"] = -1
try:
    convert_model_layout(**bad)
except RapidDocError as e:
    assert e.code == "unsupported"
else:
    raise AssertionError("missing model order accepted")
bad = copy.deepcopy(fixture["cases"][0]["request"])
bad["regions"][0]["polygon_points"] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
try:
    convert_model_layout(**bad)
except RapidDocError as e:
    assert e.code == "unsupported"
else:
    raise AssertionError("polygon accepted as rect")
try:
    convert_model_layout(
        [fixture["cases"][0]["request"]["regions"][0]] * 1001, layout_shape_mode="rect"
    )
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("unbounded layout accepted")
pdf = prepare_pdf_model_inference(
    root / "tests/migration/inputs/pdf_native.pdf",
    layout_model=session,
    layout_shape_mode="rect",
    dpi=72.0,
)
assert (
    pdf["native_page"]["chars"] and png(pdf["page_image"])["width"] == 400 and pdf["scale"] == 1.0
)
print(
    "Installed legacy model preparation: 258 conversions/86 filters, 60 exact plans/409 RGB crops, 3 actual CPU layouts and a PDF preparation route; ownership, concurrency and unsupported/budget guards passed"
)
