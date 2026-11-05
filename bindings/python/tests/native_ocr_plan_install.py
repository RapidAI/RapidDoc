"""独立wheel：旧原生检测框/native模型与实际送入检测模型的遮罩补白RGB像素。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, sys, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    derive_native_ocr_boxes,
    prepare_model_ocr,
    prepare_model_crops,
    prepare_pdf_model_inference,
    OnnxSession,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
load = lambda name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
f = load("native-ocr-plan")
base = load("legacy-model")


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
assert len(f["boxes"]) == 182 and len(f["plans"]) == 54
for i, c in enumerate(f["boxes"]):
    equal(derive_native_ocr_boxes(**c["request"]), c["expected"], "boxes/" + str(i))
for i, c in enumerate(f["plans"]):
    b = base["crops"][c["crop_case"]]
    plan = prepare_model_crops(root / b["source"], **b["request"])
    before = copy.deepcopy(plan)
    r = c["request"]
    result = prepare_model_ocr(
        plan,
        native_page=r["native_page"],
        scale=r["scale"],
        ocr_enable=r["ocr_enable"],
        use_det_mode=r["use_det_mode"],
        page_index=3,
    )
    equal({k: result[k] for k in ["model", "native", "detect"]}, c["expected"], "plan/" + str(i))
    equal(plan, before)
    equal(
        [dict(key=image["key"], **png(result["images"][image["key"]])) for image in c["images"]],
        c["images"],
        "pixels/" + str(i),
    )
samples = f["boxes"][::37]
with ThreadPoolExecutor(max_workers=4) as pool:
    for actual, c in zip(
        pool.map(lambda c: derive_native_ocr_boxes(**c["request"]), samples), samples
    ):
        equal(actual, c["expected"])
try:
    derive_native_ocr_boxes(
        {}, dict(poly=[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]), useful_list=[0] * 8
    )
except RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid geometry accepted")
try:
    prepare_model_ocr(dict(model=dict(layout_dets=[]), ocr=[], images={}))
except ValueError:
    pass
else:
    raise AssertionError("missing native page silently ignored")
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / base["model"],
    base["model_sha256"],
)
for name in ["native", "scan"]:
    plan = prepare_pdf_model_inference(
        root / f"tests/migration/inputs/pdf_{name}.pdf",
        layout_model=session,
        layout_shape_mode="rect",
        dpi=72.0,
    )
    result = prepare_model_ocr(plan)
    for task in result["detect"]:
        image = png(result["images"][task["source"]])
        assert image["width"] % 64 == image["height"] % 64 == 0
    assert (
        result["model"]["layout_dets"][: len(plan["model"]["layout_dets"])]
        == plan["model"]["layout_dets"]
    )
print(
    "Installed native OCR plan: 182 exact old bbox cases, 54 native/detection plans, 75 actual old detector-input RGB pixel refs; real PDF layout→native/det preparation, ownership/concurrency/errors passed; no detector inference claim"
)
