"""独立 wheel：实际旧后置 OCR 回填与真实 CPU 批识别/段落输出对照。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, math, struct, sys, zlib
from array import array
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    OnnxSession,
    apply_model_ocr_results,
    post_process_model_ocr,
    optimize_formula_numbers,
    split_paragraphs,
    render_model_pages,
    RapidDocError,
)
from rapid_doc_rust._native import legacy_pipeline_json, prepare_recognition_batch_bgr_json

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text())
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return f


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


fixture = load("legacy-post-ocr")
assert len(fixture["cases"]) == 73
for c in fixture["cases"]:
    before = copy.deepcopy(c["pages"])
    plan = json.loads(legacy_pipeline_json("plan_post_ocr", json.dumps(c["pages"])))
    equal([t["image"] for t in plan], c["targets"])
    equal(apply_model_ocr_results(c["pages"], c["results"]), c["expected"])
    equal(c["pages"], before)


def bgr(path):
    data = (root / path).read_bytes()
    offset = 8
    packed = bytearray()
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        value = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if kind == b"IHDR":
            width, height, depth, color, compression, filter_, interlace = struct.unpack(
                ">IIBBBBB", value
            )
        if kind == b"IDAT":
            packed.extend(value)
    assert (depth, color, compression, filter_, interlace) == (8, 2, 0, 0, 0)
    raw = zlib.decompress(packed)
    stride = width * 3
    out = bytearray()
    previous = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        method = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        for i in range(stride):
            a = row[i - 3] if i >= 3 else 0
            b = previous[i]
            c = previous[i - 3] if i >= 3 else 0
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
        previous = row
    for i in range(0, len(out), 3):
        out[i], out[i + 2] = out[i + 2], out[i]
    return bytes(out), width, height


batch = load("recognition-batch")
assert len(batch["cases"]) == 12
for p, h in batch["inputs"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
assert hashlib.sha256((root / batch["model"]).read_bytes()).hexdigest() == batch["model_sha256"]
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / batch["model"],
    batch["model_sha256"],
)
for c in batch["cases"]:
    # Frozen 'unsupported' records a historical limitation; all cases now use
    # the original native sorting backend and retain the original tensors.
    tensors = json.loads(
        prepare_recognition_batch_bgr_json([bgr(p) for p in c["sources"]], c["batch_size"])
    )
    seen = set()
    for tensor in tensors:
        count = tensor["shape"][0]
        stride = len(tensor["values"]) // count
        for row, index in enumerate(tensor["indices"]):
            assert tensor["shape"][1:] == c["tensors"][index]["shape"]
            assert (
                hashlib.sha256(
                    array("f", tensor["values"][row * stride : (row + 1) * stride]).tobytes()
                ).hexdigest()
                == c["tensors"][index]["sha256"]
            ), c["name"]
            seen.add(index)
    assert len(seen) == len(c["sources"])
    actual = session.recognize_batch_png(
        [root / p for p in c["sources"]], batch_size=c["batch_size"]
    )
    assert len(actual) == len(c["results"])
    for got, (text, score) in zip(actual, c["results"]):
        assert got["text"] == text and math.isclose(
            got["confidence"], score, abs_tol=1e-5, rel_tol=0
        ), (c["name"], got, (text, score))

chain = load("legacy-post-ocr-chain")
assert len(chain["cases"]) == 11
assert (
    hashlib.sha256((root / "build/migration/recognition-batch.json").read_bytes()).hexdigest()
    == chain["batch_sha256"]
)
for c in chain["cases"]:
    before = copy.deepcopy(c["pages"])
    actual = post_process_model_ocr(
        c["pages"],
        recognizer=session,
        images={k: root / p for k, p in c["images"].items()},
        batch_size=c["batch_size"],
    )
    equal(actual, c["expected"])
    equal(c["pages"], before)
    paragraphs = split_paragraphs(optimize_formula_numbers(actual))
    equal(paragraphs, c["paragraphs"])
    for mode, expected in c["outputs"].items():
        equal(render_model_pages(paragraphs, make_mode=mode, image_path_prefix="images"), expected)
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = fixture["cases"][::11]
    for actual, c in zip(
        pool.map(lambda c: apply_model_ocr_results(c["pages"], c["results"]), samples), samples
    ):
        equal(actual, c["expected"])
empty = [dict(preproc_blocks=[], discarded_blocks=[])]
equal(post_process_model_ocr(empty, recognizer=None, images=None), empty)
sample = chain["cases"][0]
before = copy.deepcopy(sample["pages"])
for results in [[], [["only-one", 0.9]], [[12, 0.9]]]:
    try:
        apply_model_ocr_results(sample["pages"], results)
    except RapidDocError:
        pass
    else:
        raise AssertionError("invalid results accepted")
    equal(sample["pages"], before)
stream = io.BytesIO((root / batch["cases"][0]["sources"][0]).read_bytes())
assert len(session.recognize_batch_png([stream], batch_size=1)) == 1 and not stream.closed
for size in [0, 65, True, "6"]:
    try:
        session.recognize_batch_png([], batch_size=size)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid batch size accepted")
assert session.recognize_batch_png([]) == []
path = root / batch["cases"][0]["sources"][0]
stream = io.BytesIO(path.read_bytes())
duplicates = [
    dict(
        preproc_blocks=[
            dict(type="text", lines=[dict(spans=[dict(np_img="same"), dict(np_img="same")])])
        ],
        discarded_blocks=[],
    )
]
raw = session.recognize_batch_png([path, path])
equal(
    post_process_model_ocr(duplicates, recognizer=session, images={"same": stream}),
    apply_model_ocr_results(duplicates, [(r["text"], r["confidence"]) for r in raw]),
)
assert (
    not stream.closed
    and duplicates[0]["preproc_blocks"][0]["lines"][0]["spans"][0]["np_img"] == "same"
)
print(
    "Installed wheel: 73 exact old post-OCR selection/rounding cases; 12 old CPU batch references (including formerly rejected cross-batch ties), per-image shared-padding tensors bit-exact; 11 real CPU→post OCR→paragraph chains and 33 outputs; order/ownership/errors/concurrency passed"
)
