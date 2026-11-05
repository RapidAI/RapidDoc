"""独立wheel：旧公式/图片提示、像素涂白及真实PDF/CPU到结构模型输入。"""

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
    prepare_table_hints,
    prepare_table_model_payload_png,
    recognize_table_text_stage_png,
    read_native_pdf_page,
    OnnxSession,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/table-hints.json").read_text(encoding="utf-8"))
cpu = json.loads((root / "build/migration/table-hints-cpu.json").read_text(encoding="utf-8"))


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


for fixture in [f, cpu]:
    for p, h in fixture["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
for c in f["plans"]:
    before = copy.deepcopy(c["request"])
    same(prepare_table_hints(**c["request"]), c["expected"])
    same(c["request"], before)


def request(c):
    r = copy.deepcopy(c["request"])
    r["inline_delimiters"] = (r.pop("inline_left"), r.pop("inline_right"))
    return r


for c in f["payloads"]:
    source = (root / c["source"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == c["source_sha256"]
    req = request(c)
    before = copy.deepcopy(req)
    r = prepare_table_model_payload_png(source, **req)
    image = r.pop("png")
    same(r, c["expected"])
    same(png(image), c["image"])
    same(req, before)
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
det = OnnxSession(runtime, root / cpu["det_model"], cpu["det_sha256"], threads=1)
rec = OnnxSession(runtime, root / cpu["rec_model"], cpu["rec_sha256"], threads=1)
ori = OnnxSession(runtime, root / cpu["orientation_model"], cpu["orientation_sha256"], threads=1)


def run(c):
    assert hashlib.sha256((root / c["source"]).read_bytes()).hexdigest() == c["source_sha256"]
    assert (
        hashlib.sha256((root / c["pdf"]["source"]).read_bytes()).hexdigest() == c["pdf"]["sha256"]
    )
    req = copy.deepcopy(c["request"])
    info = c["pdf"]
    facts = read_native_pdf_page(
        root / info["source"], page_index=info["page_index"], quote_loosebox=info["quote_loosebox"]
    )
    for key in facts:
        same(facts[key], req["page"][key])
    before = copy.deepcopy(req)
    force = req.pop("force_ocr")
    hints = prepare_table_hints(**req)
    text = recognize_table_text_stage_png(
        root / c["source"],
        native_page=hints["page"],
        table=hints["table"],
        useful_list=req["useful_list"],
        scale=req["scale"],
        force_ocr=force,
        detector=det,
        recognizer=rec,
        orientation=ori,
        masks=hints["formulas"],
    )
    r = prepare_table_model_payload_png(
        text["png"],
        text["ocr_result"],
        fill_images=hints["fill_images"],
        formulas=hints["formulas"],
    )
    assert not r["needs_ocr"]
    same(png(r["png"]), c["expected"]["image"])
    same(r["ocr_result"][:2], c["expected"]["ocr_result"][:2])
    error = 0.0
    assert len(r["ocr_result"][2]) == len(c["expected"]["ocr_result"][2])
    for a, b in zip(r["ocr_result"][2], c["expected"]["ocr_result"][2]):
        assert type(a) is type(b) and abs(a - b) <= 1e-5, (a, b)
        error = max(error, abs(a - b))
    same(c["request"], before)
    return error


max_error = max(run(c) for c in cpu["cases"])
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(run, cpu["cases"]))
c = next(
    c
    for c in f["payloads"]
    if c["request"]["ocr_result"] and c["request"]["ocr_result"][0] and c["request"]["fill_images"]
)


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


source = (root / c["source"]).read_bytes()
stream = Once(source)
r = prepare_table_model_payload_png(stream, **request(c))
same(png(r["png"]), c["image"])
assert stream.count == 1 and not stream.closed
chunk = struct.pack(">I", 6) + b"tRNS" + struct.pack(">HHH", 0, 0, 0)
chunk += struct.pack(">I", zlib.crc32(chunk[4:]) & 0xFFFFFFFF)
tagged = source[:33] + chunk + source[33:]
same(png(prepare_table_model_payload_png(tagged, **request(c))["png"]), c["image"])


def fails(fn):
    try:
        fn()
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid hint/payload accepted")


bad = copy.deepcopy(f["plans"][0]["request"])
bad["scale"] = 0
fails(lambda: prepare_table_hints(**bad))
bad = request(c)
bad["ocr_result"][2] = []
fails(lambda: prepare_table_model_payload_png(source, **bad))
bad = request(c)
bad["skip_text_in_image"] = "yes"
fails(lambda: prepare_table_model_payload_png(source, **bad))
bad = request(c)
bad["inline_delimiters"] = ("x",)
fails(lambda: prepare_table_model_payload_png(source, **bad))
assert (
    capabilities()["legacy_table_hints"]
    and capabilities()["legacy_table_model_payload"]
    and not capabilities()["legacy_table_execution"]
)
print(
    f"Installed wheel: 241 old formula/image hints and 343 actual RapidTableModel payload/pixel cases; 4 actual PDF→CPU→hint payloads, exact boxes/text/RGB, max score error {max_error:.9g}; tRNS, ownership, concurrency and errors passed; structure network not executed"
)
