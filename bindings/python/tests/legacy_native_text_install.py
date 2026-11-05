"""独立 wheel：旧 native 字符填充/NumPy 裁图/真实 CPU 后置 OCR，全量 JSON 和像素。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from importlib.resources import files
from rapid_doc_rust import (
    fill_model_text_spans,
    assemble_model_page_with_native_text,
    post_process_model_ocr,
    optimize_formula_numbers,
    split_paragraphs,
    render_model_pages,
    OnnxSession,
    RapidDocError,
    read_native_pdf_page,
    render_pdf_page,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text())
    for group in ["sources", "inputs"]:
        for p, h in f.get(group, {}).items():
            assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return f


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (set(a), set(b))
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b), (len(a), len(b))
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


def rgb(data):
    assert data.startswith(b"\x89PNG")
    offset = 8
    packed = bytearray()
    chunks = []
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        value = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        chunks.append(kind)
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
    return width, height, hashlib.sha256(out).hexdigest(), chunks


def images(actual, expected):
    assert list(actual) == [e["name"] for e in expected]
    for e in expected:
        w, h, sha, chunks = rgb(actual[e["name"]])
        equal(dict(name=e["name"], width=w, height=h, rgb_sha256=sha), e)
        assert not any(c in chunks for c in [b"iCCP", b"tRNS", b"eXIf"]), (
            "raw np_img pixels acquired color/transparency metadata"
        )


class Writer:
    _parent_dir = "images"

    def __init__(self):
        self.calls = []

    def write(self, name, data):
        w, h, sha, _ = rgb(data)
        self.calls.append(dict(name=name, mode="RGB", width=w, height=h, pixels_sha256=sha))


def native_source(c):
    r = copy.deepcopy(c["request"])
    if not c["name"].startswith("real-pdf/"):
        return root / c["source"], r
    _, name, index = c["name"].split("/")
    index = int(index)
    pdf = root / f"tests/migration/inputs/pdf_{name}.pdf"
    r["native_page"] = read_native_pdf_page(pdf, page_index=index)
    png = render_pdf_page(pdf, page_index=index, dpi=72.0, color_mode="RGB").png
    equal(rgb(png)[:3], rgb((root / c["source"]).read_bytes())[:3])
    return png, r


all_cases = []
for name, count in [
    ("legacy-native-text", 95),
    ("legacy-native-text-narrow", 95),
    ("legacy-native-text-metadata", 5),
    ("legacy-native-text-numeric", 3),
]:
    f = load(name)
    assert len(f["cases"]) == count
    all_cases.extend(f["cases"])
    for c in f["cases"]:
        before = copy.deepcopy(c["request"])
        actual = fill_model_text_spans(root / c["source"], **c["request"])
        equal(actual["spans"], c["expected"])
        images(actual["ocr_images"], c["crops"])
        equal(c["request"], before)
    for c in f["pages"]:
        writer = Writer()
        before = copy.deepcopy(c["request"])
        source, request = native_source(c)
        actual = assemble_model_page_with_native_text(source, image_writer=writer, **request)
        equal(actual["page"], c["expected"])
        images(actual["ocr_images"], c["crops"])
        equal(writer.calls, c["assets"])
        equal(c["request"], before)
chain = load("legacy-native-text-chain")
assert len(chain["cases"]) == 15
assert hashlib.sha256((root / chain["model"]).read_bytes()).hexdigest() == chain["model_sha256"]
session = OnnxSession(
    root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll",
    root / chain["model"],
    chain["model_sha256"],
)
for c in chain["cases"]:
    source, request = native_source(c)
    writer = Writer()
    result = assemble_model_page_with_native_text(source, image_writer=writer, **request)
    equal(result["page"], c["before_ocr"])
    images(result["ocr_images"], c["crops"])
    equal(writer.calls, c["assets"])
    r = c["request"]
    pages = (
        [result["page"]]
        if result["page"] is not None
        else [
            dict(
                preproc_blocks=[],
                discarded_blocks=[],
                page_idx=7,
                page_size=[int(r["page_width"]), int(r["page_height"])],
            )
        ]
    )
    pages = post_process_model_ocr(pages, recognizer=session, images=result["ocr_images"])
    equal(pages, c["after_ocr"])
    pages = split_paragraphs(optimize_formula_numbers(pages))
    equal(pages, c["paragraphs"])
    for mode, expected in c["outputs"].items():
        equal(render_model_pages(pages, make_mode=mode, image_path_prefix="images"), expected)
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = all_cases[::31]
    for actual, c in zip(
        pool.map(lambda c: fill_model_text_spans(root / c["source"], **c["request"]), samples),
        samples,
    ):
        equal(actual["spans"], c["expected"])
bad = copy.deepcopy(chain["cases"][0]["request"])
for d in bad["model"]["layout_dets"]:
    d["original_order"] = None
before = copy.deepcopy(bad)
writer = Writer()
try:
    assemble_model_page_with_native_text(
        root / chain["cases"][0]["source"], image_writer=writer, **bad
    )
except RapidDocError:
    pass
else:
    raise AssertionError("missing model order accepted")
assert not writer.calls
equal(bad, before)
sample = all_cases[0]
try:
    fill_model_text_spans(
        root / "build/migration/legacy-asset-images/RGBA-48-40-0.png", **sample["request"]
    )
except RapidDocError as e:
    assert e.code == "unsupported"
else:
    raise AssertionError("non-RGB native page silently converted")
assert "NumPy Developers" in files("rapid_doc_rust").joinpath(
    "licenses/NUMPY-PAIRWISE.txt"
).read_text(encoding="utf-8")
print(
    "Installed wheel: 198 exact old native span fills including signed-zero ownership, 32 complete pages including 5 real PDF fact pages, 295 contrast unit references; native crop RGB bytes and metadata isolation; 15 real native→CPU OCR→paragraph chains/45 outputs, ordinary assets, ownership/concurrency/errors passed"
)
