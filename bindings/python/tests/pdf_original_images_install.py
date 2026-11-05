"""独立wheel：旧提图兼容API参考、原始像素、完整资产页面和生命周期。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, sys, zlib
from importlib.resources import files
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import extract_original_pdf_images, assemble_pdf_model_page, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads(
    (root / "build/migration/pdf-original-images.json").read_text(encoding="utf-8")
)


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
    assert data.startswith(b"\x89PNG")
    offset = 8
    packed = bytearray()
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        value = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if kind == b"IHDR":
            w, h, depth, color, compression, filter_, interlace = struct.unpack(">IIBBBBB", value)
        if kind == b"IDAT":
            packed.extend(value)
    assert depth == 8 and color in (0, 2, 4, 6) and (compression, filter_, interlace) == (0, 0, 0)
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    mode = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[color]
    raw = zlib.decompress(packed)
    stride = w * channels
    out = bytearray()
    previous = bytearray(stride)
    for y in range(h):
        start = y * (stride + 1)
        method = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        for i in range(stride):
            a = row[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
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
    return dict(mode=mode, width=w, height=h, pixels_sha256=hashlib.sha256(out).hexdigest())


class Writer:
    _parent_dir = "images"

    def __init__(self):
        self.calls = []

    def write(self, name, data):
        self.calls.append(dict(name=name, **png(data)))


for p, h in fixture["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
assert len(fixture["cases"]) == 102
for case in fixture["cases"]:
    path = root / case["source"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
    output = extract_original_pdf_images(
        path, **{k: case[k] for k in ["page_index", "max_depth", "render", "scale_to_original"]}
    )
    assert not output["diagnostics"], output["diagnostics"]
    actual = []
    for image in output["images"]:
        decoded = png(image["source"])
        equal(
            decoded,
            {k: image[k] for k in ["mode", "width", "height"]}
            | dict(pixels_sha256=decoded["pixels_sha256"]),
        )
        actual.append(dict(uuid=image["uuid"], bbox=image["bbox"], **decoded))
    expected = [{k: v for k, v in img.items() if k != "raw_uuid"} for img in case["expected"]]
    equal(actual, expected, path.as_posix())
assert len(fixture["pages"]) == 17
for case in fixture["pages"]:
    for native_text in [False, True]:
        writer = Writer()
        before = copy.deepcopy(case["model"])
        output = assemble_pdf_model_page(
            root / case["source"],
            case["model"],
            image_writer=writer,
            page_index=case["page_index"],
            dpi=72.0,
            native_text=native_text,
            extract_original_image=True,
        )
        equal(output["page"], case["expected"])
        equal(writer.calls, case["calls"])
        equal(case["model"], before)
        assert not output["ocr_images"] and not output["diagnostics"]
source = root / fixture["cases"][0]["source"]
data = source.read_bytes()


def summarize(output):
    return [
        {k: (hashlib.sha256(v).hexdigest() if k == "source" else v) for k, v in image.items()}
        for image in output["images"]
    ]


expected = summarize(extract_original_pdf_images(data))
stream = BytesIO(data)
equal(summarize(extract_original_pdf_images(stream)), expected)
assert not stream.closed
with ThreadPoolExecutor(max_workers=4) as pool:
    for output in pool.map(lambda _: extract_original_pdf_images(data), range(12)):
        equal(summarize(output), expected)
for value, options, code in [
    (b"", {}, "invalid_input"),
    (b"bad", {}, "corrupt_document"),
    (data, {"page_index": 1000000}, "invalid_input"),
    (data, {"password": "a\0b"}, "invalid_input"),
    (data, {"max_depth": 33}, "invalid_input"),
    (data, {"max_pixels": 0}, "invalid_input"),
    (data, {"max_pixels": 1}, "resource_limit"),
]:
    try:
        extract_original_pdf_images(value, **options)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(code)
writer = Writer()
try:
    assemble_pdf_model_page(
        data,
        fixture["pages"][0]["model"],
        image_writer=writer,
        max_pixels=1,
        extract_original_image=True,
    )
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("budget accepted")
assert not writer.calls
equal(summarize(extract_original_pdf_images(data)), expected)
edges = json.loads(
    (root / "build/migration/pdf-original-image-edges/manifest.json").read_text(encoding="utf-8")
)
for case in edges["cases"]:
    source = root / case["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case["sha256"]
    if case["code"]:
        try:
            extract_original_pdf_images(source, render=case["render"])
        except RapidDocError as e:
            assert e.code == case["code"], e.code
        else:
            raise AssertionError(case["name"])
    else:
        output = extract_original_pdf_images(source, render=case["render"])
        assert output["images"] == [] and len(output["diagnostics"]) == 1
        assert output["diagnostics"][0]["code"] == "original_image_skipped"
assert "geisserml" in files("rapid_doc_rust").joinpath("licenses/MIGRATION-NOTICES.txt").read_text()
print(
    "Installed original PDF images: 102 get_pos-adapted old cases/432 image pixel references, 17 complete asset pages; deterministic UUID projection, streams, concurrency, budgets, no writes on failure passed"
)
