"""独立 wheel，无 Pillow/NumPy；标准 PNG 滤波解码校验旧像素，验证实际目录和 writer 写入。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy, hashlib, json, struct, sys, tempfile, zlib
from rapid_doc_rust import (
    assemble_model_page_with_assets,
    optimize_formula_numbers,
    split_paragraphs,
    render_model_pages,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    data = json.loads((root / f"build/migration/{name}.json").read_text())
    for p, h in data["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    return data


fixture = load("legacy-assets")
chain = load("legacy-assets-output")
assert len(fixture["cases"]) == 337
chains = {c["name"]: c for c in chain["cases"]}


def equal(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (path, set(a) ^ set(b))
        for k in a:
            equal(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def png_pixels(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    packed = bytearray()
    metadata = {}
    while offset < len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        value = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if kind == b"IHDR":
            width, height, depth, color, compression, filter_, interlace = struct.unpack(
                ">IIBBBBB", value
            )
        if kind == b"IDAT":
            packed.extend(value)
        if kind == b"iCCP":
            metadata["icc_profile_sha256"] = hashlib.sha256(
                zlib.decompress(value.split(b"\0", 1)[1][1:])
            ).hexdigest()
        if kind == b"eXIf":
            metadata["exif_sha256"] = hashlib.sha256(b"Exif\0\0" + value).hexdigest()
        if kind == b"tRNS":
            metadata["transparency"] = (
                struct.unpack(">H", value)[0]
                if len(value) == 2
                else list(struct.unpack(">HHH", value))
            )
    assert depth == 8 and compression == 0 and filter_ == 0 and interlace == 0
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    mode = {0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[color]
    raw = zlib.decompress(packed)
    stride = width * channels
    out = bytearray()
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        kind = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = prev[x]
            corner = prev[x - channels] if x >= channels else 0
            if kind == 1:
                add = left
            elif kind == 2:
                add = up
            elif kind == 3:
                add = (left + up) // 2
            elif kind == 4:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                add = left if pa <= pb and pa <= pc else up if pb <= pc else corner
            else:
                assert kind == 0
                add = 0
            row[x] = (row[x] + add) & 255
        out.extend(row)
        prev = row
    return dict(
        mode=mode,
        width=width,
        height=height,
        pixels_sha256=hashlib.sha256(out).hexdigest(),
        **metadata,
    )


class Sink:
    def __init__(self):
        self.calls = []

    def write(self, name, data):
        assert isinstance(data, bytes)
        self.calls.append((name, data))


def run(c, writer):
    r = copy.deepcopy(c["request"])
    source = root / c["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c["source_sha256"]
    return assemble_model_page_with_assets(
        source,
        r["model"],
        image_writer=writer,
        **{k: v for k, v in r.items() if k != "model"},
        original_images=[dict(bbox=o["bbox"], source=root / o["source"]) for o in c["originals"]],
        extract_original_image=c["extract_original"],
        extract_original_image_iou_thresh=c["threshold"],
    )


for c in fixture["cases"]:
    before = copy.deepcopy(c["request"])
    sink = Sink()
    page = run(c, sink)
    equal(page, c["expected"], c["name"])
    equal(c["request"], before, "ownership")
    assert len(sink.calls) == len(c["calls"]), c["name"]
    for (name, data), expected in zip(sink.calls, c["calls"]):
        equal(dict(name=name, **png_pixels(data)), expected, c["name"])
    pages = (
        [page]
        if page is not None
        else [
            dict(
                preproc_blocks=[],
                page_idx=c["request"]["page_index"],
                page_size=[int(c["request"]["page_width"]), int(c["request"]["page_height"])],
                discarded_blocks=[],
            )
        ]
    )
    pages = split_paragraphs(optimize_formula_numbers(pages))
    reference = chains[c["name"]]
    equal(pages, reference["pages"], c["name"])
    for mode, expected in reference["outputs"].items():
        equal(
            render_model_pages(pages, make_mode=mode, image_path_prefix="images"),
            expected,
            c["name"] + "/" + mode,
        )
with tempfile.TemporaryDirectory(prefix="rapid-doc-assets-") as tmp:
    directory = Path(tmp)
    for c in fixture["cases"][::20] + [
        next(c for c in fixture["cases"] if c["name"] == "duplicate-name-retains-write-order")
    ]:
        folder = directory / str(fixture["cases"].index(c))
        page = run(c, folder)
        equal(page, c["expected"])
        expected = {a["name"]: a for a in c["calls"]}
        assert {p.name for p in folder.iterdir()} == set(expected)
        for name, data in expected.items():
            equal(dict(name=name, **png_pixels((folder / name).read_bytes())), data)
    base = fixture["cases"][0]
    r = copy.deepcopy(base["request"])
    r["model"]["layout_dets"][0]["original_order"] = None
    sink = Sink()
    try:
        assemble_model_page_with_assets(root / base["source"], image_writer=sink, **r)
    except RapidDocError as e:
        assert e.code == "unsupported"
    else:
        raise AssertionError("missing model order accepted")
    assert not sink.calls, "computation error wrote partial assets"
    r = copy.deepcopy(base["request"])
    r["model"]["layout_dets"][0]["poly"] = [0, 0, 3000, 0, 3000, 3000, 0, 3000]
    try:
        assemble_model_page_with_assets(
            root / base["source"], image_writer=directory / "not-created", **r
        )
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("unbounded crop accepted")
    assert not (directory / "not-created").exists()
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = fixture["cases"][::60]
    for page, c in zip(pool.map(lambda c: run(c, Sink()), samples), samples):
        equal(page, c["expected"])


class Failing:
    def write(self, name, data):
        raise OSError("writer failure")


try:
    run(fixture["cases"][0], Failing())
except OSError as e:
    assert "writer failure" in str(e)
else:
    raise AssertionError("writer failure swallowed")
try:
    assemble_model_page_with_assets(
        b"not PNG", {}, image_writer=None, page_width=10, page_height=10
    )
except TypeError:
    pass
else:
    raise AssertionError("invalid writer accepted")
metadata_cases = load("legacy-assets-metadata")["cases"]
assert len(metadata_cases) == 11
for c in metadata_cases:
    sink = Sink()
    equal(run(c, sink), c["expected"], c["name"])
    assert len(sink.calls) == len(c["calls"]), c["name"]
    for (name, data), expected in zip(sink.calls, c["calls"]):
        equal(dict(name=name, **png_pixels(data)), expected, c["name"])
assert capabilities()["legacy_page_asset_writer"] is True
assert capabilities()["compatibility"]["python_parser"] is False
print(
    "Installed wheel: 337 complete old asset pages/writer calls plus 11 ICC/EXIF/transparent-color references; names/order/L-LA-RGB-RGBA pixels, negative/fractional/outside crops, original last-match, duplicate-name overwrite, unassigned writes, actual directories/writers, 1011 old outputs, ownership/concurrency and errors/budgets passed"
)
