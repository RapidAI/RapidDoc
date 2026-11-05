"""Installed wheel, no Pillow: exact decoded PNG pixels/metadata and page fallback transactions."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import base64, copy, hashlib, json, re, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import (
    materialize_native_table_media_page as page,
    materialize_native_table_media as pdf,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-table-media-png.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def pixels(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    packed = bytearray()
    metadata = {}
    while pos < len(data):
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        value = data[pos + 8 : pos + 8 + size]
        pos += size + 12
        if kind == b"IHDR":
            w, h, depth, color, compression, filter_, interlace = struct.unpack(">IIBBBBB", value)
        if kind == b"IDAT":
            packed.extend(value)
        if kind == b"iCCP":
            metadata["icc_profile_sha256"] = hashlib.sha256(
                zlib.decompress(value.split(b"\0", 1)[1][1:])
            ).hexdigest()
        if kind == b"tRNS":
            metadata["transparency"] = (
                struct.unpack(">H", value)[0]
                if len(value) == 2
                else list(struct.unpack(">HHH", value))
            )
        if kind == b"eXIf":
            metadata["exif_sha256"] = hashlib.sha256(b"Exif\0\0" + value).hexdigest()
    assert depth == 8 and compression == 0 and filter_ == 0 and interlace == 0
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(packed)
    stride = w * channels
    previous = bytearray(stride)
    out = bytearray()
    for y in range(h):
        start = y * (stride + 1)
        kind = raw[start]
        row = bytearray(raw[start + 1 : start + stride + 1])
        if kind == 0:
            out.extend(row)
            previous = row
            continue
        if kind == 2:
            row = (
                previous.copy()
                if not any(row)
                else bytearray((a + b) & 255 for a, b in zip(row, previous))
            )
            out.extend(row)
            previous = row
            continue
        for x in range(stride):
            left = row[x - channels] if x >= channels else 0
            up = previous[x]
            corner = previous[x - channels] if x >= channels else 0
            if kind == 0:
                add = 0
            elif kind == 1:
                add = left
            elif kind == 2:
                add = up
            elif kind == 3:
                add = (left + up) // 2
            else:
                assert kind == 4
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                add = left if pa <= pb and pa <= pc else up if pb <= pc else corner
            row[x] = (row[x] + add) & 255
        out.extend(row)
        previous = row
    return dict(
        mode={0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[color],
        width=w,
        height=h,
        pixels_sha256=hashlib.sha256(out).hexdigest(),
        **metadata,
    )


def project(pages):
    pages = copy.deepcopy(pages)
    images = []

    def replace(m):
        images.append(pixels(base64.b64decode(m[1])))
        return f"@PNG{len(images) - 1}@"

    for p in pages:
        for block in p:
            if isinstance(block.get("content"), str):
                block["content"] = re.sub(
                    r"data:image/png;base64,([A-Za-z0-9+/=]+)", replace, block["content"]
                )
    return dict(pages=pages, images=images)


def exact(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            exact(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    source = root / c["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c["source_sha256"]
    blocks = copy.deepcopy(c["blocks"])
    before = copy.deepcopy(blocks)
    result = page(blocks, source.read_bytes(), page_size=c["page_size"])
    exact(project([result]), c["expected"], c["label"])
    exact(blocks, before, "ownership")


assert len(f["cases"]) == 182 and len(f["pdf"]) == 45
assert sum(len(c["expected"]["images"]) for c in f["cases"]) == 502
assert sum(len(c["expected"]["images"]) for c in f["pdf"]) == 125
for c in f["cases"]:
    run(c)
print("182 PNG page facts passed", flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::17]))
for case_index, c in enumerate(f["pdf"]):
    source = root / c["pdf"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == c["pdf_sha256"]
    pages = copy.deepcopy(c["pages"])
    before = copy.deepcopy(pages)
    result = pdf(pages, source, page_sizes=c["page_sizes"])
    exact(project(result), c["expected"], c["label"])
    exact(pages, before, "PDF ownership")
    if (case_index + 1) % 12 == 0:
        print("Real PDF cases passed:", case_index + 1, flush=True)


class Unreadable:
    def read(self, *a):
        raise AssertionError("no-media PDF source was read")


empty = [[dict(type="text", content="x"), dict(type="table", content="empty", _table_images=[])]]
exact(pdf(empty, Unreadable()), empty, "lazy PDF")
exact(page(empty[0], b"not PNG"), empty[0], "lazy PNG")
block = dict(
    type="table",
    content='<img src="docvortex-table-image:0" alt="Image" />',
    _table_images=[dict(bbox=[0.0, 0.0, 1.0, 1.0], text="<&>\"'")],
)
fallback = page([block], None)
assert fallback[0]["content"] == "&lt;&amp;&gt;&quot;&#x27;" and "_table_images" not in fallback[0]
for options in [dict(max_pixels=10), dict(max_pixels=True), dict(max_pixels=0)]:
    try:
        pdf([[block]], root / f["pdf"][0]["pdf"], **options)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    except ValueError:
        assert options["max_pixels"] in (True, 0)
    else:
        raise AssertionError("pixel limit hidden as text fallback")
try:
    page([dict(block, _table_images=[dict(bbox=[0.0, 0.0, 1.0, 1.0], text="x")] * 5001)], None)
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("media limit ignored")
run(f["cases"][0])
assert (
    capabilities()["native_table_media_png"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (
    Path(__import__("rapid_doc_rust").__file__).parent / "licenses/MIGRATION-NOTICES.txt"
).is_file()
assert "PIL" not in sys.modules and "rapid_doc" not in sys.modules
print(
    "Installed wheel: 182 PNG page facts/502 decoded images, 45 real PDF/angle/commit chains/125 images; exact modes/dimensions/pixels/ICC/transparency, HTML and partial-page text fallback, point ticks, no-media laziness, ownership/concurrency/limits; PNG compression/data URI bytes and URI-derived names may differ"
)
