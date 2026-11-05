"""独立 wheel 对照实际旧 HTML/原字节和完整单页 writer，不依赖 Pillow。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys, tempfile
from pathlib import Path
from rapid_doc_rust import (
    replace_inline_base64_img_src,
    assemble_model_page_with_assets,
    RapidDocError,
    optimize_formula_numbers,
    split_paragraphs,
    render_model_pages,
)

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/legacy-inline-assets.json").read_text())
chains = json.loads((root / "build/migration/legacy-inline-output.json").read_text())
assert (
    hashlib.sha256((root / "build/migration/legacy-inline-assets.json").read_bytes()).hexdigest()
    == chains["source_sha256"]
)
for path, h in chains["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == h
for path, h in fixture["sources"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == h
assert (
    hashlib.sha256((root / fixture["source"]).read_bytes()).hexdigest() == fixture["source_sha256"]
)


class Slots:
    __slots__ = ("calls",)

    def __init__(self):
        self.calls = []

    def write(self, name, data):
        self.calls.append(dict(name=name, bytes_hex=data.hex()))


class Sink:
    def __init__(self):
        self.calls = []

    def write(self, name, data):
        self.calls.append((name, data))


for c in fixture["cases"]:
    writer = Slots()
    assert replace_inline_base64_img_src(c["markup"], writer, 7, c["prefix"]) == c["expected"], c[
        "markup"
    ]
    assert writer.calls == c["calls"], c["markup"]


def png_pixels(data):
    import struct, zlib

    offset = 8
    packed = bytearray()
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        value = data[offset + 8 : offset + 8 + size]
        offset += size + 12
        if kind == b"IHDR":
            width, height, depth, color, _, _, _ = struct.unpack(">IIBBBBB", value)
        if kind == b"IDAT":
            packed.extend(value)
    assert depth == 8
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    stride = width * channels
    raw = zlib.decompress(packed)
    pixels = bytearray()
    previous = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        method = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        for i in range(stride):
            a = row[i - channels] if i >= channels else 0
            b = previous[i]
            c = previous[i - channels] if i >= channels else 0
            if method == 1:
                row[i] = (row[i] + a) & 255
            elif method == 2:
                row[i] = (row[i] + b) & 255
            elif method == 3:
                row[i] = (row[i] + ((a + b) // 2)) & 255
            elif method == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                row[i] = (row[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
            else:
                assert method == 0
        pixels.extend(row)
        previous = row
    return dict(
        mode={0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[color],
        width=width,
        height=height,
        pixels_sha256=hashlib.sha256(pixels).hexdigest(),
    )


for c, chain in zip(fixture["pages"], chains["cases"], strict=True):
    writer = Sink()
    before = copy.deepcopy(c["request"])
    page = assemble_model_page_with_assets(
        root / fixture["source"], image_writer=writer, **c["request"]
    )
    assert page == c["expected"], c["request"]
    assert c["request"] == before
    actual = [
        dict(
            name=name,
            **(
                png_pixels(data)
                if data.startswith(b"\x89PNG")
                else dict(sha256=hashlib.sha256(data).hexdigest())
            ),
        )
        for name, data in writer.calls
    ]
    assert actual == c["calls"], (actual, c["calls"])
    pages = (
        [page]
        if page is not None
        else [
            dict(
                preproc_blocks=[],
                page_idx=c["request"]["page_index"],
                page_size=[48, 40],
                discarded_blocks=[],
            )
        ]
    )
    pages = split_paragraphs(optimize_formula_numbers(pages))
    assert pages == chain["pages"]
    for mode, expected in chain["outputs"].items():
        assert render_model_pages(pages, make_mode=mode, image_path_prefix="images") == expected

markup = '<img src="data:image/jpeg;base64,YWJjZA=="><img src="data:image/jpeg;base64,YWJjZA==">'
writer = Sink()
first = replace_inline_base64_img_src(markup, writer)
assert len(writer.calls) == 1
assert replace_inline_base64_img_src(markup, writer) == first and len(writer.calls) == 1
slots = Slots()
replace_inline_base64_img_src(markup, slots)
replace_inline_base64_img_src(markup, slots)
assert len(slots.calls) == 4


class Failing(Sink):
    def write(self, name, data):
        if not self.calls:
            self.calls.append(None)
            raise OSError("write failed")
        super().write(name, data)


writer = Failing()
try:
    replace_inline_base64_img_src(markup, writer)
except OSError:
    pass
else:
    raise AssertionError("writer error swallowed")
assert not writer._mineru_written_image_paths
replace_inline_base64_img_src(markup, writer)
assert len(writer.calls) == 2 and len(writer._mineru_written_image_paths) == 1
with tempfile.TemporaryDirectory() as folder:
    result = replace_inline_base64_img_src(markup, Path(folder), image_path_prefix="files/")
    files = list(Path(folder).iterdir())
    assert len(files) == 1 and files[0].suffix == ".jpg" and files[0].read_bytes() == b"abcd"
    assert f"files/{files[0].name}" in result
for markup in [
    '<img src="data:image/wmf;base64,YQ==">',
    '<img src="data:image/../png;base64,YQ==">',
    "x" * 2_000_001,
]:
    writer = Sink()
    try:
        replace_inline_base64_img_src(markup, writer)
    except RapidDocError:
        pass
    else:
        # No base64 marker is a deliberate old fast path, before native limits.
        assert "base64," not in markup
    assert not writer.calls
bad = copy.deepcopy(fixture["pages"][6]["request"])
bad["model"]["layout_dets"][0]["original_order"] = None
writer = Sink()
try:
    assemble_model_page_with_assets(root / fixture["source"], image_writer=writer, **bad)
except RapidDocError:
    pass
else:
    raise AssertionError("model order not required")
assert not writer.calls
for model in ["wrong type", None, dict(layout_dets="wrong type")]:
    writer = Sink()
    try:
        assemble_model_page_with_assets(
            root / fixture["source"], model, image_writer=writer, page_width=48, page_height=40
        )
    except RapidDocError as error:
        assert error.code == "invalid_input"
    else:
        raise AssertionError("malformed model accepted")
    assert not writer.calls
bad = copy.deepcopy(fixture["pages"][6]["request"])
det = bad["model"]["layout_dets"][0]
det["html"] = '<img src="data:image/png;base64,' + "A" * 1_100_000 + '">'
bad["model"]["layout_dets"].append(copy.deepcopy(det))
writer = Sink()
try:
    assemble_model_page_with_assets(root / fixture["source"], image_writer=writer, **bad)
except RapidDocError as error:
    assert error.code == "resource_limit"
else:
    raise AssertionError("original request budget bypassed by replacement")
assert not writer.calls
print(
    f"Installed wheel: {len(fixture['cases'])} exact old HTML/base64/raw-byte references; {len(fixture['pages'])} complete pages/crop ordering and 36 old outputs; writer lifetime/slots/retry, real directories, ownership and failure budgets passed"
)
