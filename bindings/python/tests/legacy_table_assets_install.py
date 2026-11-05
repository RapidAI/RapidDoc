"""真实独立 wheel：表内原图像素/元数据/HTML/顺序，文件名差异明确投影。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, struct, subprocess, sys, tempfile, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import (
    assemble_model_page_with_assets,
    optimize_formula_numbers,
    split_paragraphs,
    render_model_pages,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/legacy-table-assets.json").read_text())
for group in ["sources", "inputs"]:
    for p, h in fixture[group].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h


def pixels(data):
    if not data.startswith(b"\x89PNG"):
        return dict(bytes_sha256=hashlib.sha256(data).hexdigest())
    offset = 8
    packed = bytearray()
    metadata = {}
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
    assert (depth, compression, filter_, interlace) == (8, 0, 0, 0)
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color]
    raw = zlib.decompress(packed)
    stride = width * channels
    out = bytearray()
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
    return dict(
        mode={0: "L", 2: "RGB", 4: "LA", 6: "RGBA"}[color],
        width=width,
        height=height,
        pixels_sha256=hashlib.sha256(out).hexdigest(),
        **metadata,
    )


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert set(a) == set(b), (a, b)
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b), (a, b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b, (a, b)


class Sink:
    def __init__(self, parent="images"):
        self._parent_dir = parent
        self.calls = []

    def write(self, name, data):
        self.calls.append((name, data))


def run(c, writer=None):
    if writer is None:
        writer = Sink(c["parent"])
    page = assemble_model_page_with_assets(
        root / fixture["source"],
        image_writer=writer,
        table_fill_images=[
            dict(uuid=f["uuid"], bbox=f["bbox"], source=root / f["source"]) for f in c["fills"]
        ],
        **c["request"],
    )
    return page, writer


assert len(fixture["cases"]) == 57
for c in fixture["cases"]:
    before = copy.deepcopy(c)
    page, sink = run(c)
    equal(page, c["expected_with_stable_names"])
    equal(c, before)
    calls = [dict(name=name, **pixels(data)) for name, data in sink.calls]
    equal(calls, c["calls"])
    for legacy, stable in c["renames"].items():
        assert legacy != stable
    for legacy in c["legacy_calls"]:
        if "stable_seed" in legacy:
            assert (
                hashlib.sha256(legacy["stable_seed"].encode()).hexdigest() + ".png"
                == c["renames"][legacy["name"]]
            )
    pages = (
        [page]
        if page is not None
        else [
            dict(
                preproc_blocks=[],
                page_idx=7,
                page_size=[int(c["request"]["page_width"]), int(c["request"]["page_height"])],
                discarded_blocks=[],
            )
        ]
    )
    pages = split_paragraphs(optimize_formula_numbers(pages))
    equal(pages, c["pages_with_stable_names"])
    for mode, expected in c["outputs_with_stable_names"].items():
        equal(render_model_pages(pages, make_mode=mode, image_path_prefix="images"), expected)
    # 跨调用保留 UUID writer 重复写出，不能套用内联 data URI 的去重。
    if c["name"] == "multiple-and-repeated":
        old = len(sink.calls)
        run(c, sink)
        assert len(sink.calls) == old * 2
with ThreadPoolExecutor(max_workers=3) as pool:
    samples = fixture["cases"][::10]
    for (page, sink), c in zip(pool.map(run, samples), samples):
        equal(page, c["expected_with_stable_names"])
sample = fixture["cases"][0]
with tempfile.TemporaryDirectory(prefix="rapid-doc-table-images-") as directory:
    folder = Path(directory) / "images"
    page, _ = run(sample, folder)
    equal(page, sample["expected_with_stable_names"])
    assert set(p.name for p in folder.iterdir()) == {a["name"] for a in sample["calls"]}
    for path in folder.iterdir():
        equal(
            dict(name=path.name, **pixels(path.read_bytes())),
            next(c for c in sample["calls"] if c["name"] == path.name),
        )
bad = copy.deepcopy(sample["request"])
bad["model"]["layout_dets"][0]["original_order"] = None
sink = Sink()
try:
    assemble_model_page_with_assets(
        root / fixture["source"],
        image_writer=sink,
        table_fill_images=[dict(uuid="UUID-A", bbox=[1, 2, 5, 6], source=root / fixture["source"])],
        **bad,
    )
except RapidDocError:
    pass
else:
    raise AssertionError("model order accepted")
assert not sink.calls
for uuid, data in [("", b"bad"), ("UUID-A", b"bad")]:
    sink = Sink()
    try:
        assemble_model_page_with_assets(
            root / fixture["source"],
            image_writer=sink,
            table_fill_images=[dict(uuid=uuid, bbox=[1, 2, 5, 6], source=data)],
            **sample["request"],
        )
    except RapidDocError:
        pass
    else:
        raise AssertionError("invalid fill accepted")
    assert not sink.calls
# 相同输入的三个独立进程必须生成相同名称，覆盖旧地址命名问题。
code = """import json,sys
from pathlib import Path
from rapid_doc_rust import assemble_model_page_with_assets
root=Path(sys.argv[1]);f=json.loads((root/'build/migration/legacy-table-assets.json').read_text());c=f['cases'][0]
class Writer:
 _parent_dir='images'
 def __init__(self):self.names=[]
 def write(self,name,data):self.names.append(name)
w=Writer();assemble_model_page_with_assets(root/f['source'],image_writer=w,table_fill_images=[dict(uuid=i['uuid'],bbox=i['bbox'],source=root/i['source'])for i in c['fills']],**c['request']);print(json.dumps(w.names))
"""
names = [
    json.loads(subprocess.check_output([sys.executable, "-I", "-c", code, str(root)], text=True))
    for _ in range(3)
]
assert names[0] == names[1] == names[2] == [a["name"] for a in sample["calls"]]
print(
    "Installed wheel: 57 actual old table UUID asset pages, ordered calls/modes/pixels/metadata and 171 outputs pass with explicit stable-name projection; stable names across 3 processes, duplicate overwrites, filtered candidates, prefixes, concurrency, ownership and errors passed"
)
