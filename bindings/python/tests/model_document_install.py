"""独立wheel实际PDF多页编排；旧固定版面/公式+真实CPU OCR/SLANet参考。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, struct, sys, zlib
from pathlib import Path
from rapid_doc_rust import parse_pdf_models, OnnxSession, RapidDocError

root = Path(sys.argv[1]).resolve()
path = root / "build/migration/model-document-default.json"
f = json.loads(path.read_text(encoding="utf-8"))
outputs = json.loads(
    (root / "build/migration/model-document-default-paths.json").read_text(encoding="utf-8")
)
assert hashlib.sha256(path.read_bytes()).hexdigest() == outputs["reference_sha256"]
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
source = (root / f["source"]).read_bytes()
assert hashlib.sha256(source).hexdigest() == f["source_sha256"]
models = f["models"]
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
sessions = {
    name: OnnxSession(runtime, root / models[name + "_model"], models[name + "_sha256"])
    for name in ("det", "rec", "orientation", "table")
}


class Layout:
    def __init__(self, start=0):
        self.index = start

    def layout_png(self, data, **kw):
        r = copy.deepcopy(f["regions"][self.index])
        self.index += 1
        return r


class Formula:
    def recognize_formula_batch_png(self, images, **kw):
        return ["x+y"] * len(images)


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def pixels(data):
    assert data.startswith(b"\x89PNG")
    offset = 8
    packed = bytearray()
    while offset < len(data):
        n = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        v = data[offset + 8 : offset + 8 + n]
        offset += n + 12
        if kind == b"IHDR":
            w, h, depth, color, compression, filter_, interlace = struct.unpack(">IIBBBBB", v)
        if kind == b"IDAT":
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
    return dict(mode="RGB", width=w, height=h, pixels_sha256=hashlib.sha256(out).hexdigest())


def run(source=source, **kw):
    # These frozen model-only references explicitly disabled rule tables in Python.
    options = (
        dict(
            layout_model=Layout(),
            detector=sessions["det"],
            recognizer=sessions["rec"],
            table_model=sessions["table"],
            orientation=sessions["orientation"],
            formula_model=Formula(),
            dpi=72.0,
            use_rule_table=False,
        )
        | kw
    )
    return parse_pdf_models(source, **options)


for c, output in zip(f["cases"], outputs["cases"]):
    r = run(native_text=c["native_text"])
    assert r["source_page_count"] == 5 and r["page_range"] == [0, 5]
    same(r["pages"], c["expected"]["pages"], "pages")
    same(r["markdown"], output["markdown"], "markdown")
    same(r["content_list"], output["content_list"], "content_list")
    same(
        [dict(name=name, **pixels(data)) for name, data in r["assets"].items()],
        c["assets"],
        "assets",
    )
    assert [p["page_idx"] for p in r["pages"]] == list(range(5))
    for name in r["assets"]:
        assert name.startswith("images/")
# UNET单模型及旧默认双模型配置同样走完整多页入口。
route = json.loads((root / "build/migration/legacy-table-routing.json").read_text(encoding="utf-8"))
cm = route["classifiers"]
wired = OnnxSession(runtime, root / route["unet_model"], route["unet_sha256"])
classifier = OnnxSession(runtime, root / cm["model"], cm["sha256"])
secondary = OnnxSession(runtime, root / cm["qmodel"], cm["qsha256"])
for profile, options in [
    ("unet", dict(table_variant="unet", table_model=wired)),
    (
        "dual",
        dict(
            table_variant="unet-slanet-plus",
            wired_model=wired,
            classifier=classifier,
            secondary_classifier=secondary,
        ),
    ),
]:
    path = root / f"build/migration/model-document-{profile}.json"
    ref = json.loads(path.read_text(encoding="utf-8"))
    paths = json.loads(
        (root / f"build/migration/model-document-{profile}-paths.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == paths["reference_sha256"]
    for c, out in zip(ref["cases"], paths["cases"]):
        r = run(native_text=c["native_text"], **options)
        same(r["pages"], c["expected"]["pages"], profile + "/pages")
        same(r["markdown"], out["markdown"], profile + "/markdown")
        same(r["content_list"], out["content_list"], profile + "/content_list")
        same(
            [dict(name=name, **pixels(data)) for name, data in r["assets"].items()],
            c["assets"],
            profile + "/assets",
        )


class Once(io.BytesIO):
    count = 0

    def read(self, *a):
        self.count += 1
        return super().read(*a)


# 适配器返回独立冻结的实际旧PDF原生恢复结果；只证明捷径编排，未运行Rust候选生成。
native_path = root / "build/migration/model-document-native.json"
native_ref = json.loads(native_path.read_text(encoding="utf-8"))
native_out = json.loads(
    (root / "build/migration/model-document-native-paths.json").read_text(encoding="utf-8")
)
assert hashlib.sha256(native_path.read_bytes()).hexdigest() == native_out["reference_sha256"]


class NativeProvider:
    def __init__(self, mode="accept"):
        self.calls = []
        self.mode = mode

    def recover_table(self, data, **kw):
        assert data == source
        self.calls.append(kw)
        if self.mode == "error":
            raise RuntimeError("native backend failed")
        if self.mode == "import":
            raise ImportError("native backend unavailable")
        if self.mode == "none":
            return None
        entry = native_ref["native_references"][0]
        assert kw["bbox"] == entry["bbox"] and kw["page_index"] == entry["page_index"]
        return copy.deepcopy(entry["result"])


for c, out in zip(native_ref["cases"], native_out["cases"]):
    provider = NativeProvider()
    options = {} if not c["native_text"] else dict(table_model=None)
    r = run(
        native_text=c["native_text"],
        native_table_recoverer=provider,
        use_rule_table=True,
        **options,
    )
    same(r["pages"], c["expected"]["pages"], "native/pages")
    same(r["markdown"], out["markdown"], "native/markdown")
    same(r["content_list"], out["content_list"], "native/content_list")
    same(
        [dict(name=name, **pixels(data)) for name, data in r["assets"].items()],
        c["assets"],
        "native/assets",
    )
    assert len(provider.calls) == int(c["native_text"])
    assert r["native_table_routes"][0]["method"] == (
        "docvortex_native" if c["native_text"] else "model"
    )
for mode in ["none", "error"]:
    provider = NativeProvider(mode)
    r = run(native_table_recoverer=provider, use_rule_table=True)
    same(r["pages"], f["cases"][0]["expected"]["pages"], "native fallback/pages")
    assert len(provider.calls) == 1
provider = NativeProvider()
r = run(native_table_recoverer=provider, use_rule_table=False)
assert not provider.calls
try:
    run(native_table_recoverer=NativeProvider("import"), use_rule_table=True)
except ImportError:
    pass
else:
    raise AssertionError("document swallowed native import failure")
s = Once(source)
r = run(s, start_page=1, end_page=2, layout_model=Layout(1))
assert s.count == 1 and not s.closed
assert len(r["pages"]) == 1 and r["pages"][0]["page_idx"] == 1 and r["markdown"] == ""
r = run(start_page=3, end_page=3)
assert r["pages"] == [] and r["assets"] == {}


def fails(**kw):
    try:
        run(**kw)
    except (ValueError, TypeError, RapidDocError):
        return
    raise AssertionError("invalid/unsupported document accepted")


fails(max_pages=4)
fails(max_total_pixels=47_999)
fails(dpi=True)
fails(table_variant="unitable")
fails(start_page=3, end_page=4, layout_model=Layout(3), formula_model=None)
fails(start_page=2, end_page=3, layout_model=Layout(2), table_model=None)


class Broken:
    def layout_png(self, *a, **kw):
        raise RuntimeError("layout failed")


try:
    run(layout_model=Broken())
except RuntimeError as e:
    assert str(e) == "layout failed"
else:
    raise AssertionError("layout error swallowed")
# 真正V3版面推理也执行；本项仅冒烟，不能代替上方固定响应的旧版差分。
layout = json.loads((root / "build/migration/legacy-model.json").read_text(encoding="utf-8"))
actual = OnnxSession(runtime, root / layout["model"], layout["model_sha256"])
r = run(layout_model=actual, start_page=0, end_page=1, formula_enabled=False)
assert len(r["pages"]) == 1 and isinstance(r["markdown"], str)
print(
    "Installed wheel: 6 old model five-page compositions plus 2 old native-policy compositions with frozen native provider, exact JSON/Markdown/content_list/assets; native accept skips table model, decline/error falls back, ImportError propagates; actual CPU V3 PDF smoke passed; no Rust native candidate generation"
)
