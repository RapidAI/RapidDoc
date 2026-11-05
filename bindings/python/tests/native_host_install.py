"""Original native host adapter, lazy table backend, CPU document and Markdown/assets."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import ast, base64, copy, hashlib, io, json, re, struct, sys, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


for c in load("native-host-adapter")["cases"]:
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    try:
        if c["operation"] == "convert":
            out = sdk.convert_native_pdf_layout(
                q["blocks"], q["width"], q["height"], ignored_labels=q.get("ignored_labels")
            )
        elif c["operation"] == "fallback":
            out = sdk.native_pdf_layout_fallback_reason(
                q["blocks"],
                q["layout"],
                q["width"],
                q["height"],
                native_char_count=q.get("native_char_count", 0),
            )
        elif c["operation"] == "direction":
            out = sdk.native_pdf_table_text_angle(q["chars"], q["table_bbox"])
        else:
            pp = q["supplements"]
            if "supplement_identities" in q:
                cache = {}
                pp = [cache.setdefault(i, d) for i, d in zip(q["supplement_identities"], pp)]
            out = sdk.merge_native_pdf_supplements(
                q["native"],
                pp,
                include_formula=q.get("include_formula", True),
                include_seal=q.get("include_seal", True),
            )
    except (sdk.RapidDocError, ValueError):
        assert "error" in c, c["label"]
        continue
    assert "error" not in c, c["label"]
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
table = load("native-host-tables")
library = str(Path(sdk.__file__).parent / "runtime/rapiddoc_pdfium.dll")
data = {p["pdf"]: (root / p["pdf"]).read_bytes() for p in table["pages"]}
for p in table["pages"]:
    assert hashlib.sha256(data[p["pdf"]]).hexdigest() == p["pdf_sha256"]
    got = json.loads(sdk.native_pdf_table_page_json(library, data[p["pdf"]], p["page_index"], None))
    same(got, p["expected"], "detached/" + str(p["page_index"]))
recoverer = sdk.NativePdfTableRecoverer()


def recover(c):
    out = recoverer.recover_table(data[c["pdf"]], page_index=c["page_index"], bbox=c["bbox"])
    same(out, c["expected"], c["label"])
    return out


for c in table["cases"]:
    recover(c)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(recover, table["cases"][-3:]))
print(
    "257 original adapter cases, 8 detached pages, 13 actual table region positives, ownership/alias consumption/cache concurrency passed",
    flush=True,
)
tree = ast.parse(
    (root / "bindings/python/tests/native_table_media_png_install.py").read_text(encoding="utf-8")
)
exec(
    compile(
        ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "pixels"],
            type_ignores=[],
        ),
        "verified_png_decoder",
        "exec",
    ),
    globals(),
)
f = load("native-host-document")
source = (root / f["source"]).read_bytes()
assert hashlib.sha256(source).hexdigest() == f["source_sha256"]
runtime = root / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
m = f["models"]
sessions = {
    k: sdk.OnnxSession(runtime, root / m[k + "_model"], m[k + "_sha256"])
    for k in ["det", "rec", "orientation", "table"]
}


class Formula:
    def recognize_formula_batch_png(self, images, **kw):
        return ["x+y"] * len(images)


def options():
    return dict(
        docvortex_mode="rule",
        native_only=True,
        detector=sessions["det"],
        recognizer=sessions["rec"],
        orientation=sessions["orientation"],
        table_model=sessions["table"],
        formula_model=Formula(),
        dpi=72.0,
    )


for c in f["cases"]:
    got = sdk.parse_pdf_models(source, native_text=c["native_text"], **options())
    (root / f"build/native-host-document-actual-{c['native_text']}.json").write_text(
        json.dumps({k: v for k, v in got.items() if k != "assets"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    same(got["pages"], c["expected"]["pages"], "host/pages")
    same(got["markdown"], c["expected"]["markdown"], "host/markdown")
    same(got["content_list"], c["expected"]["content_list"], "host/content_list")
    same([dict(name=n, **pixels(b)) for n, b in got["assets"].items()], c["assets"], "host/assets")
    assert all(r["method"] == "native" for r in got["native_layout_routes"])
print(
    "2 complete original native host documents / 10 pages, actual CPU OCR/SLANet, Markdown/content_list/decoded assets passed; formula response fixed to x+y",
    flush=True,
)
# Existing original model-layout + native-table reference now runs the built-in Rust backend.
old = load("model-document-native")
paths = load("model-document-native-paths")


class Layout:
    def __init__(self):
        self.index = 0

    def layout_png(self, *a, **kw):
        out = copy.deepcopy(old["regions"][self.index])
        self.index += 1
        return out


got = sdk.parse_pdf_models(
    source,
    layout_model=Layout(),
    detector=sessions["det"],
    recognizer=sessions["rec"],
    orientation=sessions["orientation"],
    table_model=None,
    formula_model=Formula(),
    dpi=72.0,
)
same(got["pages"], old["cases"][0]["expected"]["pages"], "default original native table backend")
same(got["markdown"], paths["cases"][0]["markdown"], "default native table Markdown")
assert got["native_table_routes"][0]["method"] == "docvortex_native"
from native_pdf_preview import choose_native_layout


class Supplement:
    def __init__(self):
        self.calls = 0

    def layout_png(self, *a, **kw):
        self.calls += 1
        return [
            dict(label="display_formula", coordinate=[10.0, 30.0, 80.0, 50.0], score=0.9, order=0)
        ]


sup = Supplement()
blocks = [dict(type="text", bbox=[0.0, 0.0, 1.0, 0.1], content="native")]
kw = dict(
    width=100,
    height=100,
    layout_model=sup,
    supplement_layout_model=sup,
    source=b"layout contract bytes",
    native_only=False,
    ignored_labels=[],
    native_char_count=100,
    native_text=True,
    formula_enabled=True,
    threshold=0.5,
)
out, route = choose_native_layout(
    [dict(type="image", bbox=[0.0, 0.0, 1.0, 1.0])], mode="rule", **kw
)
assert route == dict(method="model", reason="text page reduced to one image") and sup.calls == 1
out, route = choose_native_layout(blocks, mode="rule_plus_s", **kw)
assert (
    sup.calls == 2
    and route["method"] == "native"
    and any(d["category_id"] == 14 for d in out["layout_dets"])
)
out, route = choose_native_layout([], mode="rule_plus_s", **(kw | dict(native_only=True)))
assert sup.calls == 3 and any(d["category_id"] == 14 for d in out["layout_dets"])


class Once(io.BytesIO):
    calls = 0

    def read(self, *a):
        self.calls += 1
        return super().read(*a)


stream = Once(source)
out = sdk.parse_pdf_models(stream, start_page=1, end_page=2, **options())
assert stream.calls == 1 and not stream.closed
assert out["pages"][0]["page_idx"] == 1
original = sdk.analyze_native_pdf


def unexpected(*a, **kw):
    raise AssertionError("empty selection ran native analysis")


try:
    sdk.analyze_native_pdf = unexpected
    out = sdk.parse_pdf_models(source, start_page=3, end_page=3, **options())
    assert out["pages"] == [] and out["assets"] == {}
finally:
    sdk.analyze_native_pdf = original


class Unreadable:
    def read(self, *a):
        raise AssertionError("invalid option read source")


for kw in [
    dict(docvortex_mode="bad"),
    dict(native_only=1),
    dict(docvortex_mode="off", native_only=True),
    dict(markdown_ignore_labels="header"),
]:
    try:
        sdk.parse_pdf_models(Unreadable(), **(options() | kw))
    except ValueError:
        pass
    else:
        raise AssertionError(kw)
assert not any(
    n == "rapid_doc" or n.startswith("rapid_doc.") or n == "pypdfium2" for n in sys.modules
)
assert not any(sdk.compatibility_report()["claims"].values())
print(
    "Built-in default table route, fallback/supplement branches, one stream read, empty selection, strict validation and default gates passed"
)
