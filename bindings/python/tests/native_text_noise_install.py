"""Installed noise candidates, paint evidence and full live PDF removal side effects."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    prepare_native_text_noise as prepare,
    apply_native_text_noise as apply,
    filter_pdf_native_text_noise as filter_pdf,
    read_pdf_text_paint as paint,
    read_pdf_native_text_geometry as read,
    build_native_visual_lines as visual,
    prepare_native_source_lines as source,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-text-noise.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-text-noise.json").read_text(encoding="utf-8"))
edges = json.loads(
    (root / "build/migration/native-text-noise-edges.json").read_text(encoding="utf-8")
)
for corpus in [f, pdf, edges]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


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


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    same(
        prepare(q["source"], visual_bboxes=q["visual_bboxes"]),
        c["prepared"],
        c["label"] + "/prepare",
    )
    got = apply(**q)
    same(got, c["expected"], c["label"])
    same(q, before, "ownership")
    got["source"]["chars"].append(dict(char="new"))
    same(q, before, "output independence")


assert len(f["cases"]) == 208 and len(edges["cases"]) == 28
f["cases"].extend(edges["cases"])
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::13]))
assert len(pdf["cases"]) == 32


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    page = c["page_index"]
    same(paint(data, c["indices"], page_index=page), c["paint"], "actual fill " + c["label"])
    g = read(data, page_index=page)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    prepared = source(rows, page_size=g["page_size"])
    s = dict(lines=prepared["lines"], chars=g["chars"], page_index=page)
    same(s, c["source"], "live source")
    same(prepare(s), c["prepared"], "live candidates")
    before = copy.deepcopy(s)
    same(filter_pdf(s, data, page_index=page), c["expected"], "live result")
    same(s, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::9]))


class Unreadable:
    def read(self, *args):
        raise AssertionError("no-candidate path accessed PDF")


empty = dict(lines=[], chars=[], metadata={"keep": 1})
same(
    filter_pdf(empty, Unreadable()),
    dict(source=empty, excluded_source_indices=[], requested_char_indices=[]),
    "lazy read",
)


def error(function, code, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


error(
    prepare,
    "resource_limit",
    dict(
        lines=[
            dict(
                text="1",
                bbox=[0, 0, 1, 1],
                source_index=i,
                font_signature=["N", 0],
                effective_height=1,
            )
            for i in range(4097)
        ],
        chars=[],
    ),
)
q = copy.deepcopy(f["cases"][0]["request"])
error(prepare, "resource_limit", q["source"], visual_bboxes=[[0, 0, 1, 1]] * 3200)
error(apply, "invalid_input", q["source"], paint={"3": [200, 200, 200, 256]})
error(
    prepare, "invalid_input", dict(lines=[dict(text="1", bbox=[0, 0, 1], source_index=0)], chars=[])
)
data = (root / pdf["cases"][0]["pdf"]).read_bytes()
error(paint, "resource_limit", data, [0] * 20001)
error(paint, "invalid_input", data, [2**31])
error(paint, "invalid_input", data, [0], password="a\0b")
error(paint, "invalid_input", data, [0], page_index=999)
error(paint, "corrupt_document", b"bad", [0])
for indices in [[True], [1.5]]:
    try:
        paint(data, indices)
    except ValueError:
        pass
    else:
        raise AssertionError("noninteger index")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_text_noise"]
    and capabilities()["pdf_text_paint"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "236 old noise decisions (199 candidate/178 removal positive), 32 actual PDF paint/source chains (4 removal positive), exact requests/source/cleanup/types/order/ownership/concurrency/lazy read/budgets/recovery passed; strong graphic detection/chart grouping/default PDF remain pending"
)
