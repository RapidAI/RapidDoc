"""Installed wheel: full old semantic table HTML and recovery, with live PDF primitives."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import (
    render_native_table_html_with_scripts as render,
    recover_native_table_result as recover,
    recover_native_table_primitives as engine,
    read_pdf_native_text_geometry,
    read_pdf_vector_geometry,
    RapidDocError,
    NativeTableRecoveryError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-table-materialization.json").read_text(encoding="utf-8")
)
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def exact(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def recovered(v):
    r = recover(**v)
    if r is None:
        return None
    assert type(r) is tuple and len(r) == 2
    return dict(content=r[0], result=r[1])


def case(c, fn):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    exact(fn(q), c["expected"], c.get("label", "real"))
    exact(q, before, "ownership")


for c in f["renders"]:
    case(c, lambda q: render(**q))
for c in f["recoveries"]:
    case(c, recovered)
assert len(f["renders"]) == 526 and len(f["recoveries"]) == 724 and len(f["real"]) == 32
assert sum("<sup>" in c["expected"] or "<sub>" in c["expected"] for c in f["renders"]) == 168
for c in f["real"]:
    assert hashlib.sha256((root / c["pdf"]).read_bytes()).hexdigest() == c["pdf_sha256"]
    q = copy.deepcopy(c["request"])
    ti = q["table_input"]
    g = read_pdf_native_text_geometry(root / c["pdf"], page_index=c["page_index"])
    vectors = read_pdf_vector_geometry(root / c["pdf"], page_index=c["page_index"])
    ti.update(
        chars=g["chars"],
        drawing_lines=vectors["drawing_lines"],
        rectangles=vectors["path_infos"],
        page_size=g["page_size"],
    )
    q.update(tight_bboxes=g["tight_bboxes"], origins=g["origins"])
    exact(recovered(q), c["expected"], "live PDF primitives -> engine -> full HTML")
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(lambda c: case(c, lambda q: render(**q)), f["renders"][::21]))
    list(pool.map(lambda c: case(c, recovered), f["recoveries"][::53]))


def fails(q, kind, code):
    try:
        recovered(q)
    except RapidDocError as e:
        assert type(e) is kind, (type(e), str(e))
        assert e.code == code, (e.code, code)
        return e
    raise AssertionError("failure silently accepted")


good = next(c["request"] for c in f["recoveries"] if c["expected"] is not None)
q = copy.deepcopy(good)
q["table_input"]["page_size"] = ["invalid", 200.0]
assert fails(q, NativeTableRecoveryError, "invalid_input").stage == "recovery"
q = copy.deepcopy(good)
q["tight_bboxes"] = {str(i): [0.0, 0.0, 1.0, 1.0] for i in range(20001)}
q["origins"] = {"0": [0.0, 0.0]}
fails(q, RapidDocError, "resource_limit")
exact(
    recovered(good),
    next(c["expected"] for c in f["recoveries"] if c["expected"] is not None),
    "recovery after error",
)
q = copy.deepcopy(next(c["request"] for c in f["renders"] if c["label"] == "tight-space:True"))
q["result"]["cells"][0]["content"] = "&" * 800000
try:
    render(**q)
except RapidDocError as e:
    assert type(e) is RapidDocError and e.code == "resource_limit"
else:
    raise AssertionError("4 MB escaped table output accepted")
case(f["renders"][0], lambda q: render(**q))
lines = [
    dict(bbox=[0.0, y, 100.0, y + 0.1], width=0.1, orientation="horizontal")
    for y in (0.0, 30.0, 59.9)
] + [
    dict(bbox=[x, 0.0, x + 0.1, 60.0], width=0.1, orientation="vertical") for x in (0.0, 50.0, 99.9)
]
chars = [
    dict(
        char_idx=i,
        char="A",
        bbox=[
            10.0 + 50.0 * ((i // 375) % 2),
            10.0 + 30.0 * ((i // 375) // 2),
            18.0 + 50.0 * ((i // 375) % 2),
            18.0 + 30.0 * ((i // 375) // 2),
        ],
    )
    for i in range(1500)
]
ti = dict(
    table_bbox=[0.0, 0.0, 100.0, 60.0],
    page_size=[200.0, 200.0],
    angle=0,
    drawing_lines=lines,
    rectangles=[],
    chars=chars,
)
tight = {c["char_idx"]: c["bbox"] for c in chars}
origins = {c["char_idx"]: [c["bbox"][0], c["bbox"][3]] for c in chars}
r = engine(ti)
assert r is not None
assert render(r, ti, tight_bboxes=tight, origins=origins, preserve_font_styles=True)
fails(
    dict(table_input=ti, tight_bboxes=tight, origins=origins, preserve_font_styles=True),
    RapidDocError,
    "resource_limit",
)
for fn, args in [(render, f["renders"][0]["request"]), (recover, good)]:
    q = copy.deepcopy(args)
    q["preserve_font_styles"] = 1
    try:
        fn(**q)
    except TypeError:
        pass
    else:
        raise AssertionError("non-bool flag accepted")
assert (
    not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (
    Path(__import__("rapid_doc_rust").__file__).parent / "licenses/MIGRATION-NOTICES.txt"
).is_file()
print(
    "Installed wheel: 526 old complete table HTML (168 automatic script-positive typed result facts), 724 full recoveries, 32 live PDF/angle/style chains; exact output/types, raw-result retention, ownership/concurrency, stage-specific errors and limits; default PDF integration remains deferred"
)
