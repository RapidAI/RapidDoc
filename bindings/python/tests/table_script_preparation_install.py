"""Independent wheel: old typography, cell preparation and actual single-line HTML chains."""

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
    measure_native_line_typography as measure,
    prepare_native_table_script_cells as prepare,
    classify_native_table_script_line as classify,
    render_native_table_styled_cell as render,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def exact(a, b, p=""):
    assert type(a) is type(b), (p, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), p
        for k in a:
            exact(a[k], b[k], p + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), p
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, p + "/" + str(i))
    else:
        assert a == b, (p, a, b)


def check(fn, c):
    v = copy.deepcopy(c["request"])
    before = copy.deepcopy(v)
    r = fn(**v)
    exact(r, c["expected"], c["label"])
    exact(v, before)
    return r


f = load("table-script-preparation")
extra = load("table-script-preparation-extra")
canonical = load("table-typography-canonical-em")
assert len(f["metrics"]) == 836 and len(extra["metrics"]) == 112 and len(f["preparation"]) == 171
assert len(canonical["cases"]) == 40
for c in f["metrics"] + extra["metrics"] + canonical["cases"]:
    check(measure, c)
prepared = [check(prepare, c) for c in f["preparation"]]
assert sum(bool(c["fraction_members"]) for q in prepared for c in q["cells"]) == 43
assert (
    sum(
        len(c["request"]["table_input"]["drawing_lines"]) > len(q["fraction_rules"])
        for c, q in zip(f["preparation"], prepared)
    )
    == 159
)


def chain(c):
    index = c["preparation_index"]
    cell_index = c["cell_index"]
    v = f["preparation"][index]["request"]
    r = prepared[index]["cells"][cell_index]
    line = r["visual_lines"][0]
    before = copy.deepcopy(line)
    roles = classify(
        line,
        page_size=v["table_input"]["page_size"],
        tight_bboxes=v["tight_bboxes"],
        origins=c["origins"],
        fraction_members=r["fraction_members"],
    )
    exact(roles, c["expected_roles"], c["label"])
    exact(line, before)
    chars = {}
    for i, q in enumerate(v["table_input"]["chars"]):
        key = q.get("char_idx", i)
        if type(key) is int:
            chars.setdefault(str(key), q)
    source = next(q for q in v["result"]["cells"] if (q["row"], q["col"]) == (r["row"], r["col"]))
    html = render(
        source,
        r["glyphs"],
        median_height=v["result"]["text"]["median_glyph_height"],
        roles=roles,
        chars_by_source=chars,
        tight_bboxes=v["tight_bboxes"],
        origins=c["origins"],
        preserve_font_styles=c["preserve_font_styles"],
    )
    exact(html, c["expected_html"], c["label"])


assert (
    len(extra["chains"]) == 3978 and sum(bool(c["expected_roles"]) for c in extra["chains"]) == 168
)
for c in extra["chains"]:
    chain(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(lambda c: check(measure, c), f["metrics"][::73]))
    list(pool.map(lambda c: check(prepare, c), f["preparation"][::17]))
    list(pool.map(chain, extra["chains"][::127]))


def rejects(fn, v, code=None):
    try:
        fn(**v)
    except (ValueError, RapidDocError) as e:
        if code:
            assert isinstance(e, RapidDocError) and e.code == code
    else:
        raise AssertionError("invalid request accepted")


rejects(measure, dict(line=dict(chars=[{}] * 1501, bbox=[0, 0, 1, 1]), page_size=[10, 10]))
rejects(measure, dict(line=dict(chars=[], bbox=[0, 0, 1, 1]), page_size=[-1, 10]))
rejects(measure, dict(line=dict(chars=[], bbox=[0, 0, 1, 1]), page_size=[float("nan"), 10]))
v = copy.deepcopy(f["preparation"][0]["request"])
v["result"]["text"]["glyphs"][0]["bbox"] = ["0", 0, 1, 1]
rejects(prepare, v)
v = copy.deepcopy(f["preparation"][0]["request"])
v["result"]["cells"] = [v["result"]["cells"][0]] * 2
rejects(prepare, v)
# Shared grid-boundary budget: every standalone rule/cell set fits its own cap.
v = copy.deepcopy(f["preparation"][0]["request"])
v["table_input"].update(table_bbox=[0, 0, 100, 100], page_size=[100, 100], angle=0)
v["table_input"]["drawing_lines"] = [
    dict(bbox=[0, 50, 100, 50.1], width=0.1, orientation="horizontal")
] * 5000
cell = dict(v["result"]["cells"][0], bbox=[0, 0, 100, 1], source_char_indices=[])
v["result"]["cells"] = [dict(cell, row=i, col=0) for i in range(1001)]
rejects(prepare, v, "resource_limit")
# Repeated source materialization is bounded before building a large owned result.
v = copy.deepcopy(f["preparation"][0]["request"])
v["table_input"]["chars"] = [
    dict(char="x", char_idx=0, bbox=[0, 0, 1, 1], font=dict(name="X" * 10000))
]
v["table_input"]["drawing_lines"] = []
g = dict(
    v["result"]["text"]["glyphs"][0], source_index=0, text="x", bbox=[0, 0, 1, 1], visual_row=0
)
v["result"]["text"]["glyphs"] = [g]
cell = dict(v["result"]["cells"][0], source_char_indices=[0] * 100)
v["result"]["cells"] = [dict(cell, col=i) for i in range(9)]
rejects(prepare, v, "resource_limit")
check(measure, f["metrics"][0])
check(prepare, f["preparation"][0])
chain(extra["chains"][0])
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").read_bytes() == (
    root / "licenses/MIGRATION-NOTICES.txt"
).read_bytes()
assert (
    capabilities()["native_line_typography"] is True
    and capabilities()["native_table_script_preparation"] is True
)
assert capabilities()["legacy_native_table_recovery"] is False
assert "rapid_doc" not in sys.modules
print(
    "Passed 988 old typography metrics, 171 preparations (43 fraction-positive cells, 159 rule-filtering cases), 3978 actual one-line preparation-to-HTML chains (168 script outputs), ownership, concurrency, shared work and clone budgets; 2D merging remains pending"
)
