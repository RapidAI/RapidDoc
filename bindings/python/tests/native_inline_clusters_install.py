"""Installed wheel: complete old 2D merge, multi-line roles and HTML stage chains."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import (
    merge_native_inline_clusters as merge,
    classify_native_table_script_cell as classify,
    classify_native_script_line as script_line,
    classify_native_script_runs as runs,
    native_fraction_member_sources as fraction,
    render_native_table_styled_cell as render,
    prepare_native_table_script_cells as prepare,
    build_native_table_text as build,
    capabilities,
    RapidDocError,
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


f = load("native-inline-clusters")
edge = load("native-inline-clusters-empty")
empty = load("script-empty-printable")
styled = load("native-inline-clusters-styled")
raw = load("native-inline-cell-pipeline")
assert (
    len(f["cases"]) == 910
    and len(edge["cases"]) == 16
    and len(empty["fractions"]) == 9
    and len(empty["runs"]) == 48
)
assert sum(any(len(g) > 1 for g in c["expected"]["groups"]) for c in f["cases"]) == 262
assert sum(c["events"].get("paired", False) for c in f["cases"]) == 18
assert sum(c["events"].get("indexed", False) for c in f["cases"]) == 155
actual_roles = {}


def case(pair):
    index, c = pair
    v = copy.deepcopy(c["request"])
    before = copy.deepcopy(v)
    r = merge(**v)
    exact(r, c["expected"], c["label"])
    exact(v, before)
    assert Counter(i for g in r["groups"] for i in g) == Counter(range(len(v["lines"])))
    if "role_request" in c:
        v = copy.deepcopy(c["role_request"])
        before = copy.deepcopy(v)
        roles = classify(**v)
        exact(roles, c["expected_roles"], c["label"])
        exact(v, before)
        actual_roles[index] = roles
        if "render_request" in c:
            exact(render(**c["render_request"], roles=roles), c["expected_html"], c["label"])


for pair in enumerate(f["cases"]):
    case(pair)
for c in edge["cases"]:
    exact(merge(**c["request"])["lines"], c["expected_lines"], c["label"])
    exact(classify(**c["role_request"]), c["expected_roles"], c["label"])
    exact(script_line(**c["line_request"]), c["expected_line"], c["label"])
for key, fn in [("fractions", fraction), ("runs", runs)]:
    for c in empty[key]:
        exact(fn(**c["request"]), c["expected"], c["label"])
assert len(styled["cases"]) == 88
for c in styled["cases"]:
    html = render(**c["request"], roles=actual_roles[c["case_index"]])
    exact(html, c["expected"], c["label"])
assert all("<sup>" in c["expected"] or "<sub>" in c["expected"] for c in styled["cases"])


def pipeline(c):
    text = build(**c["text_request"])
    exact(text, c["expected_text"], c["label"])
    v = copy.deepcopy(c["preparation_request"])
    v["result"]["text"] = text
    prepared = prepare(**v)
    cell = prepared["cells"][0]
    merged = merge(cell["visual_lines"], page_size=v["table_input"]["page_size"])
    exact(merged["lines"], c["expected_merged_lines"], c["label"])
    roles = classify(
        cell["visual_lines"],
        page_size=v["table_input"]["page_size"],
        tight_bboxes=v["tight_bboxes"],
        origins=c["origins"],
        fraction_members=cell["fraction_members"],
    )
    exact(roles, c["expected_roles"], c["label"])
    chars = {str(q["char_idx"]): q for q in v["table_input"]["chars"]}
    html = render(
        v["result"]["cells"][0],
        cell["glyphs"],
        median_height=text["median_glyph_height"],
        roles=roles,
        chars_by_source=chars,
        tight_bboxes=v["tight_bboxes"],
        origins=c["origins"],
        preserve_font_styles=c["preserve_font_styles"],
    )
    exact(html, c["expected_html"], c["label"])


assert len(raw["cases"]) == 48
for c in raw["cases"]:
    pipeline(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(case, list(enumerate(f["cases"]))[::47]))
    list(pool.map(pipeline, raw["cases"][::7]))


def rejects(fn, v, code=None):
    try:
        fn(**v)
    except (ValueError, RapidDocError) as e:
        if code:
            assert isinstance(e, RapidDocError) and e.code == code
    else:
        raise AssertionError("invalid cluster input accepted")


base = copy.deepcopy(f["cases"][0]["request"])
base["table_bboxes"] = []
rejects(merge, dict(base, lines=[base["lines"][0]] * 513), "resource_limit")
v = copy.deepcopy(base)
v["lines"][1]["source_index"] = v["lines"][0]["source_index"]
rejects(merge, v)
v = copy.deepcopy(base)
v["lines"][0]["angle"] = 45
rejects(merge, v)
v = copy.deepcopy(base)
v["lines"][0]["caption_start"] = 1
rejects(merge, v)
rejects(merge, dict(base, page_size=[0, 600]))
rejects(merge, dict(base, page_size=[float("nan"), 600]))
rejects(merge, dict(base, table_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
v = copy.deepcopy(base)
v["lines"][0]["chars"] = [{}] * 750
v["lines"][1]["chars"] = [{}] * 751
rejects(merge, v, "resource_limit")
v = copy.deepcopy(base)
v["lines"][0]["text"] = chr(0) * 1400000
rejects(merge, v, "resource_limit")
# The inner lane scan and the following cluster operations use the same work counter.
rows = []


def add(b):
    i = len(rows)
    rows.append(dict(text="body", bbox=b, angle=0, source_index=i, effective_height=10.0))


for y in (0.0, 10.0, 20.0):
    add([0.0, y, 300.0, y + 10.0])
for y in (10000.0 + i * 20.0 for i in range(10)):
    add([0.0, y, 140.0, y + 10.0])
    add([160.0, y, 300.0, y + 10.0])
for i in range(480):
    y = 30.0 + i * 10.0
    add([0.0, y, 40.0, y + 10.0])
rejects(merge, dict(lines=rows, page_size=[300.0, 20000.0]), "resource_limit")
rejects(classify, dict(lines=rows, page_size=[300.0, 20000.0]), "resource_limit")
case((0, f["cases"][0]))
pipeline(raw["cases"][0])
assert (
    capabilities()["native_inline_clusters"] is True
    and capabilities()["native_table_cell_script_roles"] is True
)
assert (
    capabilities()["legacy_native_table_recovery"] is False
    and capabilities()["legacy_table_execution"] is False
)
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").read_bytes() == (
    root / "licenses/MIGRATION-NOTICES.txt"
).read_bytes()
assert "rapid_doc" not in sys.modules
print(
    "Passed 910 complete 2D merges (262 positives, 18 paired, 155 indexed), 750 complete cell role cases, 480 cell HTML chains, 88 authored styled-positive chains, 48 raw-character negative chains, 16 empty/control boundaries, 9 fraction-scale and 48 full-run empty references; exact payloads/identity/order/types, ownership, concurrency, budgets and recovery; full table HTML orchestration pending"
)
