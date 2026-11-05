"""Installed SDK: new script payloads, full output/diagnostics and positive actual PDFs."""

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
    prepare_native_script_lines as prepare_scripts,
    trace_native_inline_evidence as trace,
    apply_native_inline_evidence as apply,
    read_pdf_native_text_geometry as read,
    read_pdf_character_primitives as raw,
    read_pdf_link_annotations as read_links,
    detect_native_inline_evidence as detect,
    build_native_visual_lines as build,
    prepare_native_source_lines as prepare,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


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


f = load("native-script-output")
g = load("native-script-payloads")
assert (
    len(f["cases"]) == 249
    and len(g["cases"]) == 516
    and len(g["excluded_preclassified_fraction_inputs"]) == 44
)


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    same(trace(**q), c["expected"], c["label"])
    same(apply(**q), c["expected"]["blocks"], "non-diagnostic")
    same(q, before, "ownership")


def payload(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    same(prepare_scripts(**q), c["expected"], c["label"])
    same(q, before, "payload ownership")


for c in f["cases"]:
    run(c)
for c in g["cases"]:
    payload(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::17]))
    list(pool.map(payload, g["cases"][::31]))
old_output = load("native-inline-output")
inline = load("native-inline-evidence")
source = load("native-source-lines")
visual = {name: load(name) for name in ["native-visual-lines", "native-visual-lines-extra"]}
negative_chains = [c for c in f["cases"] if "output_case" in c]
assert len(negative_chains) == 57
for c in negative_chains:
    oc = old_output["cases"][c["output_case"]]
    ic = inline["cases"][oc["inline_case"]]
    sc = source["cases"][ic["source_case"]]
    r = sc["visual_case"]
    v = visual[r["corpus"]]["cases"][r["index"]]
    data = (root / v["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == v["pdf_sha256"]
    dedup = v.get("reader") == "deduplicated"
    geometry = (read if dedup else raw)(data, page_index=v["page_index"])
    lines = build(
        geometry["chars" if dedup else "restored_chars"],
        page_size=geometry["page_size"],
        page_rotation=geometry["rotation"],
    )
    q = copy.deepcopy(sc["request"])
    q["lines"] = lines
    p = prepare(**q)
    all_chars = [ch for l in p["lines"] for ch in l["chars"]]
    tight = {
        ch["char_idx"]: ch["tight_bbox"] for ch in all_chars if ch.get("tight_bbox") is not None
    }
    origins = {ch["char_idx"]: ch["origin"] for ch in all_chars if ch.get("origin") is not None}
    scripts = prepare_scripts(
        p["lines"],
        page_size=geometry["page_size"],
        tight_bboxes=tight,
        origins=origins,
        all_chars=all_chars,
        drawing_lines=p["drawing_lines"],
    )
    same(scripts, c["request"]["script_lines"], "live negative payload")
    evidence = detect(
        p["lines"],
        drawing_lines=p["drawing_lines"],
        annotations=read_links(data, page_index=v["page_index"]),
    )
    blocks = [
        dict(type="text", content=l["text"], bbox=l["bbox"], source="preserved") for l in p["lines"]
    ]
    same(
        trace(blocks, page_size=geometry["page_size"], script_lines=scripts, **evidence),
        c["expected"],
        "live negative output",
    )
assert (
    len(g["real"]) == 4
    and sum(len(c["output_expected"]["materialized_ranges"]) for c in g["real"]) == 7
)
for c in g["real"]:
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    geometry = read(data, page_index=c["page_index"])
    lines = build(
        geometry["chars"],
        page_size=geometry["page_size"],
        page_rotation=geometry["rotation"],
        supported_angles=(0.0, 90.0, 180.0, 270.0),
    )
    p = prepare(lines, page_size=geometry["page_size"])
    scripts = prepare_scripts(
        p["lines"],
        page_size=geometry["page_size"],
        tight_bboxes=geometry["tight_bboxes"],
        origins=geometry["origins"],
        all_chars=geometry["chars"],
    )
    same(scripts, c["expected"], "actual PDF payload")
    blocks = [dict(type="text", content=l["text"], bbox=l["bbox"]) for l in p["lines"]]
    same(
        trace(blocks, page_size=geometry["page_size"], script_lines=scripts),
        c["output_expected"],
        "actual PDF output/diagnostics",
    )
# Omitting the new stage preserves the existing API; an explicit empty script list runs old cleanup.
b = [dict(type="text", content="x", bbox=[0.0, 0.0, 1.0, 1.0], _inline_math_regions=[])]
assert "_inline_math_regions" in apply(b, page_size=[400.0, 600.0])[0]
assert "_inline_math_regions" not in trace(b, page_size=[400.0, 600.0])["blocks"][0]
try:
    trace(
        [], page_size=[400.0, 600.0], script_lines=f["cases"][0]["request"]["script_lines"] * 4097
    )
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("script line count not enforced")
run(f["cases"][0])
assert (
    capabilities()["native_script_output"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "249 complete script outputs/diagnostics +516 old detector payloads; 44 preclassified fraction references retained separately; 57 live negative PDF chains +4 live PDF angles/7 actual materialized scripts, exact types/order/metadata, ownership/concurrency/budgets passed. Existing apply default unchanged; complete page-source/candidate/default PDF integration remain pending"
)
