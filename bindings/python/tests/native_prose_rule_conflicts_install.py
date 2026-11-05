"""Installed prose draft exclusion with live text/rule query reconstruction."""

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
    exclude_native_prose_spanning_rule_drafts as exclude,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-prose-rule-conflicts.json").read_text(encoding="utf-8")
)
pdf = json.loads(
    (root / "build/migration/pdf-prose-rule-conflicts.json").read_text(encoding="utf-8")
)
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = exclude(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["candidates"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 151
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(pdf["cases"]) == 20


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    fields = "text bbox source_index angle caption_start formula_candidate_only semantic_type dominant_font_weight".split()
    source = dict(lines=[{k: l[k] for k in fields} for l in rows], drawing_lines=v["drawing_lines"])
    draft = dict(
        kind="draft",
        angle=0,
        median_height=10.0,
        caption={} if c["label"].endswith("caption-draft") else None,
        boundaries=[r["bbox"] for r in v["drawing_lines"] if r["orientation"] == "horizontal"],
        member_source_indices=[l["source_index"] for l in rows],
    )
    q = dict(
        source=source,
        candidates=[
            draft,
            dict(kind="table", angle=0, bbox=v["path_infos"][0]["bbox"], core_bbox=None),
        ],
    )
    same(q, c["request"], "live query")
    before = copy.deepcopy(q)
    same(exclude(**q), c["expected"], c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::4]))


def error(q, code):
    try:
        exclude(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source={}, candidates=[])
error(
    dict(base, candidates=[dict(kind="table", angle=0, bbox=[0, 0, 1, 1])] * 5001), "resource_limit"
)
for key, n in [("lines", 4097), ("drawing_lines", 5001)]:
    error(dict(source={key: [{}] * n}, candidates=[]), "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["candidates"][0]["member_source_indices"] = [0] * 20001
error(q, "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["candidates"][0]["boundaries"] = [[0, 0, 1, 1]] * 20001
error(q, "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["lines"] = [q["source"]["lines"][0]] * 1800
draft, grid = q["candidates"]
q["candidates"] = [draft] * 10 + [grid] * 100
error(q, "resource_limit")
error(dict(source={}, candidates=[dict(kind="unknown")]), "invalid_input")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["lines"][0]["text"] = None
error(q, "invalid_input")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["lines"][0]["text"] = "x" * 20001
error(q, "resource_limit")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_prose_rule_conflicts"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "151 original prose-draft exclusion references/112 rejections; 20 live PDF text+rule projected queries/2 rejections; exact metadata/order/types/ownership/concurrency/budgets/recovery passed; prose projection scope verified; ordinary draft generation/assembly now covered by native_structural_table_pipeline; remaining general branches/default PDF pending"
)
