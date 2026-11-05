"""独立wheel旧数值重分行/列分组/完整候选，不调用生产Python。"""

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
    build_native_table_aligned_numeric_candidate,
    regroup_native_table_numeric_text,
    select_native_table_candidates,
    render_native_table_candidate,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-table-aligned-numeric.json").read_text(encoding="utf-8")
)
extra = json.loads(
    (root / "build/migration/native-table-aligned-numeric-pdf.json").read_text(encoding="utf-8")
)
for p, h in extra["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
f["cases"] += extra["cases"]
f["real"] += extra["real"]
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
for r in f["real"]:
    assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    result = build_native_table_aligned_numeric_candidate(**r)
    same(result, c["expected"])
    same(r, before, "ownership")
    if r["regroup"]:
        same(regroup_native_table_numeric_text(r["text"]), result["text"], "regroup")
    if result["candidate"] is not None:
        assert render_native_table_candidate(result["candidate"]) == result["html"]


for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::12]))
good = next(
    c
    for c in f["cases"]
    if c["expected"]["candidate"] is not None and c["expected"]["candidate"]["score"] == 1.0
)
result = build_native_table_aligned_numeric_candidate(**good["request"])
assert select_native_table_candidates([result["candidate"]])["selected"] == 0


def fails(r):
    try:
        build_native_table_aligned_numeric_candidate(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid aligned numeric accepted")


r = copy.deepcopy(good["request"])
r["table_input"]["drawing_lines"] = [r["table_input"]["drawing_lines"][0]] * 5001
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"] *= 3000
fails(r)
r = copy.deepcopy(good["request"])
r["regroup"] = False
r["text"]["rows"][0]["row_index"] = 100
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"][0]["spacing_bbox"] = [0.0, 0.0, 1e10, 20.0]
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["median_glyph_height"] = 1e10
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"][0]["bbox"][0] = float("nan")
fails(r)
assert (
    capabilities()["native_table_aligned_numeric"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(f['cases'])} old numeric regroup/groups/candidates including {len(f['real'])} old PDF/angle inputs; exact types/text/provenance/spans/scores/HTML, ownership/concurrency/budgets and source arbitration; component scope verified; page producer integration covered by native_all_candidate_producers; final PDF policies/default path pending"
)
