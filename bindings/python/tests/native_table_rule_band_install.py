"""独立wheel旧rule_band及共享稀疏列/空白走廊，逐值/类型对照。"""

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
    generate_native_table_rule_band_candidates,
    inspect_native_table_sparse_columns,
    select_native_table_candidates,
    render_native_table_candidate,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    for r in f.get("real", []):
        assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]
    return f


f = load("native-table-rule-band")
extra = load("native-table-rule-band-extra")
f["cases"] += extra["cases"]


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


def run(pair):
    fn, c = pair
    r = copy.deepcopy(c["request"])
    before = copy.deepcopy(r)
    result = fn(**r)
    same(result, c["expected"])
    same(r, before, "ownership")
    if fn is generate_native_table_rule_band_candidates:
        for candidate, html in zip(result["candidates"], result["html"]):
            assert render_native_table_candidate(candidate) == html


pairs = [(generate_native_table_rule_band_candidates, c) for c in f["cases"]] + [
    (inspect_native_table_sparse_columns, c) for c in f["columns"]
]
for p in pairs:
    run(p)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, pairs[::20]))
good = next(c for c in f["cases"] if c["expected"]["candidates"])
result = generate_native_table_rule_band_candidates(**good["request"])
assert select_native_table_candidates(result["candidates"])["selected"] == 0
# 发布门比builder严格：未完全捕获和少量歧义都不得输出候选，保留原诊断。
for c in extra["cases"]:
    assert (
        not c["expected"]["candidates"]
        and c["expected"]["diagnostics"][0]["candidate_rejection_gate"] is None
    )


def fails(fn, r):
    try:
        fn(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid sparse input accepted")


r = copy.deepcopy(good["request"])
r["table_input"]["drawing_lines"] = [r["table_input"]["drawing_lines"][0]] * 5001
fails(generate_native_table_rule_band_candidates, r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"][0]["visual_row"] = -1
fails(generate_native_table_rule_band_candidates, r)
r = copy.deepcopy(good["request"])
r["text"]["rows"][0]["glyph_ids"] = [999999]
fails(generate_native_table_rule_band_candidates, r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"][0]["bbox"][0] = float("nan")
fails(generate_native_table_rule_band_candidates, r)
assert (
    capabilities()["native_table_rule_band"]
    and capabilities()["native_table_sparse_columns"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(f['cases'])} rule-band cases +{len(f['columns'])} shared sparse column/gutter facts including {len(f['real'])} old PDF/angle inputs; strict publication/values/types/rowspans/HTML, ownership/concurrency/resource guards/arbitration; complete multiline/engine/PDF extraction not implemented"
)
