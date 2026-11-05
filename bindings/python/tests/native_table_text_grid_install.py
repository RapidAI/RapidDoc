"""独立wheel旧少线/无线/key-value候选与诊断，JSON值/类型和普通HTML逐项对照。"""

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
    generate_native_table_text_candidates,
    select_native_table_candidates,
    render_native_table_candidate,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-table-text-grid.json").read_text(encoding="utf-8"))
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
    result = generate_native_table_text_candidates(**r)
    same(result, c["expected"])
    same(r, before, "ownership")
    for candidate, html in zip(result["candidates"], result["html"]):
        assert render_native_table_candidate(candidate) == html


for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::15]))
good = next(
    c
    for c in f["cases"]
    if c["expected"]["candidates"] and c["expected"]["candidates"][0]["score"] == 1.0
)
result = generate_native_table_text_candidates(**good["request"])
assert select_native_table_candidates(result["candidates"])["selected"] is not None


def fails(r):
    try:
        generate_native_table_text_candidates(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid text-grid accepted")


r = copy.deepcopy(good["request"])
r["table_input"]["drawing_lines"] = [dict(bbox=[0.0, 10.0, 100.0, 11.0])] * 5001
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["rows"][0]["tokens"][0]["bbox"][0] = float("nan")
fails(r)
r = copy.deepcopy(good["request"])
r["text"]["glyphs"] *= 1000
fails(r)
assert (
    capabilities()["native_table_text_candidates"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    f"Installed wheel: {len(f['cases'])} old sparse/text/key-value candidates including {len(f['real'])} old PDF/angle inputs; exact diagnostics/scores/content/source indices/HTML/types, ownership/concurrency/budgets/source arbitration; full engine not implemented"
)
