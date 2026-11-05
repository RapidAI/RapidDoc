"""独立wheel：旧sparse_hybrid候选、诊断、普通HTML及输入合同。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    generate_native_table_sparse_hybrid_candidates as generate,
    render_native_table_candidate,
    select_native_table_candidates,
    capabilities,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
cases = []
for name in ("native-table-sparse-hybrid", "native-table-sparse-hybrid-extra"):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    for r in f.get("real", []):
        assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]
    cases += f["cases"]


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
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    result = generate(**request)
    same(result, c["expected"], c["label"])
    same(request, before, "ownership")
    for candidate, html in zip(result["candidates"], result["html"]):
        assert render_native_table_candidate(candidate) == html


for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::25]))
good = next(c for c in cases if c["expected"]["candidates"])
assert select_native_table_candidates(generate(**good["request"])["candidates"])["selected"] == 0


def invalid(change):
    r = copy.deepcopy(good["request"])
    change(r)
    try:
        generate(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid sparse hybrid input accepted")


invalid(lambda r: r["table_input"].update(drawing_lines=r["table_input"]["drawing_lines"] * 5001))
invalid(lambda r: r["table_input"].update(rectangles=[{}] * 5001))
invalid(lambda r: r["text"]["rows"][0].update(row_index=1))
invalid(lambda r: r["text"]["rows"][0].update(glyph_ids=[999999]))
invalid(lambda r: r["text"]["glyphs"][0].update(bbox=[0.0, 0.0, float("nan"), 1.0]))
invalid(lambda r: r["table_input"].update(table_bbox=[0.0, 0.0, 1e10, 20.0]))
assert (
    capabilities()["native_table_sparse_hybrid"]
    and not capabilities()["legacy_native_table_recovery"]
)
assert len(cases) == 799 and sum(bool(c["expected"]["candidates"]) for c in cases) == 571
print(
    "Installed wheel: 799 old sparse-hybrid candidates/diagnostics/plain HTML, 571 nonempty, 16 old PDF/angle primitive inputs; exact values/types, header spans, physical token splits, support/integrity gates, ownership/concurrency/budgets/arbitration; no complete engine/PDF extraction"
)
