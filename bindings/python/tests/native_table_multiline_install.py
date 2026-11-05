"""独立wheel：完整多行/整行组标题回退及原始字符空间投影。"""

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
    generate_native_table_multiline_candidates as generate,
    project_native_table_spatial_text as project,
    render_native_table_candidate,
    select_native_table_candidates,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
cases = []
projection = []
for name in ("native-table-multiline", "native-table-multiline-extra"):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    for r in f.get("real", []):
        assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]
    cases += f["cases"]
    projection += f.get("projection", [])


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
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
    result = generate(**r)
    same(result, c["expected"], c["label"])
    same(r, before, "ownership")
    for candidate, html in zip(result["candidates"], result["html"]):
        assert render_native_table_candidate(candidate) == html


for c in cases:
    run(c)
for c in projection:
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    same(project(**request), c["expected"])
    same(request, before)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::25]))
for groups in (False, True):
    c = next(
        c
        for c in cases
        if c["expected"]["candidates"] and (len(c["expected"]["diagnostics"]) == 2) == groups
    )
    assert select_native_table_candidates(generate(**c["request"])["candidates"])["selected"] == 0
good = next(c for c in cases if c["expected"]["candidates"])


def invalid(change):
    r = copy.deepcopy(good["request"])
    change(r)
    try:
        generate(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid multiline input accepted")


invalid(lambda r: r["text"]["rows"][0].update(row_index=100))
invalid(lambda r: r["text"]["glyphs"][0].update(visual_row=-1))
invalid(lambda r: r["text"]["rows"][0].update(glyph_ids=[999999]))
invalid(lambda r: r["table_input"].update(drawing_lines=[{}] * 5001))
invalid(lambda r: r["table_input"].update(rectangles=[{}] * 5001))
invalid(lambda r: r["text"]["glyphs"][0].update(bbox=[float("nan"), 0.0, 1.0, 1.0]))
for chars in (
    [{}] * 20001,
    [
        dict(char="A", char_idx=0, bbox=[0.0, 0.0, 2.0, 1.0]),
        dict(char="B", char_idx=1, bbox=[9_000_000.0, 0.0, 9_000_002.0, 1.0]),
    ],
):
    try:
        project(chars, table_bbox=[0.0, 0.0, 10_000_000.0, 10.0])
    except (RapidDocError, ValueError, TypeError):
        pass
    else:
        raise AssertionError("projection resource limit bypassed")
assert len(cases) == 1136 and len(projection) == 356
assert sum(bool(c["expected"]["candidates"]) for c in cases) == 548
assert (
    sum(bool(c["expected"]["candidates"]) and len(c["expected"]["diagnostics"]) == 2 for c in cases)
    == 54
)
assert (
    capabilities()["native_table_multiline"]
    and capabilities()["native_table_spatial_projection"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    "Installed wheel: 1136 old full multiline cases, 548 nonempty/54 grouped fallback, 356 spatial projections plus Rust 114 tight-space facts, 16 old PDF/angle primitive inputs; exact values/types/diagnostics/plain HTML, source/space integrity, ownership/concurrency/budgets/arbitration; no full engine/PDF primitive extraction"
)
