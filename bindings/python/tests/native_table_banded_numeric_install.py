"""独立wheel：分带数值候选、原始字符投影和来源整数set语义。"""

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
    build_native_table_banded_numeric_candidate as build,
    render_native_table_candidate,
    select_native_table_candidates,
    capabilities,
    RapidDocError,
)

root = Path(sys.argv[1]).resolve()
cases = []
errors = []
for name in (
    "native-table-banded-numeric",
    "native-table-banded-numeric-extra",
    "native-table-banded-numeric-ties",
):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    for r in f.get("real", []):
        assert hashlib.sha256((root / r["pdf"]).read_bytes()).hexdigest() == r["pdf_sha256"]
    cases += f["cases"]
    errors += f.get("errors", [])


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
    result = build(**r)
    same(result, c["expected"], c["label"])
    same(r, before, "ownership")
    if result["candidate"] is not None:
        assert render_native_table_candidate(result["candidate"]) == result["html"]


for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::30]))
good = next(c for c in cases if c["expected"]["candidate"] is not None)
assert select_native_table_candidates([build(**good["request"])["candidate"]])["selected"] == 0
order = next(c for c in cases if c["label"] == "order-less-than-one-accepted")["expected"][
    "candidate"
]
assert 0.98 <= order["order_consistency"] < 1.0 and order["score"] == order["order_consistency"]
assert select_native_table_candidates([order])["selected"] == 0
capture = next(c for c in cases if c["label"] == "builder-accepted-incomplete-capture")["expected"]
assert (
    capture["candidate"] is None
    and capture["diagnostics"]["candidate_diagnostics"]["candidate_rejection_gate"] is None
)


def fails(r):
    try:
        build(**r)
    except (RapidDocError, ValueError, TypeError):
        return
    raise AssertionError("invalid banded input accepted")


for c in errors:
    fails(c["request"])
for change in (
    lambda r: r["text"]["rows"][0].update(row_index=1),
    lambda r: r["text"]["rows"][1].update(glyph_ids=[999999]),
    lambda r: r["text"]["glyphs"][0].update(visual_row=-1),
    lambda r: r["text"]["glyphs"][0].update(bbox=[0.0, 0.0, float("nan"), 1.0]),
    lambda r: r["table_input"].update(rectangles=[{}] * 5001),
    lambda r: r["table_input"].update(drawing_lines=[{}] * 5001),
):
    r = copy.deepcopy(good["request"])
    change(r)
    fails(r)
assert len(cases) == 1751 and sum(c["expected"]["candidate"] is not None for c in cases) == 262
assert (
    capabilities()["native_table_banded_numeric"]
    and not capabilities()["legacy_native_table_recovery"]
)
print(
    "Installed wheel: 1751 old banded numeric candidates/diagnostics/reprojected plain HTML, 262 nonempty, 16 old PDF/angle primitive inputs; 1843 numeric/12 gutter/164 signed-set facts in Rust; modal-count ties, strict capture but old <100% order allowed, source errors, ownership/concurrency/budgets/arbitration; no full engine/PDF extraction"
)
