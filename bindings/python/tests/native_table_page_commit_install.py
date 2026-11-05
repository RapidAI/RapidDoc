"""Installed continuous page table detection -> original-line commit -> claims and continuation."""

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
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def read(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def invoke(q):
    before = copy.deepcopy(q)
    keys = "source excluded_bboxes prior_fixed_blocks prior_claimed_line_indices table_header_separator_bboxes tight_bboxes origins calibrated_char_bboxes".split()
    out = sdk.prepare_native_table_page(**{k: v for k, v in q.items() if k in keys})
    same(q, before, "ownership")
    return out


def run(c):
    same(invoke(c["request"]), c["expected"], c["label"])


def rejects(q, code):
    try:
        invoke(q)
    except sdk.RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(code)


jobs = []
duplicate_boundaries = []
for name, n in [
    ("native-table-page-commit", 120),
    ("native-table-page-commit-specials", 26),
    ("native-table-page-commit-aliases", 3),
]:
    f = read(name)
    assert len(f["cases"]) == n and not f["errors"]
    for c in f["cases"]:
        ids = [l["source_index"] for l in c["request"]["source"]["lines"]]
        if len(ids) != len(set(ids)):
            duplicate_boundaries.append(c)
            rejects(c["request"], "invalid_input")
        else:
            jobs.append(c)
assert len(jobs) == 147 and len(duplicate_boundaries) == 2
for c in jobs:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::17]))
numeric = next(c for c in jobs if c["label"] == "four-row-numeric")
out = invoke(numeric["request"])
assert any(b["_native_numeric_grid"] for b in out["table_blocks"])
for c in jobs:
    out = c["expected"]
    assert (
        out["fixed_blocks"]
        == c["request"].get("prior_fixed_blocks", [])
        + out["annotation_blocks"]
        + out["table_blocks"]
    )
    assert {l["source_index"] for l in out["remaining_lines"]}.isdisjoint(
        out["claimed_line_indices"]
    )
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    source = dict(
        page_size=g["page_size"],
        lines=[
            {k: l.get(k, False if k == "compact_formula_cluster" else None) for k in fields}
            for l in ls
        ],
        chars=g["chars"],
        drawing_lines=v["drawing_lines"],
        path_infos=v["path_infos"],
        form_bboxes=[],
        image_bboxes=[],
        signature_bboxes=[],
        page_index=c["page_index"],
    )
    same(source, c["request"]["source"], "live source")
    q = dict(
        source=source,
        tight_bboxes={str(ch["char_idx"]): ch.get("tight_bbox") for ch in g["chars"]},
        origins={str(ch["char_idx"]): ch.get("origin") for ch in g["chars"]},
    )
    same(q, c["request"], "maps and source")
    same(invoke(q), c["expected"], c["label"])


pdf = read("pdf-table-page-commit")["cases"]
assert (
    len(pdf) == 59
    and sum(bool(c["expected"]["table_blocks"]) for c in pdf) == 41
    and sum(c["expected"]["next_table_grid"] is not None for c in pdf) == 6
)
for c in pdf:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf[::19]))
base = numeric["request"]
for field, count in [("lines", 4097), ("chars", 20001), ("drawing_lines", 5001)]:
    q = copy.deepcopy(base)
    q["source"][field] = [{}] * count
    rejects(q, "resource_limit")
q = dict(copy.deepcopy(base), prior_fixed_blocks=[{}] * 4001)
rejects(q, "resource_limit")
q = copy.deepcopy(base)
q["source"]["page_size"] = [0, 600]
rejects(q, "invalid_input")
out = invoke(base)
out["table_blocks"][0]["content"] = "mutated"
out["source"]["lines"][0]["text"] = "mutated"
run(numeric)
assert (
    sdk.capabilities()["native_table_page_commit"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "147 typed old continuous table-stage results including 3 source-alias facts + 2 duplicate-source contract rejections; 59 live PDF pages/41 table outputs/6 continuation grids; numeric-grid/protected members/separators/maps, strict types/order/ownership/concurrency/limits/recovery passed; full page/default PDF remains pending"
)
