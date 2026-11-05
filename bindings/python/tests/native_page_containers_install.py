"""Installed complete index/Form selectors and continuous container/table stage."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
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


def invoke(kind, q):
    before = copy.deepcopy(q)
    if kind == "index":
        out = sdk.extract_native_index_blocks(**q)
    elif kind == "forms":
        out = sdk.select_native_form_regions(q["source"])
    else:
        keys = "source excluded_bboxes prior_fixed_blocks prior_claimed_line_indices table_header_separator_bboxes tight_bboxes origins calibrated_char_bboxes".split()
        out = sdk.prepare_native_container_table_page(**{k: v for k, v in q.items() if k in keys})
    same(q, before, "ownership")
    return out


def run(job):
    kind, c = job
    same(invoke(kind, c["request"]), c["expected"], kind + "/" + c["label"])


jobs = []
for name, kind, n in [
    ("native-index-containers", "index", 213),
    ("native-form-selection-canonical", "forms", 88),
    ("native-form-map-keys", "forms", 6),
    ("native-container-table-page-canonical", "page", 70),
]:
    f = read(name)
    assert len(f["cases"]) == n and not f["errors"]
    jobs.extend((kind, c) for c in f["cases"])
boundary = read("native-page-container-boundaries")["cases"]
assert len(boundary) == 37
jobs.extend((c["kind"], c) for c in boundary)
for job in jobs:
    run(job)
integer_limits = read("native-index-integer-limits")
assert len(integer_limits["cases"]) == 2 and len(integer_limits["errors"]) == 2
for c in integer_limits["cases"]:
    run(("index", c))
for c in integer_limits["errors"]:
    try:
        invoke("index", c["request"])
    except sdk.RapidDocError as e:
        assert e.code == c["error"]
    else:
        raise AssertionError(c["label"])
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::19]))
for kind, c in jobs:
    out = c["expected"]
    if kind == "index":
        assert set(out["claimed_line_indices"]).isdisjoint(
            l["source_index"] for l in out["remaining_lines"]
        )
    if kind == "page":
        assert (
            out["fixed_blocks"]
            == c["request"].get("prior_fixed_blocks", [])
            + out["rule_code_blocks"]
            + out["annotation_blocks"]
            + out["table_blocks"]
            + out["code_blocks"]
            + out["index_blocks"]
        )
        assert set(out["claimed_line_indices"]).isdisjoint(
            l["source_index"] for l in out["remaining_lines"]
        )
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster median_glyph_width ink_bbox run_index".split()
pdf = read("pdf-page-containers")["cases"]
assert len(pdf) == 75
assert sum(bool(c["expected"]["index_blocks"]) for c in pdf) == 4
assert sum(bool(c["expected"]["table_blocks"]) for c in pdf) == 41
assert (
    sum(bool(c["expected"]["rule_code_blocks"] or c["expected"]["code_blocks"]) for c in pdf) == 3
)


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    expected = c["request"]["source"]
    keys = [
        k
        for k in fields
        if k not in ["median_glyph_width", "ink_bbox", "run_index"]
        or expected["lines"]
        and k in expected["lines"][0]
    ]
    source = dict(
        page_size=g["page_size"],
        lines=[
            {k: l.get(k, False if k == "compact_formula_cluster" else None) for k in keys}
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
    same(source, expected, "live source")
    q = dict(c["request"], source=source)
    if "tight_bboxes" in q:
        q["tight_bboxes"] = {str(ch["char_idx"]): ch.get("tight_bbox") for ch in g["chars"]}
    if "origins" in q:
        q["origins"] = {str(ch["char_idx"]): ch.get("origin") for ch in g["chars"]}
    same(invoke("page", q), c["expected"], c["label"])


for c in pdf:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf[::31]))


def rejects(kind, q, code):
    try:
        invoke(kind, q)
    except sdk.RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(code)


for kind in ["index", "forms", "page"]:
    base = next(c["request"] for k, c in jobs if k == kind)
    q = copy.deepcopy(base)
    s = q if kind == "index" else q["source"]
    s["lines"] = [{}] * 4097
    rejects(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    s = q if kind == "index" else q["source"]
    s["lines"][1]["source_index"] = s["lines"][0]["source_index"]
    rejects(kind, q, "invalid_input")
    q = copy.deepcopy(base)
    s = q if kind == "index" else q["source"]
    s["page_size"] = [float("nan"), 600.0]
    try:
        invoke(kind, q)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite")
    if kind != "index":
        for field in ["path_infos", "drawing_lines"]:
            q = copy.deepcopy(base)
            q["source"][field] = [{}] * 5001
            rejects(kind, q, "resource_limit")
    mutated = invoke(kind, base)
    mutated.clear()
    run(next(job for job in jobs if job[0] == kind))
assert all(
    sdk.capabilities()[k]
    for k in ["native_index_containers", "native_form_selection", "native_container_table_page"]
)
assert (
    not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "242 old index + 104 Form selection + 70 continuous stage results, 2 old integer errors; 75 distinct actual PDF pages/4 index/41 table/3 code outputs; strict types/order/semantic changes/maps/ownership/concurrency/limits pass; real Form ownership extraction and image submission remain pending"
)
