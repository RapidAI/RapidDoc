"""Installed complete code builders and rule-code/table/filled-code preparation order."""

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


def invoke(kind, q):
    before = copy.deepcopy(q)
    if kind == "code":
        out = sdk.build_native_code_blocks(
            **{
                k: v
                for k, v in q.items()
                if k in ["source", "kind", "excluded_bboxes", "claimed_line_indices"]
            }
        )
    else:
        keys = "source excluded_bboxes prior_fixed_blocks prior_claimed_line_indices table_header_separator_bboxes tight_bboxes origins calibrated_char_bboxes".split()
        out = sdk.prepare_native_code_table_page(**{k: v for k, v in q.items() if k in keys})
    same(q, before, "ownership")
    return out


def run(job):
    kind, c = job
    same(invoke(kind, c["request"]), c["expected"], kind + "/" + c["label"])


jobs = []
for name, kind, n in [
    ("native-code-containers", "code", 197),
    ("native-code-container-boundaries", "code", 26),
    ("native-code-table-page", "page", 60),
    ("native-code-table-grid-mixtures", "page", 3),
]:
    f = read(name)
    assert len(f["cases"]) == n and not f["errors"]
    jobs.extend((kind, c) for c in f["cases"])
integer_cases = json.loads(
    (root / "build/migration/native-code-integer-boundaries.json").read_text(encoding="utf-8")
)["cases"]
assert len(integer_cases) == 4
jobs.extend(("code", c) for c in integer_cases)
for job in jobs:
    run(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::19]))
for kind, c in jobs:
    out = c["expected"]
    if kind == "code":
        assert out["claimed_line_indices"] == sorted(set(out["claimed_line_indices"]))
    else:
        assert (
            out["fixed_blocks"]
            == c["request"].get("prior_fixed_blocks", [])
            + out["rule_code_blocks"]
            + out["annotation_blocks"]
            + out["table_blocks"]
            + out["code_blocks"]
        )
        assert {l["source_index"] for l in out["remaining_lines"]}.isdisjoint(
            out["claimed_line_indices"]
        )
mixed = read("native-code-table-grid-mixtures")["cases"]
assert all(
    c["expected"]["table_blocks"]
    and (c["expected"]["rule_code_blocks"] or c["expected"]["code_blocks"])
    for c in mixed
)
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()


def live(job):
    kind, c = job
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    keys = fields + (
        ["median_glyph_width"]
        if c["request"]["source"]["lines"]
        and "median_glyph_width" in c["request"]["source"]["lines"][0]
        else []
    )
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
    same(source, c["request"]["source"], "live source")
    q = dict(c["request"], source=source)
    if "tight_bboxes" in q:
        q["tight_bboxes"] = {str(ch["char_idx"]): ch.get("tight_bbox") for ch in g["chars"]}
    if "origins" in q:
        q["origins"] = {str(ch["char_idx"]): ch.get("origin") for ch in g["chars"]}
    same(invoke(kind, q), c["expected"], c["label"])


pdf = read("pdf-code-containers")["cases"]
page = read("pdf-code-table-page")["cases"]
assert len(pdf) == 16 and len(page) == 67
assert sum(bool(c["expected"]["code_blocks"]) for c in pdf) == 3
assert (
    sum(bool(c["expected"]["rule_code_blocks"] or c["expected"]["code_blocks"]) for c in page) == 3
)
assert sum(bool(c["expected"]["table_blocks"]) for c in page) == 41
for job in [("code", c) for c in pdf] + [("page", c) for c in page]:
    live(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, [("page", c) for c in page[::23]]))


def rejects(kind, q, code):
    try:
        invoke(kind, q)
    except sdk.RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(code)


for kind in ["code", "page"]:
    base = next(c["request"] for k, c in jobs if k == kind)
    for field, n in [("lines", 4097), ("drawing_lines", 5001)]:
        q = copy.deepcopy(base)
        q["source"][field] = [{}] * n
        rejects(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["lines"][1]["source_index"] = q["source"]["lines"][0]["source_index"]
    rejects(kind, q, "invalid_input")
q = copy.deepcopy(jobs[0][1]["request"])
q["kind"] = "unknown"
rejects("code", q, "invalid_input")
base = jobs[0][1]["request"]
out = invoke("code", base)
out["code_blocks"][0]["content"] = "mutated"
run(jobs[0])
assert (
    sdk.capabilities()["native_code_containers"]
    and sdk.capabilities()["native_code_table_page"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "227 full old code-builder (including 4 integer boundaries) + 63 code/table stage facts; 67 distinct live PDF pages, 16 standalone builder queries, 3 code/41 table page outputs; mixed code+physical-grid retention, blank rows/Unicode/lanes/claims/types/order/ownership/concurrency/limits passed; full page/default PDF remains pending"
)
