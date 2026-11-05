"""Installed complete table detector and final page policies against immutable old references."""

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
from rapid_doc_rust import RapidDocError

root = Path(sys.argv[1]).resolve()


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        if path.endswith("/line_bboxes"):
            assert list(a) == list(b), path
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def decode(v):
    v = copy.deepcopy(v)
    for c in v["candidates"]:
        for a in c["annotations"]:
            a["line_bboxes"] = {int(k): b for k, b in a["line_bboxes"]}
    return v


def read(name):
    data = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in data["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return data


def invoke(kind, q):
    q = copy.deepcopy(q)
    before = copy.deepcopy(q)
    if kind == "policy":
        q["candidates"] = decode(dict(candidates=q["candidates"]))["candidates"]
        before = copy.deepcopy(q)
        out = sdk.apply_native_table_page_policies(**q)
    elif kind == "trailing":
        out = sdk.trailing_native_table_grid(**q)
    else:
        out = sdk.detect_native_table_candidates(
            **{k: q[k] for k in ["source", "excluded_bboxes"] if k in q}
        )
    same(q, before, "ownership")
    return out


def run(job):
    kind, c = job
    same(
        invoke(kind, c["request"]),
        c["expected"] if kind == "trailing" else decode(c["expected"]),
        kind + "/" + c["label"],
    )


def error(kind, q, code):
    try:
        invoke(kind, q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


jobs = []
errors = 0
for name, kind, count in [
    ("native-table-page-policies", "policy", 75),
    ("native-table-page-policy-boundaries", "policy", 28),
    ("native-complete-table-candidates", "page", 1404),
    ("native-trailing-table-grid", "trailing", 26),
]:
    data = read(name)
    assert len(data["cases"]) == count
    jobs.extend((kind, c) for c in data["cases"])
    for e in data.get("errors", []):
        error(kind, e["request"], e["error"])
        errors += 1
for job in jobs:
    run(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::61]))
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    lines = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    s = dict(
        page_size=g["page_size"],
        lines=[
            {k: l.get(k, False if k == "compact_formula_cluster" else None) for k in fields}
            for l in lines
        ],
        chars=g["chars"],
        drawing_lines=v["drawing_lines"],
        path_infos=v["path_infos"],
        form_bboxes=[],
        image_bboxes=[],
        signature_bboxes=[],
        page_index=c["page_index"],
    )
    same(s, c["request"]["source"], "live source")
    same(invoke("page", dict(c["request"], source=s)), decode(c["expected"]), c["label"])


pdf = read("pdf-complete-table-candidates")["cases"]
assert len(pdf) == 55
for c in pdf:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf[::17]))
positive = read("pdf-table-page-policies-positive")["cases"]
assert len(positive) == 3 and sum(c["complete_changed"] for c in positive) == 2
for c in positive:
    v = copy.deepcopy(c)
    v["expected"] = v["complete_expected"]
    live(v)
    run(("policy", c))
policy = jobs[0][1]["request"]
page = next(c["request"] for kind, c in jobs if kind == "page")
for kind, base in [("policy", policy), ("page", page)]:
    for field, count in [("lines", 4097), ("drawing_lines", 5001)]:
        q = copy.deepcopy(base)
        q["source"][field] = [{}] * count
        error(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["lines"] = [
        dict(text="x", bbox=[0, 0, 1, 1], source_index=0, angle=0, effective_height=1.0)
    ]
    q["source"]["lines"][0]["bbox"][0] = "bad"
    error(kind, q, "invalid_input")
error("policy", dict(policy, operation="unknown"), "invalid_input")
for kind, c in jobs:
    if kind != "trailing" and c["expected"]["candidates"]:
        result = invoke(kind, c["request"])
        result["candidates"][0]["bbox"][0] = -999
        run((kind, c))
        break
assert errors == 3
assert (
    sdk.capabilities()["native_complete_table_candidates"]
    and sdk.capabilities()["native_table_page_policies"]
)
assert (
    sdk.capabilities()["native_fixed_cjk_font_runtime"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "103 final policy + 1404 original complete detector + 26 terminal grid facts; 3 old exceptions; 58 live PDF pages (3 positive policy projections, 2 complete detector changes); strict types/order, ownership/concurrency/limits passed; default PDF remains pending"
)
