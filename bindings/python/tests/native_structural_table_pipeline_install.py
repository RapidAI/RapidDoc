"""Installed full ordinary producer, five supplemental branches, and joint page arbitration."""

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
        assert len(a) == len(b), path
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
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def invoke(kind, q):
    q = copy.deepcopy(q)
    before = copy.deepcopy(q)
    if kind == "ordinary":
        out = sdk.detect_native_rule_table_candidates(
            **{
                k: v
                for k, v in q.items()
                if k in ["source", "angle", "median_height", "excluded_bboxes"]
            }
        )
    elif kind == "supplemental":
        out = sdk.detect_native_supplemental_tables(
            **{k: v for k, v in q.items() if k in ["source", "kind", "excluded_bboxes"]}
        )
    else:
        out = sdk.detect_native_structural_table_candidates(
            **{k: v for k, v in q.items() if k in ["source", "excluded_bboxes"]}
        )
    same(q, before, "ownership")
    return out


def run(job):
    kind, c = job
    same(invoke(kind, c["request"]), decode(c["expected"]), kind + "/" + c["label"])


corpora = [
    ("ordinary", "native-rule-interval-pipeline", 230),
    ("supplemental", "native-supplemental-rule-tables", 146),
    ("supplemental", "native-shaded-table-pipeline", 120),
    ("supplemental", "native-form-table-pipeline", 117),
    ("supplemental", "native-blank-answer-positive", 48),
    ("page", "native-structural-table-expanded-page", 793),
]
jobs = []
for kind, name, count in corpora:
    f = read(name)
    assert len(f["cases"]) == count
    jobs.extend((kind, c) for c in f["cases"])
for job in jobs:
    run(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::53]))
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id".split()


def live(c, structural):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    lines = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    selected = fields + (
        ["dominant_font_weight", "paragraph_terminal", "semantic_type", "caption_start"]
        if structural
        else []
    )
    source = dict(
        page_size=g["page_size"],
        lines=[{k: l[k] for k in selected} for l in lines],
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
    if structural:
        same(invoke("page", q), decode(c["expected"]["page"]), c["label"])
        if c["kind"]:
            same(
                invoke("supplemental", dict(q, kind=c["kind"])),
                decode(c["expected"]["individual"]),
                c["label"] + "/individual",
            )
    else:
        same(invoke("ordinary", q), decode(c["expected"]["ordinary"]), c["label"])


for name, structural, count in [
    ("pdf-rule-interval-pipeline", False, 36),
    ("pdf-structural-tables", True, 41),
]:
    f = read(name)
    assert len(f["cases"]) == count
    for c in f["cases"]:
        live(c, structural)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda c: live(c, structural), f["cases"][::13]))
base = copy.deepcopy(jobs[0][1]["request"])


def error(kind, q, code):
    try:
        invoke(kind, q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


for kind in ["ordinary", "supplemental", "page"]:
    q = copy.deepcopy(base)
    q["kind"] = "short_caption"
    q["source"]["lines"] = [{}] * 4097
    error(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    q["kind"] = "short_caption"
    q["source"]["drawing_lines"] = [{}] * 5001
    error(kind, q, "resource_limit")
error("supplemental", dict(base, kind="unknown"), "invalid_input")
error("ordinary", dict(base, median_height=0), "invalid_input")
for job in jobs[:1] + jobs[230:231]:
    kind, c = job
    out = invoke(kind, c["request"])
    out["candidates"][0]["bbox"][0] = -999
    run(job)
run(jobs[0])
assert (
    sdk.capabilities()["native_structural_table_pipeline"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "1454 old full producer/composition facts; ordinary 36 PDF queries and structural 41 PDF pages/28 positives; actual open/short/shaded/clipped positives, blank/stage-grid synthetic positives; exact types/order, ownership, concurrency, limits/recovery passed; remaining general producers and default PDF pending"
)
