"""Installed old diagram component chain; source positives and explicit negative-only PDF coverage."""

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
        assert len(a) == len(b), path
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
    out = sdk.detect_native_diagram_components(**q)
    same(q, before, "ownership")
    return out


def run(c):
    same(invoke(c["request"]), c["expected"], c["label"])


jobs = []
errors = []
for name, count in [("native-diagram-components", 200), ("native-diagram-component-specials", 26)]:
    f = read(name)
    assert len(f["cases"]) == count
    jobs += f["cases"]
    errors += f["errors"]
for c in jobs:
    run(c)
for c in errors:
    try:
        invoke(c["request"])
    except sdk.RapidDocError as e:
        assert e.code == c["error"], c["label"]
    else:
        raise AssertionError(c["label"])
assert len(errors) == 11
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::13]))
pdf = read("pdf-diagram-components")["cases"]
assert len(pdf) == 55
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    s = dict(
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
    same(s, c["request"]["source"], "live source")
    same(invoke(dict(source=s, em=c["request"]["em"])), c["expected"], c["label"])


for c in pdf:
    live(c)
assert not any(c["expected"]["diagram_component_bboxes"] for c in pdf)
base = copy.deepcopy(jobs[0]["request"])
for k, n in [("chars", 20001), ("lines", 4097), ("path_infos", 5001)]:
    q = copy.deepcopy(base)
    q["source"][k] = [{}] * n
    try:
        invoke(q)
    except sdk.RapidDocError as e:
        assert e.code == "resource_limit", (k, e.code)
    else:
        raise AssertionError(k)
q = copy.deepcopy(base)
q["em"] = 0
try:
    invoke(q)
except sdk.RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid em")
run(jobs[0])
result = invoke(jobs[0]["request"])
result["diagram_component_bboxes"][0][0] = -999
run(jobs[0])
assert (
    sdk.capabilities()["native_diagram_components"]
    and sdk.capabilities()["native_fixed_cjk_font_runtime"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "226 full old diagram facts + 11 old exceptions; table barriers, prose/photo/timeline positives; 55 actual PDF negative-only source reconstructions; types/order/ownership/concurrency/limits/recovery passed; default PDF remains pending"
)
