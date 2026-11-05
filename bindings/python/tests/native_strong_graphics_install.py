"""Installed actual strong-graphic orchestration and source side effects."""

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
    out = sdk.detect_native_strong_graphics(**q)
    same(q, before, "ownership")
    return out


def run(c):
    same(invoke(c["request"]), c["expected"], c["label"])


facts = read("native-strong-graphics-complete")
assert len(facts["cases"]) == 493 and len(facts["errors"]) == 37
for c in facts["cases"]:
    run(c)
for c in facts["errors"]:
    try:
        invoke(c["request"])
    except sdk.RapidDocError as e:
        assert e.code == c["error"], (c["label"], e.code)
    else:
        raise AssertionError(c["label"])
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, facts["cases"][::23]))
mutation = next(c for c in facts["cases"] if c["request"]["source"] != c["expected"]["source"])
out = invoke(mutation["request"])
assert any(l.get("semantic_type") == "caption" for l in out["source"]["lines"])
run(mutation)
# Raster detection must not run the separate paragraph-description grouping helper.
raster = next(
    c
    for c in facts["cases"]
    if c["label"].startswith("native-raster-charts/") and c["expected"]["graphic_bboxes"]
)
same(
    invoke(raster["request"])["source"],
    raster["request"]["source"],
    "raster core does not group descriptions",
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
    same(s, c["request"]["source"], "actual source")
    same(invoke(dict(source=s)), c["expected"], c["label"])


pdf = read("pdf-strong-graphics")["cases"] + read("pdf-strong-graphics-positive")["cases"]
assert len(pdf) == 59 and sum(bool(c["expected"]["graphic_bboxes"]) for c in pdf) == 2
for c in pdf:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf[::17]))
base = copy.deepcopy(mutation["request"])
for field, count in [
    ("chars", 20001),
    ("lines", 4097),
    ("path_infos", 5001),
    ("image_bboxes", 5001),
]:
    q = copy.deepcopy(base)
    q["source"][field] = [{}] * count
    try:
        invoke(q)
    except sdk.RapidDocError as e:
        assert e.code == "resource_limit", (field, e.code)
    else:
        raise AssertionError(field)
q = copy.deepcopy(base)
q["source"]["lines"][0]["bbox"][0] = "invalid"
try:
    invoke(q)
except sdk.RapidDocError as e:
    assert e.code == "invalid_input"
else:
    raise AssertionError("invalid coordinate")
out = invoke(base)
out["graphic_bboxes"][0][0] = -999
out["source"]["lines"][0]["text"] = "changed"
run(mutation)
assert (
    sdk.capabilities()["native_strong_graphics"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "493 full old strong-graphic results (335 source positives) + 37 old errors; caption side effects and raster core-only behavior; 59 live PDF pages including 2 accepted/2 rejected contour rotations; ownership/types/order/concurrency/limits/recovery passed; default PDF remains pending"
)
