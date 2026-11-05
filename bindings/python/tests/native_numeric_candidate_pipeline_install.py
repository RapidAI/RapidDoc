"""Installed five complete producers and combined page, with explicit CJK runtime boundary."""

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
    keys = ["source", "excluded_bboxes"] + (["kind"] if kind == "individual" else [])
    call = (
        sdk.detect_native_supplemental_tables
        if kind == "individual"
        else sdk.detect_native_structural_table_candidates
    )
    out = call(**{k: v for k, v in q.items() if k in keys})
    same(q, before, "ownership")
    return out


def run(job):
    kind, c = job
    same(invoke(kind, c["request"]), decode(c["expected"]), kind + "/" + c["label"])


jobs = []
for name, kind, count in [
    ("native-numeric-candidate-segments", "individual", 222),
    ("native-text-numeric-candidates", "individual", 144),
    ("native-text-numeric-page", "page", 1159),
]:
    f = read(name)
    assert len(f["cases"]) == count
    jobs.extend((kind, c) for c in f["cases"])
for job in jobs:
    run(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::41]))
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id dominant_font_weight paragraph_terminal semantic_type caption_start formula_candidate_only compact_formula_cluster".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = sdk.read_pdf_vector_geometry(data, page_index=c["page_index"])
    g = sdk.read_pdf_native_text_geometry(data, page_index=c["page_index"])
    lines = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    source = dict(
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
    same(source, c["request"]["source"], "live source")
    q = dict(c["request"], source=source)
    same(invoke("page", q), decode(c["expected"]["page"]), c["label"])
    if c["kind"]:
        same(
            invoke("individual", dict(q, kind=c["kind"])),
            decode(c["expected"]["individual"]),
            c["label"] + "/individual",
        )


pdf = read("pdf-numeric-candidates")
embedded = read("pdf-embedded-metric-candidate")
assert len(pdf["cases"]) == 46 and len(embedded["cases"]) == 1
actual = [c for c in pdf["cases"] if c["label"] != "metric_pair"] + embedded["cases"]
assert len(actual) == 46
for c in actual:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, actual[::11]))
# Frozen old-source comparison is NOT counted as live CID geometry parity.
cid = next(c for c in pdf["cases"] if c["label"] == "metric_pair")
same(
    invoke("individual", dict(cid["request"], kind="metric_pair")),
    decode(cid["expected"]["individual"]),
    "CID old-source candidate only",
)


def error(kind, q, code):
    try:
        invoke(kind, q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = jobs[0][1]["request"]
for kind in ["individual", "page"]:
    q = copy.deepcopy(base)
    q["source"]["lines"] = [{}] * 4097
    error(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["drawing_lines"] = [{}] * 5001
    error(kind, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["page_size"] = [0, 600]
    error(kind, q, "invalid_input")
error("individual", dict(base, kind="unknown"), "invalid_input")
for job in [jobs[0], jobs[222]]:
    kind, c = job
    out = invoke(kind, c["request"])
    out["candidates"][0]["bbox"][0] = -999
    for a in out["candidates"][0]["annotations"]:
        for b in a["line_bboxes"].values():
            b[0] = -999
    run(job)
assert (
    sdk.capabilities()["native_numeric_candidate_pipeline"]
    and sdk.capabilities()["native_fixed_cjk_font_runtime"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "1525 old producer/joint facts; 46 live PDF pages excluding the separately tested unembedded CID page; exact types/order, glyph gates, ownership, concurrency, limits/recovery passed; pre-policy regression scope, default PDF pending"
)
