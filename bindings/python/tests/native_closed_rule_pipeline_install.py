"""Installed complete closed physical grid branch and physical page orchestration."""

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
calls = dict(
    inspect=sdk.inspect_native_rule_table_geometry,
    detect=sdk.detect_native_closed_rule_tables,
    expand=sdk.expand_native_connected_table_grids,
    page=sdk.detect_native_physical_table_candidates,
)
names = dict(
    inspect="native-rule-page",
    detect="native-closed-rule-tables",
    expand="native-connected-table-grids",
    page="native-physical-table-page",
)


def decode(v):
    v = copy.deepcopy(v)
    for c in v.get("candidates", []):
        for a in c.get("annotations", []):
            a["line_bboxes"] = {int(k): b for k, b in a["line_bboxes"]}
    return v


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


def invoke(key, q):
    q = copy.deepcopy(q)
    if key == "page":
        q = {k: v for k, v in q.items() if k in ["source", "excluded_bboxes"]}
    else:
        if key != "expand":
            q.pop("candidates", None)
        if key == "inspect":
            q.pop("excluded_bboxes", None)
        if key == "expand":
            q["candidates"] = decode(dict(candidates=q["candidates"]))["candidates"]
    before = copy.deepcopy(q)
    result = calls[key](**q)
    same(q, before, "ownership")
    return result


def read(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


corpora = {key: read(name) for key, name in names.items()}
jobs = [(key, c) for key, f in corpora.items() for c in f["cases"]]
assert [len(corpora[k]["cases"]) for k in calls] == [129, 129, 129, 132]


def run(job):
    key, c = job
    same(invoke(key, c["request"]), decode(c["expected"]), key + "/" + c["label"])


for job in jobs:
    run(job)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, jobs[::19]))

pdf = read("pdf-closed-rule-pipeline")
assert len(pdf["cases"]) == 24
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars visual_row_id".split()


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
        lines=[{k: l[k] for k in fields} for l in lines],
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
    for key in calls:
        same(invoke(key, q), decode(c["expected"][key]), key + "/" + c["label"])


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::7]))

base = copy.deepcopy(corpora["detect"]["cases"][0]["request"])


def error(key, q, code):
    try:
        invoke(key, q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


for key in calls:
    q = copy.deepcopy(base)
    q["source"]["lines"] = [{}] * 4097
    error(key, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["drawing_lines"] = [{}] * 5001
    error(key, q, "resource_limit")
    q = copy.deepcopy(base)
    q["source"]["page_size"] = [0, 600]
    error(key, q, "invalid_input")
error("detect", dict(base, median_height=0), "invalid_input")
q = copy.deepcopy(base)
q["candidates"] = [{}] * 5001
error("expand", q, "resource_limit")
# Aggregate candidate members are capped even when no core is eligible for expansion.
q = copy.deepcopy(base)
q["candidates"] = [dict(core_bbox=None, line_indices=list(range(10001)), annotations=[])] * 2
error("expand", q, "resource_limit")
out = invoke("detect", base)
out["candidates"][0]["bbox"][0] = -999
for a in out["candidates"][0]["annotations"]:
    for b in a["line_bboxes"].values():
        b[0] = -999
same(
    invoke("detect", base),
    decode(corpora["detect"]["cases"][0]["expected"]),
    "independence/recovery",
)
assert (
    sdk.capabilities()["native_closed_rule_pipeline"]
    and not sdk.capabilities()["legacy_native_table_recovery"]
    and not sdk.capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "519 old closed physical chain facts; 24 live PDF pages x4 APIs (8 closed/16 physical positives); exact types/order, ownership, concurrency, limits/recovery passed; closed physical scope verified; ordinary/expanded structural coverage is recorded in native_structural_table_pipeline; default PDF pending"
)
