"""Installed materialized merge and ordered annotation dictionary bridge."""

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
    merge_native_table_candidates as merge,
    detect_native_filled_grid_tables as filled,
    detect_native_tiled_tables as tiled,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-table-candidate-merge.json").read_text(encoding="utf-8")
)
pdf = json.loads(
    (root / "build/migration/pdf-table-candidate-merge.json").read_text(encoding="utf-8")
)
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def decode(c):
    c = copy.deepcopy(c)
    for a in c["annotations"]:
        a["line_bboxes"] = {int(k): b for k, b in a["line_bboxes"]}
    return c


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        if path.endswith("/line_bboxes"):
            assert list(a) == list(b), (path, "insertion order")
        for k in a:
            same(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    raw = [decode(x) for x in c["request"]["candidates"]]
    before = copy.deepcopy(raw)
    out = merge(raw)
    expected = dict(candidates=[decode(x) for x in c["expected"]["candidates"]])
    same(out, expected, c["label"])
    same(raw, before, "ownership")
    out["candidates"].append({})
    same(raw, before, "output independence")


assert len(f["cases"]) == 158
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(pdf["cases"]) == 44


def queries(candidates, mode):
    base = copy.deepcopy(candidates)
    new = copy.deepcopy(candidates)
    if mode != "self":
        new = []
        for c in candidates:
            b = c.get("core_bbox") or c["bbox"]
            mid = (b[1] + b[3]) / 2
            for bounds in (
                ([b[0], b[1], b[2], mid],)
                if mode == "partial"
                else ([b[0], b[1], b[2], mid], [b[0], mid, b[2], b[3]])
            ):
                c = copy.deepcopy(c)
                for k in ["bbox", "local_bbox", "core_bbox"]:
                    c[k] = bounds
                new.append(c)
    return base + new


def encode(c):
    c = copy.deepcopy(c)
    for a in c["annotations"]:
        a["line_bboxes"] = [[int(k), v] for k, v in a["line_bboxes"].items()]
    return c


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    s = dict(
        page_size=g["page_size"],
        chars=g["chars"],
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        lines=[{k: l[k] for k in ["text", "bbox", "source_index", "chars"]} for l in rows],
    )
    same(s, c["source"], "live source")
    detected = (
        filled(v["path_infos"], page_size=v["page_size"]) if c["kind"] == "filled" else tiled(s)
    )
    raw = queries(detected["candidates"], c["mode"])
    same([encode(x) for x in raw], c["request"]["candidates"], "live detected query")
    before = copy.deepcopy(raw)
    same(merge(raw), dict(candidates=[decode(x) for x in c["expected"]["candidates"]]), c["label"])
    same(raw, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::6]))


def error(raw, code):
    try:
        merge(raw)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


c = dict(
    bbox=[0.0, 0.0, 100.0, 100.0],
    local_bbox=[0.0, 0.0, 100.0, 100.0],
    core_bbox=None,
    angle=0,
    score=100.0,
    line_indices=[],
    annotations=[],
)
error([c] * 5001, "resource_limit")
error([dict(c, angle=i) for i in range(4700)], "resource_limit")
error([dict(c, line_indices=[0] * 20001)], "resource_limit")
for raw in [[dict(c, bbox=[0, 1, 2])], [dict(c, score=None)]]:
    error(raw, "invalid_input")
# Mutating nested returned annotation boxes must not alias the original dictionaries.
ref = next(x for x in f["cases"] if x["label"] == "same-kind-union")
raw = [decode(x) for x in ref["request"]["candidates"]]
before = copy.deepcopy(raw)
out = merge(raw)
a = out["candidates"][0]["annotations"][0]
next(iter(a["line_bboxes"].values()))[0] = -999
same(raw, before, "deep independence")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_table_candidate_merge"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "158 old materialized merge references/142 changed; 28 PDF pages/44 live detector-seeded queries/24 changed; annotation order/body priority/types/metadata/ownership/concurrency/budgets/recovery passed; lazy draft materialization/general candidate producers/diagram/default PDF pending"
)
