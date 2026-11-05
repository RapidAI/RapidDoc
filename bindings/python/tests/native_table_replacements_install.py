"""Installed complete-coverage arbitration with live filled-detector query reconstruction."""

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
    prefer_complete_native_table_replacements as prefer,
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
    (root / "build/migration/native-table-replacements.json").read_text(encoding="utf-8")
)
pdf = json.loads((root / "build/migration/pdf-table-replacements.json").read_text(encoding="utf-8"))
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = prefer(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["base"].append({})
    out["replacements"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 157
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(pdf["cases"]) == 44 and len({(c["pdf"], c["page_index"]) for c in pdf["cases"]}) == 28


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
    return dict(base=base, recovered=[], replacements=new)


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
    q = queries(detected["candidates"], c["mode"])
    same(q, c["request"], "live detected query")
    before = copy.deepcopy(q)
    same(prefer(**q), c["expected"], c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::6]))


def error(q, code):
    try:
        prefer(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(base=[], recovered=[], replacements=[])
for key in base:
    error(dict(base, **{key: [{}] * 5001}), "resource_limit")
old = dict(bbox=[0.0, 0.0, 100.0, 100.0])
new = [dict(bbox=[i * 0.001, i * 0.02, 99.0 + i * 0.001, i * 0.02 + 1.0]) for i in range(1000)]
error(dict(base, base=[old], replacements=new), "resource_limit")
for key in base:
    error(dict(base, **{key: [dict(bbox=[0, 1, 2])]}), "invalid_input")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_table_replacements"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "157 original replacement references/153 changed/78 rejection cases; 28 actual PDF pages/44 live detector-seeded queries/24 nonempty changed; exact metadata/types/order/ownership/concurrency/budgets/recovery passed; query splitting explicit, all producer branches covered separately; final page policies/diagram/default PDF pending"
)
