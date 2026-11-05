"""Installed closed-grid proof over explicit queries; live rules and source-ID membership."""

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
    inspect_native_closed_grids as inspect,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-closed-grids.json").read_text(encoding="utf-8"))
pdf = json.loads((root / "build/migration/pdf-closed-grids.json").read_text(encoding="utf-8"))
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
    out = inspect(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["closed_grid_flags"].append(False)
    same(q, before, "output independence")


assert len(f["cases"]) == 176
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::9]))
q = copy.deepcopy(f["cases"][0]["request"])
q.pop("allow_two_by_two")
same(inspect(**q), f["cases"][0]["expected"], "default mode")
assert len(pdf["cases"]) == 32 and len({c["page_index"] for c in pdf["cases"]}) == 16


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    lines = [{k: l[k] for k in ["bbox", "source_index", "effective_height"]} for l in rows]
    p = v["path_infos"][0]
    frame = p["rectangle_bboxes"][0] if p["rectangle_bboxes"] else p["bbox"]
    q = dict(
        source=dict(form_bboxes=[], drawing_lines=v["drawing_lines"], lines=lines),
        candidates=[
            dict(
                bbox=frame, core_bbox=None, angle=0, line_indices=[l["source_index"] for l in rows]
            )
        ],
        allow_two_by_two=c["request"]["allow_two_by_two"],
    )
    same(q, c["request"], "live frame/IDs query")
    before = copy.deepcopy(q)
    same(inspect(**q), c["expected"], c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::5]))


def error(q, code):
    try:
        inspect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


assert len(f["errors"]) == 2
for c in f["errors"]:
    error(c["request"], c["expected_code"])
base = dict(source={}, candidates=[])
error(dict(base, candidates=[{}] * 5001), "resource_limit")
for key, n, value in [
    ("lines", 4097, {}),
    ("drawing_lines", 5001, {}),
    ("form_bboxes", 5001, [0, 0, 1, 1]),
]:
    error(dict(source={key: [value] * n}, candidates=[]), "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["candidates"][0]["line_indices"] = [0] * 20001
error(q, "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["candidates"] = [dict(q["candidates"][0], line_indices=[0] * 11000)] * 2
error(q, "resource_limit")
q = copy.deepcopy(f["cases"][0]["request"])
q["source"]["lines"] = [
    dict(bbox=[60.0, 110.0, 80.0, 120.0], source_index=0, effective_height=10.0)
] * 1024
q["candidates"] = [dict(q["candidates"][0], line_indices=[0])] * 300
error(q, "resource_limit")
error(dict(base, allow_two_by_two=1), "invalid_input")
error(dict(source={}, candidates=[dict(bbox=[0, 1, 2], angle=0)]), "invalid_input")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_closed_grids"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "176 original closed-grid references/117 positive, 2 old median errors preserved; 16 actual PDF pages/32 live frame+source-ID queries/7 positive; exact flags/order/ownership/concurrency/budgets/recovery passed; closed-grid helper regression scope; real Form extraction/default PDF pending"
)
