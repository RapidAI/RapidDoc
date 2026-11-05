"""Installed exact original occupied-grid protection and graphic-tail decisions."""

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
    arbitrate_native_graphic_candidates as select,
    inspect_native_graphic_grids as inspect,
    detect_native_drawing_graphics as candidates,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as chars,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
cases = []
for name in ["native-graphic-arbitration", "native-graphic-arbitration-edges"]:
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    cases.extend(f["cases"])


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
    out = select(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["graphic_bboxes"].append([0, 0, 1, 1])
    same(q, before, "output independence")
    ordered = sorted(
        q["candidate_bboxes"],
        key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]),
        reverse=True,
    )
    rejects = {
        r["sorted_candidate_index"]: r["grid_bboxes"] for r in c["expected"]["grid_rejections"]
    }
    same(
        inspect(q["source"], ordered, median_height=q["median_height"]),
        dict(grid_bboxes=[rejects.get(i, []) for i in range(len(ordered))]),
        "grid-only " + c["label"],
    )


assert len(cases) == 178
for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::13]))
pdf = json.loads(
    (root / "build/migration/pdf-graphic-arbitration.json").read_text(encoding="utf-8")
)
assert len(pdf["cases"]) == 8
for p, h in pdf["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    page = c["page_index"]
    g = chars(data, page_index=page)
    v = vectors(data, page_index=page)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    source = dict(
        page_size=g["page_size"],
        drawing_lines=v["drawing_lines"],
        form_bboxes=[],
        lines=[
            {k: l[k] for k in ["text", "bbox", "angle", "source_index", "visual_row_id"]}
            for l in rows
        ],
    )
    same(source, c["source"], "live source")
    generated = candidates(
        v["drawing_lines"], v["path_infos"], page_size=g["page_size"], median_height=10.0
    )["graphic_bboxes"]
    same(generated, c["candidate_bboxes"], "live candidates")
    same(
        select(source, generated, median_height=10.0),
        c["expected"],
        "live arbitration " + c["label"],
    )


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::3]))


def error(function, q, code):
    try:
        function(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source=dict(page_size=[400.0, 600.0]), median_height=10.0, candidate_bboxes=[])
error(select, dict(base, candidate_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(
    select,
    dict(base, source=dict(page_size=[400.0, 600.0], lines=[dict(bbox=[0, 0, 1, 1])] * 4097)),
    "resource_limit",
)
error(
    select,
    dict(
        base,
        source=dict(
            page_size=[400.0, 600.0],
            drawing_lines=[
                dict(bbox=[i * 100.0, 50.0, i * 100.0 + 100.0, 50.0], orientation="horizontal")
                for i in range(4600)
            ],
        ),
    ),
    "resource_limit",
)
error(
    select,
    dict(base, candidate_bboxes=[[i * 100.0, 0.0, i * 100.0 + 10.0, 10.0] for i in range(4600)]),
    "resource_limit",
)
error(
    inspect,
    dict(source=base["source"], median_height=10.0, regions=[[0, 0, 1, 1]] * 5001),
    "resource_limit",
)
for q in [
    dict(base, source=dict(page_size=[400.0])),
    dict(base, median_height=None),
    dict(base, candidate_bboxes=[[0, 0, 1]]),
]:
    error(select, q, "invalid_input")
run(cases[0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_graphic_arbitration"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
for name in [
    "MIGRATION-NOTICES.txt",
    "NUM-RATIONAL-LICENSE-MIT.txt",
    "NUM-TRAITS-LICENSE-APACHE.txt",
]:
    assert (Path(sdk.__file__).parent / "licenses" / name).is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "178 original tail/grid facts (120 grid rejection positive), 8 live real PDF text+lines+candidate+arbitration chains (1 table rejection positive); exact grid evidence/merge order/types/ownership/concurrency/budgets/recovery passed; arbitration helper regression scope; default PDF pending"
)
