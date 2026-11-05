"""Installed complete automatic table-note collector, live glyph/text queries and no note verdict input."""

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
    collect_native_table_footnotes as collect,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-table-footnotes.json").read_text(encoding="utf-8"))
real = []
extra = json.loads(
    (root / "build/migration/native-table-footnote-boundaries.json").read_text(encoding="utf-8")
)
for corpus in [f, extra] + [
    json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))
    for n in ["pdf-table-footnotes", "pdf-table-footnotes-closeaux"]
]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    if corpus is not f and corpus is not extra:
        real += corpus["cases"]
assert (
    hashlib.sha256(
        (root / "rust/rapiddoc/src/legacy_model/table_footnotes/lns.json").read_bytes()
    ).hexdigest()
    == f["category_sha256"]
)


def decode(v):
    v = copy.deepcopy(v)
    if v["annotation"] is not None:
        v["annotation"]["line_bboxes"] = {int(k): b for k, b in v["annotation"]["line_bboxes"]}
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


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = collect(**q)
    same(out, decode(c["expected"]), c["label"])
    same(q, before, "ownership")
    out["footnote_rows"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 155 and len(extra["cases"]) == 22
for c in f["cases"] + extra["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(real) == 20
fields = "text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars".split()


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    lines = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    frame = v["path_infos"][0]["bbox"]
    rows = []
    core = []
    for l in lines:
        b = l["bbox"]
        cy = (b[1] + b[3]) / 2
        cx = (b[0] + b[2]) / 2
        if frame[0] <= cx <= frame[2] and frame[1] <= cy <= frame[3]:
            core.append(l["source_index"])
        rows.append(
            dict(
                fragments=[
                    dict(
                        text=l["text"],
                        bbox=b,
                        local_bbox=b,
                        line_index=l["source_index"],
                        visual_row_id=None,
                    )
                ],
                center_y=cy,
                bbox=b,
                visual_row_id=l["visual_row_id"],
            )
        )
    q = dict(
        rows=rows,
        lines=[{k: l[k] for k in fields} for l in lines],
        rule_bbox=frame,
        median_height=10.0,
        core_line_indices=core,
        page_size=g["page_size"],
        angle=0,
    )
    same(q, c["request"], "live query")
    before = copy.deepcopy(q)
    same(collect(**q), decode(c["expected"]), c["label"])
    same(q, before, "live ownership")


for c in real:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, real[::4]))


def error(q, code):
    try:
        collect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = copy.deepcopy(f["cases"][0]["request"])
error(dict(base, rows=[{}] * 4097), "resource_limit")
error(dict(base, lines=[{}] * 4097), "resource_limit")
error(dict(base, core_line_indices=[0] * 20001), "resource_limit")
q = copy.deepcopy(base)
template = q["lines"][1]
q["lines"] = [
    dict(template, source_index=i + 10, font_signature=[f"UniqueFont{i}", 0]) for i in range(1500)
]
frag = q["rows"][1]["fragments"][0]
r = copy.deepcopy(q["rows"][1])
r["fragments"] = [dict(frag, line_index=i + 10) for i in range(1500)]
q["rows"] = [r]
q["core_line_indices"] = []
error(q, "resource_limit")
for q in [dict(base, page_size=[1]), dict(base, median_height=0), dict(base, angle=None)]:
    error(q, "invalid_input")
q = copy.deepcopy(base)
for l in q["lines"]:
    l["angle"] = 0.0
same(collect(**q), decode(f["cases"][0]["expected"]), "numeric line angle")
q = copy.deepcopy(base)
before = copy.deepcopy(q)
out = collect(**q)
next(iter(out["annotation"]["line_bboxes"].values()))[0] = -999
same(q, before, "deep independence")
run(f["cases"][0])
live(real[0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_table_footnotes"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "177 complete old automatic-note/angle/chain references/129 positive plus 75 marker facts in Rust; 20 live PDF queries/6 positive including compact and superscript reference; Unicode16/ordered boxes/types/ownership/concurrency/budgets/recovery passed; full candidate materialization/default PDF pending"
)
