"""Installed rule core/caption/annotation preparation with ordered source-box output."""

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
    prepare_native_rule_table_annotations as prepare,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-rule-annotation-preparation.json").read_text(encoding="utf-8")
)
pdf = json.loads(
    (root / "build/migration/pdf-rule-annotation-preparation.json").read_text(encoding="utf-8")
)
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def decode(v):
    v = copy.deepcopy(v)
    for a in v["annotations"]:
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


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    out = prepare(**q)
    same(out, decode(c["expected"]), c["label"])
    same(q, before, "ownership")
    out["annotations"].append({})
    same(q, before, "output independence")


assert len(f["cases"]) == 136
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(pdf["cases"]) == 12


def live(c):
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    v = vectors(data, page_index=c["page_index"])
    g = geometry(data, page_index=c["page_index"])
    lines = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    frame = v["path_infos"][0]["bbox"]
    rows = []
    core = []
    caption = None
    foot = []
    for i, l in enumerate(lines):
        b = l["bbox"]
        cy = (b[1] + b[3]) / 2
        cx = (b[0] + b[2]) / 2
        if frame[0] <= cx <= frame[2] and frame[1] <= cy <= frame[3]:
            core.append(i)
        if caption is None and l["text"].startswith("Table"):
            caption = l["source_index"]
        if l["text"].startswith("Note"):
            foot.append(i)
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
        rule_bboxes=[r["bbox"] for r in v["drawing_lines"]],
        core_row_indices=core,
        median_height=10.0,
        caption_source_index=caption,
        footnote_row_indices=foot,
    )
    same(q, c["request"], "live query")
    before = copy.deepcopy(q)
    same(prepare(**q), decode(c["expected"]), c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::3]))


def error(q, code):
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = copy.deepcopy(f["cases"][0]["request"])
error(dict(base, rows=[{}] * 4097), "resource_limit")
error(dict(base, rule_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
error(dict(base, core_row_indices=[0] * 4097), "resource_limit")
empty = dict(fragments=[], bbox=[0, 100, 100, 200], center_y=150, visual_row_id=None)
frag = dict(
    text="note",
    bbox=[0.0, 220.0, 10.0, 230.0],
    local_bbox=[0.0, 220.0, 10.0, 230.0],
    line_index=0,
    visual_row_id=None,
)
r = dict(
    fragments=[dict(frag, line_index=i) for i in range(6000)],
    bbox=[0.0, 220.0, 10.0, 230.0],
    center_y=225.0,
    visual_row_id=None,
)
error(
    dict(
        base,
        rows=[empty, r],
        core_row_indices=[0],
        caption_source_index=None,
        footnote_row_indices=[1],
    ),
    "resource_limit",
)
for q in [
    dict(base, rule_bboxes=[]),
    dict(base, core_row_indices=[]),
    dict(base, core_row_indices=[999]),
    dict(base, median_height=0.0),
]:
    error(q, "invalid_input")
ref = next(c for c in f["cases"] if c["label"] == "base")
q = copy.deepcopy(ref["request"])
before = copy.deepcopy(q)
out = prepare(**q)
next(iter(out["annotations"][0]["line_bboxes"].values()))[0] = -999
same(q, before, "deep independence")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_rule_annotation_preparation"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "136 old core/caption/annotation preparation references (124 caption rows/136 annotations); 12 live PDF queries (2 caption/7 annotation positive); ordered source boxes/local-vs-original/body exclusion/types/ownership/concurrency/budgets/recovery passed; footnote selection explicit, automatic notes/full draft materialization pending"
)
