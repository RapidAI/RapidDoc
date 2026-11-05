"""Installed wheel: old annotation transactions and front-panel rollback side effects."""

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
from rapid_doc_rust import (
    prepare_native_table_annotations as prepare,
    restore_native_front_matter_panel as restore,
    prepare_native_table_block as prior,
    read_pdf_native_text_geometry,
    read_pdf_vector_geometry,
    build_native_table_text,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()


def load(name):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    return f


def exact(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


f = load("native-table-annotations")
typed = load("native-table-annotations-types")
assert len(f["cases"]) == 904 and len(typed["cases"]) == 24 and len(f["front"]) == 240
assert sum(bool(c["expected"]["annotation_blocks"]) for c in f["cases"]) == 327
assert sum(c["expected"]["restore"] for c in f["front"]) == 102
assert (
    sum(
        len(c["expected"]["annotation_blocks"])
        > len(c["expected"]["candidate"]["annotations"]) - len(c["expected"]["failed_annotations"])
        for c in f["cases"]
    )
    == 141
)
assert (
    sum(
        any(b["_output_bbox_repaired"] for b in c["expected"]["annotation_blocks"])
        for c in f["cases"]
    )
    == 183
)


def case(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    r = prepare(**q)
    exact(r, c["expected"], c["label"])
    exact(q, before, "ownership")
    assert not set(r["projection_line_indices"]) & set(r["externalized_line_indices"])
    return r


def front(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    exact(restore(**q), c["expected"], c["label"])
    exact(q, before, "front ownership")


for c in f["cases"] + typed["cases"]:
    case(c)
for c in f["front"]:
    front(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(case, f["cases"][::23] + typed["cases"]))
    list(pool.map(front, f["front"][::11]))
for meta in f["real"]:
    assert hashlib.sha256((root / meta["pdf"]).read_bytes()).hexdigest() == meta["pdf_sha256"]
    c = f["cases"][meta["case_index"]]
    q = copy.deepcopy(c["request"])
    g = read_pdf_native_text_geometry(root / meta["pdf"], page_index=meta["page_index"])
    vectors = read_pdf_vector_geometry(root / meta["pdf"], page_index=meta["page_index"])
    assert len(g["chars"]) == len(q["source"]["chars"])
    chars = [{k: new[k] for k in old} for new, old in zip(g["chars"], q["source"]["chars"])]
    if q["source"]["lines"]:
        text = build_native_table_text(
            chars, table_bbox=q["candidate"]["bbox"], angle=q["candidate"]["angle"]
        )
        groups = {}
        by = {c["char_idx"]: c for c in chars}
        lines = []
        for glyph in text["glyphs"]:
            groups.setdefault(glyph["visual_row"], []).append(glyph)
        for row, gs in groups.items():
            bs = [by[glyph["source_index"]]["bbox"] for glyph in gs]
            box = [
                min(b[0] for b in bs),
                min(b[1] for b in bs),
                max(b[2] for b in bs),
                max(b[3] for b in bs),
            ]
            lines.append(
                dict(
                    text="".join(glyph["text"] for glyph in gs),
                    bbox=box,
                    source_index=row,
                    angle=q["candidate"]["angle"],
                )
            )
        exact(lines, q["source"]["lines"], "live glyph visual lines")
        q["source"]["lines"] = lines
    q["source"].update(
        chars=chars, drawing_lines=vectors["drawing_lines"], page_size=g["page_size"]
    )
    exact(prepare(**q), c["expected"], "live PDF annotation preparation negative control")
base = copy.deepcopy(f["cases"][0]["request"])


def rejects(q):
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == "resource_limit", e
    else:
        raise AssertionError("resource budget reset or ignored")


# Source preparation and annotation output each fit; combined costs cannot reset the outer Work.
q = copy.deepcopy(base)
q["source"].update(
    chars=[dict(char="A", char_idx=i, font=dict(name=f"Font{i}")) for i in range(4000)],
    lines=[
        dict(text="body", source_index=i, angle=0, bbox=[10.0, 10.0, 20.0, 20.0])
        for i in range(1200)
    ],
    drawing_lines=[],
)
ann = dict(kind="footnote", bbox=[0.0, 0.0, 100.0, 100.0], line_indices=[0], line_bboxes={})
q["candidate"].update(
    annotations=[copy.deepcopy(ann) for _ in range(1600)],
    angle=0,
    bbox=[0.0, 0.0, 100.0, 100.0],
    core_bbox=None,
    line_indices=[],
)
assert prior(q["source"], q["candidate"])
a = copy.deepcopy(q)
a["source"]["chars"] = []
assert prepare(**a)
rejects(q)
q = copy.deepcopy(base)
q["candidate"]["annotations"] = [{}] * 2001
rejects(q)
case(f["cases"][0])
front(f["front"][1])
assert (
    capabilities()["native_table_annotations"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (
    Path(__import__("rapid_doc_rust").__file__).parent / "licenses/MIGRATION-NOTICES.txt"
).is_file()
print(
    "Installed wheel: 904 old annotation transactions +24 coordinate/height type facts (327 output, 141 split, 183 tight bbox cases); 240 front panels/102 restorations, 16 live PDF negative controls; exact metadata, ownership/concurrency/shared budget; annotation helper regression scope; full page/default PDF pending"
)
