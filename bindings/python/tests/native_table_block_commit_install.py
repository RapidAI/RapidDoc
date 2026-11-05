"""Installed wheel: complete old native table block transactions, without stage substitution."""

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
    materialize_native_table_blocks as materialize,
    build_native_table_text,
    read_pdf_native_text_geometry,
    read_pdf_vector_geometry,
    recover_native_table_primitives,
    render_native_table_html_with_scripts,
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


f = load("native-table-block-commit")
cache = load("native-table-block-commit-cache")
media = load("native-table-media")
assert len(f["cases"]) == 781 and len(cache["cases"]) == 1 and len(media["cases"]) == 96
assert sum(c["events"].get("native", False) for c in f["cases"]) == 192
assert sum(c["events"].get("projection", False) for c in f["cases"]) == 512
assert sum(c["events"].get("front_restore", False) for c in f["cases"]) == 102
assert sum(c["events"].get("annotation_rollback", False) for c in f["cases"]) == 109
assert sum(c["events"].get("media", False) for c in f["cases"]) == 24
assert sum(c["events"].get("recovery_error", False) for c in f["cases"]) == 4
assert len(f["excluded_engine"]) == 2 and len(cache["cases"][0]["expected"]["table_blocks"]) == 8


def case(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    r = materialize(**q)
    exact(r, c["expected"], c["label"])
    exact(q, before, "ownership")
    if c["events"].get("annotation_rollback"):
        assert (
            not r["annotation_blocks"] and not r["table_blocks"] and not r["claimed_line_indices"]
        )
    assert r["claimed_line_indices"] == sorted(set(r["claimed_line_indices"]))
    return r


for c in f["cases"] + cache["cases"]:
    case(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(case, f["cases"][::31]))
for meta in f["real"]:
    assert hashlib.sha256((root / meta["pdf"]).read_bytes()).hexdigest() == meta["pdf_sha256"]
    c = f["cases"][meta["case_index"]]
    q = copy.deepcopy(c["request"])
    candidate = q["candidates"][0]
    geometry = read_pdf_native_text_geometry(root / meta["pdf"], page_index=meta["page_index"])
    vectors = read_pdf_vector_geometry(root / meta["pdf"], page_index=meta["page_index"])
    assert len(geometry["chars"]) == len(q["source"]["chars"])
    chars = [{k: new[k] for k in old} for new, old in zip(geometry["chars"], q["source"]["chars"])]
    if q["source"]["lines"]:
        text = build_native_table_text(
            chars, table_bbox=candidate["bbox"], angle=candidate["angle"]
        )
        groups = {}
        by = {c["char_idx"]: c for c in chars}
        lines = []
        for g in text["glyphs"]:
            groups.setdefault(g["visual_row"], []).append(g)
        for row, gs in groups.items():
            bs = [by[g["source_index"]]["bbox"] for g in gs]
            b = [
                min(b[0] for b in bs),
                min(b[1] for b in bs),
                max(b[2] for b in bs),
                max(b[3] for b in bs),
            ]
            lines.append(
                dict(
                    text="".join(g["text"] for g in gs),
                    bbox=b,
                    source_index=row,
                    angle=candidate["angle"],
                )
            )
        exact(lines, q["source"]["lines"], "live glyph rows")
        q["source"]["lines"] = lines
    q["source"].update(
        chars=chars,
        drawing_lines=vectors["drawing_lines"],
        path_infos=vectors["path_infos"],
        page_size=geometry["page_size"],
    )
    q.update(tight_bboxes=geometry["tight_bboxes"], origins=geometry["origins"])
    exact(materialize(**q), c["expected"], "live PDF -> native/media/projection/block commit")

base = copy.deepcopy(f["cases"][0]["request"])


def limited(q):
    before = copy.deepcopy(q)
    try:
        materialize(**q)
    except RapidDocError as e:
        assert e.code == "resource_limit", e
        exact(q, before, "failed ownership")
    else:
        raise AssertionError("resource error hidden as projection or empty table")


q = copy.deepcopy(base)
q["candidates"] *= 129
limited(q)
q = copy.deepcopy(base)
q["source"]["image_bboxes"] = [[0.0, 0.0, 1.0, 1.0]] * 5001
limited(q)
# Ordinary recovery and rendering fit separately; complete orchestration must propagate its shared limit.
rules = [
    dict(bbox=[0.0, y, 100.0, y + 0.1], width=0.1, orientation="horizontal")
    for y in (0.0, 30.0, 59.9)
] + [
    dict(bbox=[x, 0.0, x + 0.1, 60.0], width=0.1, orientation="vertical") for x in (0.0, 50.0, 99.9)
]
chars = [
    dict(
        char_idx=i,
        char="A",
        bbox=[
            10.0 + 50.0 * ((i // 375) % 2),
            10.0 + 30.0 * ((i // 375) // 2),
            18.0 + 50.0 * ((i // 375) % 2),
            18.0 + 30.0 * ((i // 375) // 2),
        ],
    )
    for i in range(1500)
]
ti = dict(
    table_bbox=[0.0, 0.0, 100.0, 60.0],
    page_size=[200.0, 200.0],
    angle=0,
    drawing_lines=rules,
    rectangles=[],
    chars=chars,
)
tight = {c["char_idx"]: c["bbox"] for c in chars}
origins = {c["char_idx"]: [c["bbox"][0], c["bbox"][3]] for c in chars}
result = recover_native_table_primitives(ti)
assert result is not None
assert render_native_table_html_with_scripts(
    result, ti, tight_bboxes=tight, origins=origins, preserve_font_styles=True
)
q = copy.deepcopy(base)
q["source"].update(
    page_size=[200.0, 200.0], lines=[], chars=chars, drawing_lines=rules, page_index=1
)
q["candidates"][0].update(
    bbox=ti["table_bbox"],
    local_bbox=ti["table_bbox"],
    core_bbox=None,
    angle=0,
    line_indices=[],
    annotations=[],
    preserve_inline_font_styles=True,
)
q.update(tight_bboxes=tight, origins=origins)
limited(q)
case(f["cases"][0])
assert (
    capabilities()["native_table_block_commit"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (
    Path(__import__("rapid_doc_rust").__file__).parent / "licenses/MIGRATION-NOTICES.txt"
).is_file()
print(
    "Installed wheel: 781 complete old block transactions +one eight-candidate cache case; 192 native, 512 projection, 102 front rollbacks, 109 atomic annotation rollbacks, 24 media, four real recovery errors; 16 live PDF chains, exact side effects/types/ownership/concurrency/shared budgets; commit helper regression scope; full page/default PDF pending"
)
