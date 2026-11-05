"""Installed wheel: old block preparation and live PDF primitives -> full table HTML."""

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
    prepare_native_table_block as prepare,
    recover_native_table_result as recover,
    build_native_table_text,
    read_pdf_native_text_geometry,
    read_pdf_vector_geometry,
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


f = load("native-table-block")
typed = load("native-table-block-types")
chains = load("native-table-block-chains")
assert len(f["cases"]) == 612 and len(typed["cases"]) == 12 and len(chains["cases"]) == 16
assert sum(c["split_note"] for c in f["cases"]) == 53
assert sum(c["expanded"] for c in f["cases"]) == 207
assert sum(c["repaired"] for c in f["cases"]) == 46


def case(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    r = prepare(**q)
    exact(r, c["expected"], c["label"])
    exact(q, before, "ownership")
    assert r["projection_line_indices"] == sorted(set(r["projection_line_indices"]))
    for old, new in zip(q["source"]["chars"], r["native_chars"]):
        assert old.keys() == new.keys() or new.keys() == old.keys() | {"bbox"}
        exact(
            {k: v for k, v in new.items() if k != "bbox"},
            {k: v for k, v in old.items() if k != "bbox"},
        )
    return r


for c in f["cases"] + typed["cases"]:
    case(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(case, f["cases"][::19] + typed["cases"]))
expected_chains = {c["case_index"]: c["expected"] for c in chains["cases"]}
for meta in f["real"]:
    assert hashlib.sha256((root / meta["pdf"]).read_bytes()).hexdigest() == meta["pdf_sha256"]
    c = f["cases"][meta["case_index"]]
    q = copy.deepcopy(c["request"])
    geometry = read_pdf_native_text_geometry(root / meta["pdf"], page_index=meta["page_index"])
    vectors = read_pdf_vector_geometry(root / meta["pdf"], page_index=meta["page_index"])
    # All values come from a live SDK read; preserve the independently frozen character field projection.
    chars = [{k: new[k] for k in old} for new, old in zip(geometry["chars"], q["source"]["chars"])]
    assert len(chars) == len(geometry["chars"]) == len(q["source"]["chars"])
    if q["source"]["lines"]:
        text = build_native_table_text(
            chars, table_bbox=q["candidate"]["bbox"], angle=q["candidate"]["angle"]
        )
        groups = {}
        by = {c["char_idx"]: c for c in chars}
        lines = []
        for g in text["glyphs"]:
            groups.setdefault(g["visual_row"], []).append(g)
        for row, gs in groups.items():
            bs = [by[g["source_index"]]["bbox"] for g in gs]
            box = [
                min(b[0] for b in bs),
                min(b[1] for b in bs),
                max(b[2] for b in bs),
                max(b[3] for b in bs),
            ]
            lines.append(
                dict(
                    text="".join(g["text"] for g in gs),
                    bbox=box,
                    source_index=row,
                    angle=q["candidate"]["angle"],
                )
            )
        exact(lines, q["source"]["lines"], "live chars -> glyph visual row projection")
        q["source"]["lines"] = lines
    q["source"].update(
        chars=chars, drawing_lines=vectors["drawing_lines"], page_size=geometry["page_size"]
    )
    prepared = prepare(**q)
    exact(prepared, c["expected"], "live PDF -> preparation")
    ti = dict(
        table_bbox=prepared["body_bbox"],
        page_size=geometry["page_size"],
        angle=prepared["candidate"]["angle"],
        chars=prepared["native_chars"],
        drawing_lines=vectors["drawing_lines"],
        rectangles=vectors["path_infos"],
    )
    value = recover(ti, tight_bboxes=geometry["tight_bboxes"], origins=geometry["origins"])
    value = None if value is None else dict(content=value[0], result=value[1])
    exact(
        value,
        expected_chains[meta["case_index"]],
        "live PDF -> preparation -> engine -> semantic HTML",
    )

base = copy.deepcopy(f["cases"][0]["request"])


def rejects(q, code):
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("invalid/oversized block input accepted")


q = copy.deepcopy(base)
q["source"]["lines"] = [
    dict(text="body", bbox=[0.0, 0.0, 10.0, 10.0], source_index=i, angle=0) for i in range(4097)
]
rejects(q, "resource_limit")
q = copy.deepcopy(base)
q["source"]["chars"] = [{}] * 20001
rejects(q, "resource_limit")
q = copy.deepcopy(base)
q["candidate"]["annotations"] = [{}] * 2001
rejects(q, "resource_limit")
q = copy.deepcopy(base)
q["source"]["lines"] = [dict(text="x", bbox=[0.0, 0.0, 10.0, 10.0], source_index=1, angle=0)] * 2
rejects(q, "invalid_input")
q = copy.deepcopy(base)
q["calibrated_char_bboxes"] = {"0": [10.0, 0.0, 0.0, 10.0]}
rejects(q, "invalid_input")
# Each stage fits on its own; shared font-group comparisons + baseline comparisons exceed 10M.
chars = [dict(char="A", char_idx=i, font=dict(name=f"Font{i}")) for i in range(4000)]
lines = [
    dict(
        text="body",
        source_index=i,
        angle=0,
        bbox=[10.0, 0.0 if i < 1500 else 100.0, 20.0, 1.0 if i < 1500 else 101.0],
    )
    for i in range(3000)
]
q = copy.deepcopy(base)
q["source"].update(chars=chars, lines=lines, drawing_lines=[])
q["candidate"].update(
    bbox=[0.0, 0.0, 300.0, 200.0],
    core_bbox=None,
    angle=0,
    line_indices=list(range(1500)),
    annotations=[],
)
a = copy.deepcopy(q)
a["source"]["lines"] = []
assert prepare(**a)
a = copy.deepcopy(q)
a["source"]["chars"] = []
assert prepare(**a)
rejects(q, "resource_limit")
case(f["cases"][0])
assert capabilities()["native_table_block_preparation"]
assert (
    not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (
    Path(__import__("rapid_doc_rust").__file__).parent / "licenses/MIGRATION-NOTICES.txt"
).is_file()
print(
    "Installed wheel: 612 old block preparations +12 numeric-type facts (53 note splits, 207 claim expansions, 46 repair cases); 16 live PDF/angle preparation -> engine -> semantic HTML chains, 12 nonempty; exact values/types, source retention, ownership/concurrency/shared budget/limits; preparation helper regression scope; full page/default PDF pending"
)
