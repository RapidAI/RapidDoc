"""Installed full ordinary rule candidate assembly with automatic notes and inverse rotation."""

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
    expand_native_rule_table_candidate as expand,
    read_pdf_vector_geometry as vectors,
    read_pdf_native_text_geometry as geometry,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads(
    (root / "build/migration/native-rule-candidate-expansion.json").read_text(encoding="utf-8")
)
pdf = json.loads(
    (root / "build/migration/pdf-rule-candidate-expansion.json").read_text(encoding="utf-8")
)
for corpus in [f, pdf]:
    for p, h in corpus["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def decode(v):
    v = copy.deepcopy(v)
    for a in v["candidate"]["annotations"]:
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
    out = expand(**q)
    same(out, decode(c["expected"]), c["label"])
    same(q, before, "ownership")
    out["candidate"]["bbox"][0] = -999
    same(q, before, "output independence")


assert len(f["cases"]) == 155
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::8]))
assert len(pdf["cases"]) == 20
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
    caption = None
    for i, l in enumerate(lines):
        b = l["bbox"]
        cy = (b[1] + b[3]) / 2
        cx = (b[0] + b[2]) / 2
        if frame[0] <= cx <= frame[2] and frame[1] <= cy <= frame[3]:
            core.append(i)
        if caption is None and l["text"].startswith("Table"):
            caption = l["source_index"]
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
        rule_bboxes=[r["bbox"] for r in v["drawing_lines"]] or [frame],
        core_row_indices=core,
        median_height=10.0,
        page_size=g["page_size"],
        angle=0,
        caption_source_index=caption,
        has_possible_table_notes=True,
    )
    same(q, c["request"], "live query")
    before = copy.deepcopy(q)
    same(expand(**q), decode(c["expected"]), c["label"])
    same(q, before, "live ownership")


for c in pdf["cases"]:
    live(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(live, pdf["cases"][::4]))


def error(q, code):
    try:
        expand(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = copy.deepcopy(f["cases"][0]["request"])
error(dict(base, rows=[{}] * 4097), "resource_limit")
error(dict(base, lines=[{}] * 4097), "resource_limit")
error(dict(base, rule_bboxes=[[0, 0, 1, 1]] * 5001), "resource_limit")
for q in [
    dict(base, rule_bboxes=[]),
    dict(base, core_row_indices=[]),
    dict(base, page_size=[1]),
    dict(base, has_possible_table_notes=1),
]:
    error(q, "invalid_input")
# The old negative gate must avoid parsing unconsumed note text.
q = copy.deepcopy(base)
q["has_possible_table_notes"] = False
q["lines"] = [None]
same(
    expand(**q),
    decode(next(c for c in f["cases"] if c["label"] == "angle-0-notes-False")["expected"]),
    "negative note gate",
)
q = copy.deepcopy(base)
before = copy.deepcopy(q)
out = expand(**q)
next(iter(out["candidate"]["annotations"][0]["line_bboxes"].values()))[0] = -999
same(q, before, "deep independence")
run(f["cases"][0])
live(pdf["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_rule_candidate_expansion"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "155 full ordinary rule expansion references/139 annotated; 20 live PDF full assemblies/7 annotated; inverse rotation/types/ordered boxes/body-note separation/negative gate/ownership/concurrency/shared budgets passed; ordinary generation now covered by native_structural_table_pipeline; prepared+owned optimization/default PDF pending"
)
