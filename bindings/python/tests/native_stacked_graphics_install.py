"""Installed old CJK scale/stacked-column facts plus real PDF input reads."""

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
    detect_native_stacked_graphics as detect,
    read_pdf_native_text_geometry as chars,
    read_pdf_vector_geometry as vectors,
    build_native_visual_lines as visual,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-stacked-graphics.json").read_text(encoding="utf-8"))
for p, h in f["sources"].items():
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
    out = detect(**q)
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")
    out["stacked_column_bboxes"].append([0, 0, 1, 1])
    same(q, before, "independent output")


assert len(f["cases"]) == 304
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::19]))
live = [c for c in f["cases"] if "pdf_ref" in c]
assert len(live) == 41
for c in live:
    ref = c["pdf_ref"]
    data = (root / ref["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["pdf_sha256"]
    i = ref["page_index"]
    g = chars(data, page_index=i)
    v = vectors(data, page_index=i)
    rows = visual(g["chars"], page_size=g["page_size"], page_rotation=g["rotation"])
    source = dict(
        chars=g["chars"],
        path_infos=v["path_infos"],
        drawing_lines=v["drawing_lines"],
        lines=[{k: l[k] for k in ["text", "bbox", "source_index", "angle"]} for l in rows],
    )
    q = copy.deepcopy(c["request"])
    q["source"] = source
    same(detect(**q), c["expected"], "live " + c["label"])


def error(q, code):
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError("expected " + code)


base = dict(source={}, em=10.0, fallback_em=30.0)
error(
    dict(
        base,
        source=dict(
            chars=[dict(char="中", bbox=[0, 0, 10, 30], tight_bbox=[0, 0, 10, 10])] * 20001
        ),
    ),
    "resource_limit",
)
error(
    dict(base, source=dict(path_infos=[dict(bbox=[0, 0, 1, 1], segment_count=5)] * 5001)),
    "resource_limit",
)
error(dict(base, em="10"), "invalid_input")
error(dict(base, source=dict(chars=[dict(char="中", tight_bbox=[0, 0, 1, 1])])), "invalid_input")
error(dict(base, source=dict(lines=[dict(text="x", bbox=[0, 0, 1])])), "invalid_input")
# Valid JSON count domain; repeated candidate rectangles prepay peer membership scans.
q = copy.deepcopy(f["cases"][0]["request"])
path = q["source"]["path_infos"][0]
path["rectangle_bboxes"] = [[50 + i * 20, 100 - i, 60 + i * 20, 150] for i in range(4000)]
error(q, "resource_limit")
run(f["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_stacked_graphics"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "304 old CJK-scale/stacked facts, 162 stacked positives; 41 live PDF chars/vectors/visual-line inputs; exact fields/types/order/ownership/concurrency/budgets/recovery passed; stacked helper regression scope; default PDF orchestration pending"
)
