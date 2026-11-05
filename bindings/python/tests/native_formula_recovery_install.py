"""Installed full original formula recognizer, dense/baseline sequence and live PDFium."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import rapid_doc_rust as sdk

root = Path(sys.argv[1]).resolve()
load = lambda n: json.loads((root / f"build/migration/{n}.json").read_text(encoding="utf-8"))


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b), a, b)
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (path, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


def run(c, sequence=False):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    fn = sdk.prepare_native_formula_input if sequence else sdk.recover_native_formula_blocks
    out = fn(
        q["lines"], q["page_size"], table_bboxes=q["table_bboxes"], drawing_lines=q["drawing_lines"]
    )
    same(out, c["expected"], c["label"])
    same(q, before, "ownership")


full = [
    c
    for i, c in enumerate(load("native-formula-full")["cases"])
    if "kind" not in c["request"]
    and (i % 53 == 0 or c["label"].startswith(("continuation/", "variation/")))
]
sequence = [
    c
    for i, c in enumerate(load("native-formula-input")["cases"])
    if not c["request"].get("dense_only")
    and (i % 53 == 0 or c["label"].startswith(("continuation/", "variation/", "sequence/")))
]
for c in full:
    run(c)
for c in sequence:
    run(c, True)
with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(lambda c: run(c, True), sequence[-4:]))
live = load("native-formula-runtime")
data = (root / live["pdf"]).read_bytes()
library = str(root / ".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll")
for i, c in enumerate(live["cases"]):
    g = sdk.read_pdf_native_text_geometry(data, page_index=i, pdfium_library=library)
    v = sdk.read_pdf_vector_geometry(data, page_index=i, pdfium_library=library)
    ls = sdk.build_native_visual_lines(
        g["chars"], page_size=g["page_size"], page_rotation=g["rotation"]
    )
    same(ls, c["request"]["lines"], "live-lines")
    same(v["drawing_lines"], c["request"]["drawing_lines"], "live-rules")
    actual = sdk.prepare_native_formula_input(ls, g["page_size"], drawing_lines=v["drawing_lines"])
    same(actual, c["expected"], "live-formula")
    assert actual["formula_blocks"]
for fn in [sdk.recover_native_formula_blocks, sdk.prepare_native_formula_input]:
    for ls in ["invalid", [dict(text=None, bbox=[0.0, 0.0, 10.0, 10.0], source_index=0)]]:
        try:
            fn(ls, [400.0, 600.0])
        except sdk.RapidDocError as e:
            assert e.code in ("invalid_input", "resource_limit")
        else:
            raise AssertionError("invalid formula lines")
assert sdk.capabilities()["native_formula_block_recovery"]
assert sdk.capabilities()["native_formula_input_preparation"]
assert sdk.capabilities()["compatibility"]["python_parser"] is False
print(
    f"{len(full)} full formula cases, {len(sequence)} dense/baseline/formula sequences, strict types/ownership/concurrency and 2 fresh positive PDFium pages passed"
)
