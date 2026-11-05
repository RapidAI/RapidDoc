"""Installed SDK: old complete font/drawing style and URI link evidence."""

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
    detect_native_inline_evidence as detect,
    prepare_native_source_lines as prepare,
    build_native_visual_lines as build,
    read_pdf_native_text_geometry as read,
    read_pdf_character_primitives as raw,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-inline-evidence.json").read_text(encoding="utf-8"))
extra = json.loads(
    (root / "build/migration/native-inline-evidence-edges.json").read_text(encoding="utf-8")
)
f["cases"].extend(extra["cases"])
f["sources"].update(extra["sources"])
for p, h in f["sources"].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p


def same(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
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


def run(c):
    q = copy.deepcopy(c["request"])
    before = copy.deepcopy(q)
    same(detect(**q), c["expected"], c["label"])
    same(q, before, "ownership")


assert len(f["cases"]) == 285
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::17]))
s = json.loads((root / "build/migration/native-source-lines.json").read_text(encoding="utf-8"))
visual = {
    name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-visual-lines", "native-visual-lines-extra"]
}
chains = [c for c in f["cases"] if "source_case" in c]
assert len(chains) == 57
assert sum(bool(c["expected"]["style_lines"]) for c in chains) == 2
assert sum(bool(c["expected"]["link_lines"]) for c in chains) == 4
for c in chains:
    sc = s["cases"][c["source_case"]]
    ref = sc["visual_case"]
    v = visual[ref["corpus"]]["cases"][ref["index"]]
    data = (root / v["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == v["pdf_sha256"]
    dedup = v.get("reader") == "deduplicated"
    g = (read if dedup else raw)(data, page_index=v["page_index"])
    lines = build(
        g["chars" if dedup else "restored_chars"],
        page_size=g["page_size"],
        page_rotation=g["rotation"],
    )
    q = copy.deepcopy(sc["request"])
    q["lines"] = lines
    prepared = prepare(**q)
    # Link annotations are independent old-PDF facts, not yet read by the native SDK.
    same(
        detect(
            prepared["lines"],
            drawing_lines=prepared["drawing_lines"],
            annotations=c["request"]["annotations"],
        ),
        c["expected"],
        "live " + c["label"],
    )
line = dict(
    text="ab",
    source_index=0,
    angle=0,
    bbox=[10.0, 20.0, 22.0, 30.0],
    chars=[
        dict(char="a", char_idx=0, bbox=[10.0, 20.0, 16.0, 30.0]),
        dict(char="b", char_idx=1, bbox=[16.0, 20.0, 22.0, 30.0]),
    ],
)
for q in [
    dict(lines=[line] * 4097),
    dict(lines=[line], annotations=[dict(target="x", bboxes=[[10.0, 20.0, 22.0, 30.0]])] * 5001),
    dict(
        lines=[line] * 1000,
        annotations=[dict(target="x", bboxes=[[10.0, 20.0, 22.0, 30.0]])] * 5000,
    ),
]:
    try:
        detect(**q)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("resource limit ignored")
run(f["cases"][0])
assert (
    capabilities()["native_inline_evidence"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "285 old complete style/link facts including 10 compact-marker boundaries, 57 live PDF character -> visual/source line chains (2 style-positive/4 link-positive); exact Unicode/codepoint intervals, values/types/order, ownership/concurrency/limits passed. Link annotations are supplied old-PDF facts; native annotation reading/pixel glyph-weight and output projection remain pending"
)
