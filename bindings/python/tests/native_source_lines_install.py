"""Installed SDK: old local paragraph/rule source stage and actual live PDF chains."""

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
    prepare_native_source_lines as prepare,
    build_native_visual_lines as build,
    read_pdf_character_primitives as raw,
    read_pdf_native_text_geometry as read,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-source-lines.json").read_text(encoding="utf-8"))
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
    request = copy.deepcopy(c["request"])
    before = copy.deepcopy(request)
    same(prepare(**request), c["expected"], c["label"])
    same(request, before, "ownership")


assert len(f["cases"]) == 290
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::17]))
chains = [c for c in f["cases"] if "visual_case" in c]
assert len(chains) == 57
corpora = {
    name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-visual-lines", "native-visual-lines-extra"]
}
for c in chains:
    ref = c["visual_case"]
    v = corpora[ref["corpus"]]["cases"][ref["index"]]
    data = (root / v["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == v["pdf_sha256"]
    dedup = v.get("reader") == "deduplicated"
    g = (read if dedup else raw)(data, page_index=v["page_index"])
    lines = build(
        g["chars" if dedup else "restored_chars"],
        page_size=g["page_size"],
        page_rotation=g["rotation"],
    )
    q = copy.deepcopy(c["request"])
    q["lines"] = lines
    same(prepare(**q), c["expected"], "live " + c["label"])
for q in [
    dict(
        lines=[
            dict(text="a)", bbox=[10.0, 10.0, 18.0, 20.0], angle=0, source_index=i)
            for i in range(4097)
        ],
        page_size=[400.0, 600.0],
    ),
    dict(
        lines=[],
        page_size=[400.0, 600.0],
        drawing_lines=[dict(bbox=[0.0, 0.0, 1.0, 1.0], width=1.0, orientation="horizontal")] * 5001,
    ),
]:
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("input count limit not enforced")
# Valid input size, repeated eligible markers exhaust shared scans instead of succeeding unbounded.
crowded = [
    dict(text="a)", bbox=[10.0, i * 20.0, 18.0, i * 20.0 + 10.0], angle=0, source_index=i)
    for i in range(1000)
]
try:
    prepare(crowded, page_size=[400.0, 20010.0])
except RapidDocError as e:
    assert e.code == "resource_limit"
else:
    raise AssertionError("scan budget not enforced")
for q in [
    dict(lines=[], page_size=[0.0, 1.0]),
    dict(
        lines=[dict(text="x", bbox=[0.0, 0.0, 1.0, 1.0], angle=45, source_index=0)],
        page_size=[400.0, 600.0],
    ),
]:
    try:
        prepare(**q)
    except RapidDocError as e:
        assert e.code == "invalid_input"
    else:
        raise AssertionError("invalid geometry accepted")
run(f["cases"][0])
import rapid_doc_rust as sdk

assert (
    capabilities()["native_source_lines"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "290 old source-line facts: drop caps/hanging paragraphs/decorative and thin raster rules; 57 live PDF characters -> visual lines -> source-line stage, exact fields/types/order/ownership/concurrency/budgets passed; styles/links/image/noise/document preparation/candidate detection remain pending"
)
