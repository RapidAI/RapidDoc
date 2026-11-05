"""Installed SDK: old complete style/link output and real PDF character/annotation chains."""

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
    apply_native_inline_evidence as apply,
    detect_native_inline_evidence as detect,
    read_pdf_link_annotations as read_links,
    read_pdf_native_text_geometry as read,
    read_pdf_character_primitives as raw,
    build_native_visual_lines as build,
    prepare_native_source_lines as prepare,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/native-inline-output.json").read_text(encoding="utf-8"))
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
    same(apply(**q), c["expected"], c["label"])
    same(q, before, "ownership")


assert len(f["cases"]) == 297
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::17]))
inline = json.loads(
    (root / "build/migration/native-inline-evidence.json").read_text(encoding="utf-8")
)
source = json.loads((root / "build/migration/native-source-lines.json").read_text(encoding="utf-8"))
vc = {
    name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-visual-lines", "native-visual-lines-extra"]
}
chains = [c for c in f["cases"] if "inline_case" in c]
assert len(chains) == 57
for c in chains:
    ic = inline["cases"][c["inline_case"]]
    sc = source["cases"][ic["source_case"]]
    r = sc["visual_case"]
    v = vc[r["corpus"]]["cases"][r["index"]]
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
    p = prepare(**q)
    evidence = detect(
        p["lines"],
        drawing_lines=p["drawing_lines"],
        annotations=read_links(data, page_index=v["page_index"]),
    )
    # Explicit visual-line-derived blocks test output projection; this does not replace page block detection.
    blocks = [
        dict(type="text", content=l["text"], bbox=l["bbox"], source="preserved") for l in p["lines"]
    ]
    same(
        apply(blocks, page_size=g["page_size"], **evidence),
        c["expected"],
        "live output " + c["label"],
    )
for q in [
    dict(
        blocks=[dict(type="text", content="x", bbox=[0.0, 0.0, 1.0, 1.0])] * 4001,
        page_size=[400.0, 600.0],
    ),
    dict(
        blocks=[dict(type="text", content="\\(" * 5000, bbox=[0.0, 0.0, 1.0, 1.0])],
        page_size=[400.0, 600.0],
    ),
]:
    try:
        apply(**q)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("input/unterminated formula scan limit not enforced")
run(f["cases"][0])
assert (
    capabilities()["native_inline_output"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "297 complete old style/link InlineSpan outputs; 57 live PDF character+annotation -> visual/source/evidence -> output chains, exact values/types/offsets/fields/owned script markers, ownership/concurrency/limits passed; blocks are explicit visual-line-derived inputs, script-line projection and full page/default PDF remain pending"
)
