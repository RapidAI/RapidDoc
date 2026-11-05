"""Installed SDK: real PDF URI annotations and fully live character+annotation evidence."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import hashlib, io, json, sys, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from rapid_doc_rust import (
    read_pdf_link_annotations as read,
    read_pdf_native_text_geometry as geometry,
    read_pdf_character_primitives as raw,
    build_native_visual_lines as build,
    prepare_native_source_lines as prepare,
    detect_native_inline_evidence as detect,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/pdf-link-annotations.json").read_text(encoding="utf-8"))
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
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["pdf_sha256"]
    same(read(data, page_index=c["page_index"]), c["expected"], c["pdf"] + str(c["page_index"]))


assert len(f["cases"]) == 36 and sum(len(c["expected"]) for c in f["cases"]) == 26
for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::5]))
# No injected annotations or characters: both now read by Rust for existing old page evidence.
source = json.loads((root / "build/migration/native-source-lines.json").read_text(encoding="utf-8"))
inline = json.loads(
    (root / "build/migration/native-inline-evidence.json").read_text(encoding="utf-8")
)
vc = {
    name: json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for name in ["native-visual-lines", "native-visual-lines-extra"]
}
chains = [c for c in inline["cases"] if "source_case" in c]
assert len(chains) == 57
for c in chains:
    sc = source["cases"][c["source_case"]]
    r = sc["visual_case"]
    v = vc[r["corpus"]]["cases"][r["index"]]
    data = (root / v["pdf"]).read_bytes()
    dedup = v.get("reader") == "deduplicated"
    g = (geometry if dedup else raw)(data, page_index=v["page_index"])
    lines = build(
        g["chars" if dedup else "restored_chars"],
        page_size=g["page_size"],
        page_rotation=g["rotation"],
    )
    q = copy.deepcopy(sc["request"])
    q["lines"] = lines
    p = prepare(**q)
    annotations = read(data, page_index=v["page_index"])
    same(annotations, c["request"]["annotations"], "live annotations")
    same(
        detect(p["lines"], drawing_lines=p["drawing_lines"], annotations=annotations),
        c["expected"],
        "live evidence",
    )
stream = io.BytesIO((root / f["cases"][0]["pdf"]).read_bytes())
same(read(stream), f["cases"][0]["expected"])
assert not stream.closed
for options in [
    dict(page_index=-1),
    dict(page_index=True),
    dict(page_index=1.5),
    dict(page_index=999),
    dict(password="a\0b"),
]:
    try:
        read(root / f["cases"][0]["pdf"], **options)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(options)
for password in [None, "wrong"]:
    try:
        read(root / "build/migration/encrypted-native.pdf", password=password)
    except RapidDocError as e:
        assert e.code == "encrypted"
    else:
        raise AssertionError("encrypted input accepted")
read(root / "build/migration/encrypted-native.pdf", password="test-password")
limits = json.loads(
    (root / "build/migration/pdf-link-annotations/limits.json").read_text(encoding="utf-8")
)
for c in limits["cases"]:
    data = (root / c["pdf"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == c["sha256"]
    try:
        read(data)
    except RapidDocError as e:
        assert e.code == "resource_limit"
    else:
        raise AssertionError("URI/quad budget hidden as an omitted link")
try:
    read(b"invalid")
except RapidDocError as e:
    assert e.code == "corrupt_document"
else:
    raise AssertionError("corrupt input accepted")
run(f["cases"][0])
assert (
    capabilities()["pdf_link_annotations"]
    and not capabilities()["legacy_native_table_recovery"]
    and not capabilities()["legacy_table_execution"]
)
import rapid_doc_rust as sdk

assert (Path(sdk.__file__).parent / "licenses/MIGRATION-NOTICES.txt").is_file()
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "36 actual PDF pages/26 URI annotations exact; 57 fully live PDF character+annotation -> visual/source lines -> inline evidence chains; hidden/action/URI/quad fallback/rotation/CropBox/source ids, ownership/concurrency/encryption/errors passed; no full page-source/default PDF or final output projection claim"
)
