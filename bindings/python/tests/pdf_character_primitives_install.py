"""独立wheel：DocVortex去重前字符、元数据及代理对恢复逐项对照。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import hashlib, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import read_pdf_character_primitives as read, RapidDocError, capabilities

root = Path(sys.argv[1]).resolve()
cases = []
for name in ("pdf-character-primitives", "pdf-character-edges"):
    f = json.loads((root / f"build/migration/{name}.json").read_text(encoding="utf-8"))
    for p, h in f["sources"].items():
        assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, p
    cases.extend(f["cases"])


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
    actual = read(data, page_index=c["page_index"], include_geometry=c["extended"])
    same(actual, c["expected"], c["pdf"] + ":" + str(c["page_index"]))
    for ch in actual["chars"]:
        assert ch["source_indices"] == [ch["char_idx"]]
        assert (ch["text_object_id"] is None) or (
            0 <= ch["text_object_id"] < actual["raw_char_count"]
        )
        assert ("tight_bbox" in ch) == c["extended"] and ("loose_bbox" in ch) == c["extended"]
    return actual


for c in cases:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, cases[::7]))
assert len(cases) == 58 and sum(len(c["expected"]["chars"]) for c in cases) == 7650
edge = cases[-1]["expected"]
assert any(c["char"] == "😀" and len(c["source_indices"]) == 2 for c in edge["restored_chars"])
assert any(c["char"] == "𠀀" and len(c["source_indices"]) == 2 for c in edge["restored_chars"])
assert sum(c["char"] == "f" for c in edge["chars"]) == 2
assert any(c["raw_code"] == 0xD800 and c["char"] == "�" for c in edge["restored_chars"])
stream = io.BytesIO((root / cases[0]["pdf"]).read_bytes())
same(read(stream, include_geometry=False), cases[0]["expected"])
assert not stream.closed
for kwargs in [
    dict(page_index=-1),
    dict(page_index=True),
    dict(page_index=1.5),
    dict(page_index=999),
    dict(password="a\0b"),
    dict(include_geometry=1),
]:
    try:
        read(root / cases[0]["pdf"], **kwargs)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(kwargs)
for password in (None, "wrong"):
    try:
        read(root / "build/migration/encrypted-native.pdf", password=password)
    except RapidDocError as exc:
        assert exc.code == "encrypted"
    else:
        raise AssertionError("encrypted PDF accepted")
read(root / "build/migration/encrypted-native.pdf", password="test-password")
try:
    read(b"invalid")
except RapidDocError as exc:
    assert exc.code == "corrupt_document"
else:
    raise AssertionError("invalid PDF accepted")
run(cases[0])
assert capabilities()["docvortex_pdf_character_primitives"] is True
assert capabilities()["legacy_native_table_recovery"] is False
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
print(
    "Passed 58 actual PDF/mode cases / 7650 raw records and surrogate restorations, metadata, concurrency, ownership and errors; no dedup/visible-only claim"
)
