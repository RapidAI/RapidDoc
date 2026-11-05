"""旧Python实际PDF绘图证据；独立wheel不可导入旧解析器。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import copy, hashlib, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rapid_doc_rust import (
    read_pdf_vector_geometry as read,
    evaluate_native_table_primitives,
    RapidDocError,
    capabilities,
)

root = Path(sys.argv[1]).resolve()
f = json.loads((root / "build/migration/pdf-vector-primitives.json").read_text(encoding="utf-8"))
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
    same(
        read(data, page_index=c["page_index"]), c["expected"], c["pdf"] + ":" + str(c["page_index"])
    )


for c in f["cases"]:
    run(c)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(run, f["cases"][::4]))
assert len(f["cases"]) == 41
assert sum(len(c["expected"]["drawing_lines"]) for c in f["cases"]) == 428
assert sum(len(c["expected"]["path_infos"]) for c in f["cases"]) == 3694
stream = io.BytesIO((root / f["cases"][0]["pdf"]).read_bytes())
same(read(stream), f["cases"][0]["expected"])
assert not stream.closed
# 16实际PDF/页面角度：绘图实时提取、原始字符保留独立旧基线；不宣称字符已从Rust读取。
e = json.loads((root / "build/migration/native-table-engine.json").read_text(encoding="utf-8"))
for r in e["real"]:
    case = e["cases"][r["case_index"]]
    request = copy.deepcopy(case["request"])
    original = request["table_input"]
    geometry = read(root / r["pdf"], page_index=r["page_index"])
    lines = [{k: x[k] for k in ("bbox", "orientation", "width")} for x in geometry["drawing_lines"]]
    rects = [
        {k: x[k] for k in ("bbox", "fill_visible", "form_depth", "segment_count", "stroke_visible")}
        for x in geometry["path_infos"]
    ]
    same(lines, original["drawing_lines"], "engine-lines")
    same(rects, original["rectangles"], "engine-rects")
    original.update(
        drawing_lines=geometry["drawing_lines"],
        rectangles=geometry["path_infos"],
        page_size=geometry["page_size"],
    )
    same(
        evaluate_native_table_primitives(**request),
        case["expected"]["evaluation"],
        "live-vector-engine",
    )
for kwargs in [
    dict(page_index=-1),
    dict(page_index=True),
    dict(page_index=1.5),
    dict(page_index=999),
    dict(password="a\0b"),
]:
    try:
        read(root / f["cases"][0]["pdf"], **kwargs)
    except (ValueError, RapidDocError):
        pass
    else:
        raise AssertionError(kwargs)
for source, code in [(b"", None), (b"invalid", "corrupt_document")]:
    try:
        read(source)
    except (ValueError, RapidDocError) as exc:
        if code:
            assert exc.code == code, exc
    else:
        raise AssertionError("invalid PDF accepted")
encrypted = root / "build/migration/encrypted-native.pdf"
for password in (None, "wrong"):
    try:
        read(encrypted, password=password)
    except RapidDocError as exc:
        assert exc.code == "encrypted"
    else:
        raise AssertionError("missing password accepted")
read(encrypted, password="test-password")
run(f["cases"][0])
assert capabilities()["docvortex_pdf_vector_geometry"] is True
assert "rapid_doc" not in sys.modules and "pypdfium2" not in sys.modules
import rapid_doc_rust as sdk

for name in (
    "PYTHON-MATH.txt",
    "MIGRATION-NOTICES.txt",
    "DOCVORTEX-PDFIUM-PATHS.txt",
    "MIGRATION-NOTICES.txt",
):
    assert (Path(sdk.__file__).parent / "licenses" / name).read_bytes() == (
        root / "bindings/python/python/rapid_doc_rust/licenses" / name
    ).read_bytes()
print(
    "Passed 41 actual PDF pages / 428 lines / 3694 paths, 16 live-vector engine compositions, ownership, concurrency and errors"
)
