"""独立安装 wheel 对照真实旧 PDFium 字符和完整 pdftext 分组。"""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__("runpy").run_path(
    str(
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "scripts/migration/preview_compat.py"
    )
)["activate"]()
import hashlib, json, sys
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from rapid_doc_rust import read_native_pdf_page, RapidDocError

root = Path(sys.argv[1]).resolve()
fixture = json.loads((root / "build/migration/legacy-native-page.json").read_text(encoding="utf-8"))


def equal(a, b, path=""):
    assert type(a) is type(b), (path, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            equal(a[k], b[k], path + "/" + k)
    elif isinstance(a, list):
        assert len(a) == len(b), (path, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, path + "/" + str(i))
    else:
        assert a == b, (path, a, b)


for case in fixture["cases"]:
    data = (root / case["source"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == case["sha256"]
    equal(
        read_native_pdf_page(
            data, page_index=case["page_index"], quote_loosebox=case["quote_loosebox"]
        ),
        case["expected"],
        case["source"],
    )
data = (root / "tests/migration/inputs/pdf_native.pdf").read_bytes()
expected = read_native_pdf_page(data)
stream = BytesIO(data)
equal(read_native_pdf_page(stream), expected)
assert not stream.closed
with ThreadPoolExecutor(max_workers=4) as pool:
    for value in pool.map(lambda _: read_native_pdf_page(data), range(16)):
        equal(value, expected)
for value, options, code in [
    (b"", {}, "invalid_input"),
    (b"bad", {}, "corrupt_document"),
    (data, {"page_index": 1000000}, "invalid_input"),
    (data, {"password": "x\0y"}, "invalid_input"),
]:
    try:
        read_native_pdf_page(value, **options)
    except RapidDocError as e:
        assert e.code == code, (e.code, code)
    else:
        raise AssertionError(code)
equal(read_native_pdf_page(data), expected)
assert "Apache License" in files("rapid_doc_rust").joinpath("licenses/PDFTEXT.txt").read_text()
print(
    f"Installed native PDF bridge: {len(fixture['cases'])} full old page JSON references, streams, concurrency and errors passed"
)
