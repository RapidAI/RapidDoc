"""对真实安装 wheel 执行 PDF 基线、资源限制、并发和生命周期验证。"""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import sys

from rapid_doc_rust import RapidDocError, parse_pdf_text


def check_error(code, source, **options):
    """断言稳定错误码，避免只检查任意异常而掩盖错误分类。"""
    try:
        parse_pdf_text(source, **options)
    except RapidDocError as error:
        assert error.code == code, (error.code, code)
    else:
        raise AssertionError(f"Expected {code}")


def main():
    """在没有旧解析器依赖的环境中读取已冻结基线，保持解释器隔离。"""
    root = Path(sys.argv[1]).resolve()
    options = {"pdfium_library": Path(sys.argv[2]).resolve()} if len(sys.argv) > 2 else {}
    inputs = root / "tests/migration/inputs"
    for name in ("native", "scan", "mixed"):
        path = inputs / f"pdf_{name}.pdf"
        actual = parse_pdf_text(path, allow_partial=True, **options)
        expected = json.loads(
            (root / f"tests/migration/golden/pdf_{name}.json").read_text(encoding="utf-8")
        )
        assert len(actual.pages) == len(expected["pages"])
        for page, golden in zip(actual.pages, expected["pages"]):
            assert page["page_index"] == golden["page_index"]
            assert page["text"] == "".join(char["char"] for char in golden["chars"])
            assert len(page["characters"]) == len(golden["chars"])
        assert bool(actual.diagnostics) == (name != "native")
    native = (inputs / "pdf_native.pdf").read_bytes()
    first = parse_pdf_text(native, **options)
    stream = BytesIO(native)
    assert parse_pdf_text(stream, **options).text == first.text
    assert not stream.closed
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: parse_pdf_text(native, **options).text, range(24)))
    assert values == [first.text] * 24
    selected = parse_pdf_text(inputs / "pdf_mixed.pdf", start_page=2, end_page=3, **options)
    assert selected.pages[0]["page_index"] == 2
    assert "Text after scanned page" in selected.text
    check_error("model_missing", inputs / "pdf_scan.pdf", **options)
    check_error("model_missing", inputs / "pdf_mixed.pdf", **options)
    check_error("invalid_input", b"", **options)
    check_error("corrupt_document", b"%PDF-1.7 invalid", **options)
    check_error("invalid_input", native, end_page=2, **options)
    check_error("invalid_input", native, start_page=1, end_page=0, **options)
    check_error("resource_limit", native, max_input_bytes=10, **options)
    check_error("resource_limit", native, max_pages=0, **options)
    check_error("resource_limit", native, max_chars_per_page=1, **options)
    check_error("resource_limit", native, max_total_chars=1, **options)
    # 连续失败不能污染运行库，之后的合法文档仍能解析。
    assert parse_pdf_text(native, **options).text == first.text
    encrypted = root / "build/migration/encrypted-native.pdf"
    check_error("encrypted", encrypted, **options)
    check_error("encrypted", encrypted, password="wrong", **options)
    assert parse_pdf_text(encrypted, password="test-password", **options).text == first.text
    for rotation in (0, 90, 180, 270):
        result = parse_pdf_text(
            root / f"build/migration/geometry/forms-rot{rotation}.pdf", **options
        )
        assert "Inside Form" in result.text and "Cropped rotated page" in result.text
        page = result.pages[0]
        assert page["rotation"] == rotation
        assert page["form_count"] == 2
        assert len(page["paths"]) == 2 and len(page["images"]) == 1
        assert page["paths"][0]["object_path"] == [0, 0, 0]
        assert page["paths"][0]["clip_bounds"] is not None
        assert all(character["visual_loose_bounds"] for character in page["characters"])
    print(
        "Installed wheel: 3 PDF goldens, 24 concurrent calls, page selection, stream ownership, errors and password pass"
    )
    print("Installed wheel: 4 crop/rotation/nested-Form fixtures preserve text and geometry")


if __name__ == "__main__":
    main()
