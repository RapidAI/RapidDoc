"""真实 wheel 的已知区域表格恢复测试，使用独立安装环境。"""

from pathlib import Path
import sys
from rapid_doc_rust import recover_pdf_table, RapidDocError


def main():
    """验证原 PDF 单元格文本、合并表头和无效区域错误。"""
    root = Path(sys.argv[1]).resolve()
    result = recover_pdf_table(
        root / "tests/migration/inputs/pdf_native.pdf", [20.0, 200.0, 180.0, 320.0]
    )
    assert result.page_index == 0
    assert (result.table["rows"], result.table["columns"]) == (3, 2)
    assert [cell["text"] for cell in result.table["cells"]] == [
        "0:0",
        "0:1",
        "1:0",
        "1:1",
        "2:0",
        "2:1",
    ]
    assert result.html.count("<tr>") == 3 and result.html.count("<td") == 6
    source = root / "build/migration/geometry/merged-header.pdf"
    merged = recover_pdf_table(source, [20.0, 40.0, 180.0, 140.0])
    assert len(merged.table["cells"]) == 3
    assert merged.table["cells"][0]["column_span"] == 2
    assert merged.table["cells"][0]["text"] == "merged header"
    assert '<td colspan="2">merged header</td>' in merged.html
    assert [cell["text"] for cell in merged.table["cells"]][1:] == ["left", "right"]
    try:
        recover_pdf_table(source, [0.0, 0.0, 10.0, 10.0])
    except RapidDocError as error:
        assert error.code == "invalid_input"
    else:
        raise AssertionError("Unruled region accepted")
    print(
        "Installed wheel: physical table text, merged header, HTML and invalid region rejection pass"
    )


if __name__ == "__main__":
    main()
