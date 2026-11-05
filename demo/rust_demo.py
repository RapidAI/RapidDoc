"""修改 FILES，使用项目 .venv 直接运行；自动选择 PDF/Office 原生解析器。"""
import time
from pathlib import Path

from rapid_doc_rust import RapidDoc

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    # 本地相对路径在这里加 ROOT，URL 直接写字符串。
    ROOT / "demo/ms_office_docs/docx_01.doc",
    ROOT / "demo/ms_office_docs/docx_01.docx",
    ROOT / "demo/ms_office_docs/xlsx_01.xls",
    ROOT / "demo/ms_office_docs/xlsx_01.xlsx",
    ROOT / "demo/ms_office_docs/pptx_01.ppt",
    ROOT / "demo/ms_office_docs/pptx_01.pptx",
    ROOT / "demo/pdfs/示例1-论文模板.pdf",
    ROOT / "demo/pdfs/49－长表格-密集型.pdf",
    ROOT / "demo/pdfs/比亚迪财报.pdf",
    ROOT / "demo/images/table_10.png",
    ROOT / "demo/images/seal_text_det.png",


]

if __name__ == "__main__":
    output = ROOT / "demo/output/rust-documents"
    engine = RapidDoc(output_dir=output)
    for source in FILES:
        start_time = time.time()
        label = source.name if isinstance(source, Path) else source
        result = engine(source)
        print(f"总运行时间: {time.time() - start_time}秒")
        print(f"{label}: {len(result.middle_json['pdf_info'])} logical pages, {len(result.images)} assets")
    print(f"Output: {output}")
