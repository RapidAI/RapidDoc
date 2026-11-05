"""生成不含外部文档的六格式烟雾语料；不冒充缺失的历史样本。"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import json
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED
from common import ROOT, digest, environment, write_json


def canonical_zip(blob):
    """固定 ZIP 成员顺序和时间，消除生成时间造成的输入哈希变化。"""
    result = BytesIO()
    with ZipFile(BytesIO(blob)) as source, ZipFile(result, "w") as target:
        for name in sorted(source.namelist()):
            info = ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            target.writestr(info, source.read(name))
    return result.getvalue()


def write_input(path, data):
    """内容相同则保留现有文件，避免无意义改写被查看器打开的样本。"""
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)


def generate():
    """生成中英文、样式、表格、图片及旧格式最小文件，记录生成器哈希。"""
    from docx import Document
    from pptx import Presentation
    from pptx.util import Inches
    from openpyxl import Workbook
    from PIL import Image
    sys.path.insert(0, str(ROOT / "tests/native_office"))
    from _legacy_doc_test_utils import build_doc
    from _legacy_ppt_test_utils import build_sparse_notes_ppt
    from _legacy_xls_test_utils import SheetFixture, build_xls, label_cell, merged_cells

    stamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    picture = BytesIO()
    Image.new("RGB", (16, 16), "blue").save(picture, format="PNG")
    doc = Document()
    doc.core_properties.created = doc.core_properties.modified = stamp
    doc.add_paragraph("中文 English  spacing").add_run(" bold").bold = True
    table = doc.add_table(rows=2, cols=2)
    for index, cell in enumerate(table._cells):
        cell.text = str(index)
    doc.add_picture(BytesIO(picture.getvalue()))
    deck = Presentation()
    deck.core_properties.created = deck.core_properties.modified = stamp
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1)).text = "中文 English slide"
    slide.shapes.add_picture(BytesIO(picture.getvalue()), Inches(1), Inches(3))
    book = Workbook()
    book.properties.created = book.properties.modified = stamp.replace(tzinfo=None)
    book.active.append(["中文", "English"])
    book.active.append([12.5, "merged"])
    book.active.merge_cells("B2:C2")
    values = {}
    for suffix, document in (("docx", doc), ("pptx", deck), ("xlsx", book)):
        stream = BytesIO()
        document.save(stream)
        # openpyxl 会在保存时重置 modified，统一核心属性后再规范 ZIP。
        data = stream.getvalue()
        if suffix == "xlsx":
            import re
            fixed = BytesIO()
            with ZipFile(BytesIO(data)) as source, ZipFile(fixed, "w") as target:
                for name in source.namelist():
                    body = source.read(name)
                    if name == "docProps/core.xml":
                        body = re.sub(rb'(<dcterms:modified[^>]*>).*?(</dcterms:modified>)', rb'\g<1>2000-01-01T00:00:00Z\2', body)
                    target.writestr(name, body)
            data = fixed.getvalue()
        values[suffix] = canonical_zip(data)
    values["doc"] = build_doc("中文 English\rSecond paragraph\r")
    values["ppt"] = build_sparse_notes_ppt()
    values["xls"] = build_xls([SheetFixture("Data", label_cell(0, 0, "中文 English") + label_cell(1, 0, "merged") + merged_cells((1, 0, 1, 1)))])
    generator_paths = [Path(__file__), Path(__file__).with_name("pdf_corpus.py"), *sorted((ROOT / "tests/native_office").glob("*_test_utils.py"))]
    provenance = {p.relative_to(ROOT).as_posix(): digest(p.read_bytes()) for p in generator_paths}
    cases = []
    for suffix, data in sorted(values.items()):
        path = ROOT / f"tests/migration/inputs/synthetic.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_input(path, data)
        case_id = "synthetic_" + suffix
        cases.append({"id": case_id, "input": path.relative_to(ROOT).as_posix(),
                      "sha256": digest(data), "format": suffix,
                      "license": "Repository-generated fixture; no external document content",
                      "source": "scripts/migration/generate_corpus.py", "generator_sha256": provenance,
                      "tags": ["synthetic", "smoke", "native-office"], "config": {},
                      "model_manifest": [], "expected": case_id + ".json"})
    from pdf_corpus import generate_pdfs
    for kind, data in generate_pdfs().items():
        case_id = "pdf_" + kind
        path = ROOT / f"tests/migration/inputs/{case_id}.pdf"
        write_input(path, data)
        cases.append({"id": case_id, "input": path.relative_to(ROOT).as_posix(),
                      "sha256": digest(data), "format": "pdf", "capture": "pdf-facts",
                      "license": "Repository-generated fixture; no external document content",
                      "source": "scripts/migration/pdf_corpus.py", "generator_sha256": provenance,
                      "tags": ["synthetic", kind, "pdf-facts"], "config": {},
                      "model_manifest": [], "expected": case_id + ".json"})
    restored = json.loads((ROOT / "tests/fixtures/provenance.json").read_text(encoding="utf-8"))
    for item in restored["files"]:
        path = ROOT / item["target"]
        if path.suffix == ".json":
            continue  # 字符去重片段由原回归测试验证，不作为完整文档解析。
        case_id = "restored_" + path.stem.replace("-", "_") + "_" + path.suffix[1:]
        cases.append({"id": case_id, "input": item["target"], "sha256": item["sha256"],
                      "format": path.suffix[1:], "license": "See tests/fixtures/DOCVORTEX-LICENSE.md and provenance.json",
                      "source": item["source"], "source_commit": restored["source_commit"],
                      "tags": ["restored", "native-office"], "config": {},
                      "model_manifest": [], "expected": case_id + ".json"})
    write_json(ROOT / "tests/migration/corpus_manifest.json", {"schema_version": 1, "cases": cases,
               "generator_packages": environment()["packages"]})


if __name__ == "__main__":
    generate()
