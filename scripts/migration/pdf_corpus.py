"""构建不依赖外部字体的 PDF 事实提取基线。"""

from io import BytesIO


def generate_pdfs():
    """生成表格/多栏/旋转、扫描和混合 PDF；固定 ReportLab 元数据。"""
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    scan = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(scan).text((20, 20), "Scanned page 123", fill="black")
    result = {}
    for kind in ("native", "scan", "mixed"):
        output = BytesIO()
        canvas = Canvas(output, pagesize=(400, 400), invariant=1)
        if kind != "scan":
            for index in range(5):
                canvas.drawString(20, 370 - index * 16, f"Left row {index}")
                canvas.drawString(220, 370 - index * 16, f"Right row {index}")
            for x in (20, 100, 180):
                canvas.line(x, 80, x, 200)
            for y in (80, 120, 160, 200):
                canvas.line(20, y, 180, y)
            for row in range(3):
                for col in range(2):
                    canvas.drawString(25 + col * 80, 180 - row * 40, f"{row}:{col}")
            canvas.saveState()
            canvas.translate(350, 80)
            canvas.rotate(90)
            canvas.drawString(0, 0, "Rotated text")
            canvas.restoreState()
            canvas.linkURL("https://example.test/", (20, 350, 150, 380), relative=0)
            canvas.showPage()
        if kind != "native":
            canvas.drawImage(ImageReader(scan), 0, 0, 400, 400)
            canvas.showPage()
        if kind == "mixed":
            canvas.drawString(20, 360, "Text after scanned page")
            canvas.showPage()
        canvas.save()
        result[kind] = output.getvalue()
    return result


def extract_facts(data):
    """物化字符、路径、线段、图片框与链接，不初始化模型或宣称 OCR 已完成。"""
    from dataclasses import asdict
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox

    def materialize(value):
        """只转换明确的几何容器，未知对象仍交由 JSON 报错。"""
        if isinstance(value, Bbox):
            return list(value.bbox)
        if isinstance(value, dict):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [materialize(item) for item in value]
        return value

    pages = []
    with PDFDocument(data) as document:
        for index in range(len(document)):
            pages.append({"page_index": index,
                          "chars": document.get_page_chars(index),
                          "drawing_lines": [asdict(item) for item in document.get_page_drawing_lines(index)],
                          "paths": [asdict(item) for item in document.get_page_path_infos(index)],
                          "image_bboxes": document.get_page_image_bboxes(index),
                          "links": [asdict(item) for item in document.get_page_link_annotations(index)]})
    return materialize({"capture": "pdf-facts", "pages": pages})
