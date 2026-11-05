"""生成包含嵌套 Form、非等比变换、图片、CropBox 和页面旋转的 PDF。"""

from io import BytesIO
from pathlib import Path
from common import ROOT


def generate(output=None):
    """固定 PDF 元数据；只写入自生成内容，四种页旋转分别保存。"""
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    output = Path(output or ROOT / "build/migration/geometry")
    output.mkdir(parents=True, exist_ok=True)
    stream = BytesIO()
    canvas = Canvas(stream, pagesize=(400, 500), invariant=1)
    canvas.beginForm("inner", 0, 0, 100, 100)
    canvas.setLineWidth(2)
    canvas.line(10, 10, 90, 10)
    canvas.line(10, 10, 10, 90)
    canvas.drawString(15, 50, "Inside Form")
    canvas.drawImage(ImageReader(Image.new("RGB", (8, 8), "green")), 20, 20, 10, 10)
    canvas.endForm()
    canvas.beginForm("outer", 0, 0, 200, 200)
    canvas.saveState()
    canvas.translate(20, 30)
    canvas.scale(1.5, 2)
    canvas.doForm("inner")
    canvas.restoreState()
    canvas.endForm()
    canvas.saveState()
    canvas.translate(30, 40)
    canvas.doForm("outer")
    canvas.restoreState()
    canvas.drawString(50, 420, "Cropped rotated page")
    canvas.save()
    paths = []
    for rotation in (0, 90, 180, 270):
        writer = PdfWriter()
        page = writer.add_page(PdfReader(BytesIO(stream.getvalue())).pages[0])
        page.cropbox.lower_left = (10, 20)
        page.cropbox.upper_right = (390, 480)
        if rotation:
            page.rotate(rotation)
        path = output / f"forms-rot{rotation}.pdf"
        writer.write(path)
        paths.append(path)
    merged=BytesIO()
    canvas=Canvas(merged,pagesize=(200,160),invariant=1)
    for y in (20,70,120):canvas.line(20,y,180,y)
    for x in (20,180):canvas.line(x,20,x,120)
    canvas.line(100,20,100,70)
    canvas.drawString(30,95,"merged header")
    canvas.drawString(30,45,"left")
    canvas.drawString(110,45,"right")
    canvas.save()
    (output/"merged-header.pdf").write_bytes(merged.getvalue())
    return paths


if __name__ == "__main__":
    print("\n".join(str(path) for path in generate()))
