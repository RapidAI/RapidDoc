"""生成原生/扫描同页、同一行、旋转和空白语料；冻结独立 PDF 事实与旧 CPU OCR。"""
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader,PdfWriter
import pypdfium2 as pdfium
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from common import ROOT,write_json,digest
from pdf_corpus import extract_facts
from export_page_blocks import make_ocr

def main():
    output=ROOT/'build/migration/native-fusion';output.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    scan=Image.new('RGB',(330,50),'white');ImageDraw.Draw(scan).text((5,8),'Scanned line 456',font=font,fill='black')
    suffix=Image.new('RGB',(240,50),'white');ImageDraw.Draw(suffix).text((5,8),'Raster 456',font=font,fill='black')
    buffer=BytesIO();canvas=Canvas(buffer,pagesize=(600,500),invariant=1)
    canvas.setFont('Helvetica',24)
    for y,text in [(420,'Native text wins 123'),(370,'Original words and spaces'),(320,'Final native line 789')]:canvas.drawString(60,y,text)
    canvas.showPage();canvas.setFont('Helvetica',24)
    canvas.drawString(60,420,'Native title 123');canvas.drawImage(ImageReader(scan),60,270,width=330,height=50)
    canvas.drawString(60,180,'Native footer 789');canvas.showPage();canvas.setFont('Helvetica',24)
    canvas.drawString(60,360,'Prefix text');canvas.drawImage(ImageReader(suffix),185,345,width=240,height=50);canvas.showPage()
    canvas.drawImage(ImageReader(scan),60,270,width=330,height=50);canvas.showPage();canvas.setFont('Helvetica',24)
    canvas.drawString(60,420,'Rotated native 123');canvas.showPage();canvas.showPage();canvas.save()
    reader=PdfReader(BytesIO(buffer.getvalue()));writer=PdfWriter()
    for index,page in enumerate(reader.pages):
        if index==4:page.rotate(90)
        writer.add_page(page)
    path=output/'mixed.pdf';writer.write(path)
    baseline=extract_facts(path.read_bytes());ocr=make_ocr();cases=[]
    with pdfium.PdfDocument(path)as doc:
        for index,dpi in [(0,72),(1,72),(2,72),(3,72),(4,72),(5,72),(0,137)]:
            page=doc[index];bitmap=page.render(scale=dpi/72)
            image=bitmap.to_pil().convert('RGB');bitmap.close();page.close()
            png=output/f'page-{index}-{dpi}.png';image.save(png)
            cases.append(dict(page_index=index,dpi=dpi,width=image.width,height=image.height,input=str(png),
                              rgb_sha256=digest(image.tobytes()),ocr=ocr(image)))
    assert len(baseline['pages'])==6 and cases[0]['ocr']and cases[1]['ocr']and not cases[5]['ocr']
    write_json(ROOT/'build/migration/native-fusion.json',dict(pdf_input=str(path),facts=baseline,cases=cases))
    print('Exported 7 page/DPI OCR references and 6 independent native PDF facts')
if __name__=='__main__':main()
