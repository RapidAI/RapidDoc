"""Page-clipped images, crop coordinates and damaged graphics-state fallback."""
import json
from dataclasses import asdict
from io import BytesIO
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    pdf=ROOT/'build/migration/form-runtime-edges.pdf';target=ROOT/'build/migration/native-form-runtime-edges.json'
    if pdf.exists()or target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import DecodedStreamObject,NameObject,RectangleObject
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);c.beginForm('image',-100,-100,500,700);c.drawImage(ImageReader(Image.new('RGB',(3,3),(128,70,30))),-50,100,500,200);c.setFont('Helvetica',10);c.drawString(20,350,'Native picture label');c.endForm()
    for r in [0,90,270]:c.setPageRotation(r);c.doForm('image');c.showPage()
    c.save();reader=PdfReader(pdf);writer=PdfWriter();writer.append(reader);bad=writer.pages[2];stream=DecodedStreamObject();stream.set_data(b'Q\n'+bad.get_contents().get_data());bad[NameObject('/Contents')]=writer._add_object(stream);writer.pages[1].cropbox=RectangleObject([20,30,380,570]);buf=BytesIO();writer.write(buf);pdf.write_bytes(buf.getvalue());data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i in range(3):
            snap=doc._extract_native_page(i)
            with doc._open_page(i)as page:frame=page.get_bbox()
            cases.append(dict(page_index=i,label=['image-page-clip','crop-rotated-image','damaged-Q-fallback'][i],expected=clean(dict(page_size=snap.page_size,page_bbox=frame,rotation=snap.rotation,form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes))))
    refs=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text())['sources'];write_json(target,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),sources=refs,geometry=cases));print('Form runtime edge pages',len(cases))
if __name__=='__main__':main()
