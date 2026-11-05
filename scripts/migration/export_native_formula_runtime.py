"""Two actual PDFium pages through original formula-input recovery."""
from dataclasses import asdict
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_formula_input import reference

def main():
    target=ROOT/'build/migration/native-formula-runtime.json';pdf=ROOT/'build/migration/native-formula-runtime.pdf'
    if target.exists()or pdf.exists():raise FileExistsError('immutable formula runtime exists')
    from loguru import logger
    logger.remove()
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    buf=BytesIO();c=Canvas(buf,pagesize=(400,600))
    for page in range(2):
        c.setFont('Helvetica',10)
        for y in [520,500,480,340,320,300]:c.drawString(30,y,'Original complete body paragraph with ordinary text')
        c.setFont('Helvetica-Oblique',10);c.drawString(130,410,'x = y + z');c.drawString(350,410,'(1)')
        if page:
            c.setFont('Helvetica-Oblique',8);c.drawString(165,425,'a');c.drawString(165,398,'b');c.setLineWidth(.5);c.line(160,407,185,407)
        c.showPage()
    c.save();data=buf.getvalue();cases=[]
    with PDFDocument(data)as doc:
        for index in range(2):
            s=doc._extract_native_page(index);g=doc.get_page_chars_with_geometry(index);ls=_build_native_line_items_from_chars(g.chars,s.page_size,page_rotation=s.rotation)
            q=clean(dict(lines=[project(l)for l in ls],page_size=s.page_size,table_bboxes=[],drawing_lines=[asdict(r)for r in s.drawing_lines]));out=reference(q)
            assert out['formula_blocks'],index
            cases.append(dict(label='live-formula/'+str(index),request=q,expected=out))
    pdf.write_bytes(data);write_json(target,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),cases=cases));print('2 actual PDFium pages, both with positive formula recovery')
if __name__=='__main__':main()
