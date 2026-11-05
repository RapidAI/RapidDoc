"""Real PDFium page sources with repeated headers, footers, numbers and a separator."""
from dataclasses import asdict
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_marginals import reference
from export_native_late_image_text import project
from export_native_inline_clusters import clean
def main():
    pdf=ROOT/'build/migration/native-marginal-runtime.pdf';target=ROOT/'build/migration/native-marginal-runtime.json'
    if pdf.exists()or target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from loguru import logger
    logger.remove();buf=BytesIO();c=Canvas(buf,pagesize=(400,600),pageCompression=1)
    for page in range(3):
        c.setFont('Helvetica',10);c.drawString(30,580,'Original Journal Research');c.drawString(30,565,'Volume 12 Research Article');c.line(20,550,380,550)
        for j in range(5):c.drawString(30,430-j*22,'Original body paragraph with complete native words')
        c.drawString(30,25,'Original publication footer');c.drawString(350,25,str(page+1));c.showPage()
    c.save();data=buf.getvalue();sources=[]
    with PDFDocument(data)as doc:
        for i in range(3):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            sources.append(clean(dict(page_size=snap.page_size,page_index=i,lines=[project(l)for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes,image_bboxes=[info.bbox for info in snap.image_infos],signature_bboxes=[])))
    expected=reference(sources);assert sum(l['semantic_type']=='header'for s in expected['sources']for l in s['lines'])==6
    assert sum(len(bs)for bs in expected['header_separator_bboxes'])==3
    pdf.write_bytes(data);write_json(target,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),request=dict(sources=sources),expected=expected));print('3 real pages, 6 headers, 3 footers, 3 page numbers, 3 separator exclusions')
if __name__=='__main__':main()
