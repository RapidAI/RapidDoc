"""Actual native PDF source/geometry with a positively identified page footnote."""
from dataclasses import asdict
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_document_pages import reference

def main():
    pdf=ROOT/'build/migration/native-footnote-runtime.pdf';target=ROOT/'build/migration/native-footnote-runtime.json'
    if pdf.exists()or target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from loguru import logger
    logger.remove();buf=BytesIO();c=Canvas(buf,pagesize=(400,600));c.setFont('Helvetica',12)
    for i in range(6):c.drawString(30,440-i*28,'Original body paragraph with complete native words')
    c.line(30,110,150,110);c.setFont('Helvetica',8);c.drawString(30,90,'1. Original native footnote with smaller body text');c.drawString(30,78,'Additional original continuation of the same footnote');c.save();data=buf.getvalue()
    with PDFDocument(data)as doc:
        snap=doc._extract_native_page(0);g=doc.get_page_chars_with_geometry(0);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
        s=clean(dict(page_size=snap.page_size,page_index=0,lines=[project(l)for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes,image_bboxes=[info.bbox for info in snap.image_infos],signature_bboxes=[]))
        geometry=clean(dict(chars=g.chars,tight_bboxes=[dict(char_idx=i,bbox=b)for i,b in sorted(g.tight_bboxes.items())],origins=[dict(char_idx=i,origin=o)for i,o in sorted(g.origins.items())],loose_bboxes=[dict(char_idx=i,bbox=b)for i,b in sorted(g.loose_bboxes.items())]))
    q=dict(pages=[dict(source=s,geometry=geometry)],include_post_footnotes=True);expected=reference(q,post_text=True,footnotes=True);groups=expected['pages'][0]['page_footnote_groups'];assert groups;assert sum(l['semantic_type']=='page_footnote'for l in expected['pages'][0]['remaining_lines'])==2
    pdf.write_bytes(data);write_json(target,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),request=q,expected=expected));print('1 actual native PDF, 2 page-footnote lines, groups:',groups)
if __name__=='__main__':main()
