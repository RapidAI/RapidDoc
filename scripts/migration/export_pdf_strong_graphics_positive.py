"""Actual complex contour PDF positives through complete old strong-graphic orchestration."""
import math,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_native_strong_graphics import reference
def main():
    target=ROOT/'build/migration/pdf-strong-graphics-positive.json';pdf=ROOT/'build/migration/strong-graphics-contours.pdf'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    if not pdf.exists():
        c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0)
        for rotation in [0,90,180,270]:
            c.setPageRotation(rotation)
            for j in range(3):
                c.setFillColorRGB(.15+j*.2,.2,.7-j*.15);p=c.beginPath()
                for i in range(25):
                    t=math.pi*2*i/24;x=80+j*38+20*math.cos(t);y=440+20*math.sin(t)
                    if i==0:p.moveTo(x,y)
                    else:p.lineTo(x,y)
                p.close();c.drawPath(p,fill=1,stroke=0)
            c.setFont('Helvetica',10);c.drawString(40,100,'Unrelated body text');c.showPage()
        c.save()
    data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i in range(4):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            s=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=s);expected=reference(q)
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label='contours-rotation-'+str(snap.rotation),request=q,expected=expected))
    assert any(c['expected']['graphic_bboxes']for c in cases)
    refs=json.loads((ROOT/'build/migration/native-strong-graphics-complete.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,scope='Four actual PDF rotations with real complex filled contours; complete old strong detector, actual accepted and rejected rotations, no candidate injection.',cases=cases));print('Frozen actual strong rotations',len(cases),'positive',sum(bool(c['expected']['graphic_bboxes'])for c in cases))
if __name__=='__main__':main()
