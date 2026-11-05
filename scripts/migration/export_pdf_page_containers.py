"""Actual PDF geometry through index/Form/code/table stage; no Form metadata injection."""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_page_containers import page_reference
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA

def main():
    pdf=ROOT/'build/migration/page-containers.pdf';target=ROOT/'build/migration/pdf-page-containers.json'
    if pdf.exists()or target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from loguru import logger
    logger.remove();labels=[('paged',0),('leaders',0),('unpaged',0),('paged',90),('paged',180),('paged',270),('prose',0),('table',0)];c=Canvas(str(pdf),pagesize=(500,600),pageCompression=0)
    for kind,rotation in labels:
        c.setPageRotation(rotation);c.setFont('Helvetica-Bold',20);c.drawString(200,530,'Contents');c.setFont('Helvetica',10)
        for i in range(7):
            if kind=='prose':c.drawString(40,490-i*20,'Ordinary prose describes the original document and its content.')
            elif kind=='unpaged':c.drawString(40,490-i*20,f'{i+1}. Original topic heading')
            elif kind=='table':
                c.drawString(40,490-i*20,'Amount');c.drawRightString(300,490-i*20,str(2000+i));c.drawRightString(460,490-i*20,str(4000+i))
            else:
                c.drawString(40,490-i*20,'Original topic heading'+(' ...............'if kind=='leaders'else''));c.drawRightString(460,490-i*20,str(i+1))
        if kind=='table':
            for y in [500,480,460,440,420,400,380,360]:c.line(35,y,465,y)
            for x in [35,200,350,465]:c.line(x,360,x,500)
        c.showPage()
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,(kind,rotation)in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);fields=list(dict.fromkeys(FIELDS+EXTRA+['median_glyph_width','ink_bbox','run_index']))
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in fields}for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,tight_bboxes={str(ch['char_idx']):ch.get('tight_bbox')for ch in g.chars},origins={str(ch['char_idx']):ch.get('origin')for ch in g.chars})
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=kind+'/'+str(rotation),request=q,expected=page_reference(q)))
    previous=json.loads((ROOT/'build/migration/pdf-code-table-page.json').read_text(encoding='utf-8'))
    for c in previous['cases']:
        q=copy.deepcopy(c['request']);cases.append(dict(c,request=q,expected=page_reference(q)))
    refs=json.loads((ROOT/'build/migration/native-index-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,scope='75 distinct actual pages: new 8 index/prose/table and previous 67. Form boxes/ownership are empty, so real Form positives are not claimed; full Form metadata extraction/normalization is pending.',cases=cases));print('PDF pages',len(cases),'index',sum(bool(c['expected']['index_blocks'])for c in cases),'tables',sum(bool(c['expected']['table_blocks'])for c in cases),'code',sum(bool(c['expected']['rule_code_blocks']or c['expected']['code_blocks'])for c in cases))

if __name__=='__main__':main()
