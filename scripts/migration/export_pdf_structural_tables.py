"""Live PDF geometry -> expanded structural table pipeline, no frozen row/core/caption verdicts."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_physical_table_page import reference
from export_supplemental_rule_tables import reference as supplemental
from export_shaded_table_pipeline import reference as shaded
from export_native_form_tables import reference as forms
def main():
    target=ROOT/'build/migration/pdf-structural-tables.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-structural-tables';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'structural.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);kinds=['open_vertical','short_caption','shaded_header','blank_answer','page_clipped']
    for kind in kinds:
        c.setFont('Helvetica',10)
        if kind=='open_vertical':
            for y in [100,140,180,220,260]:c.line(40,600-y,340,600-y)
            for x in [140,240]:c.line(x,340,x,500)
            for i in range(4):
                for j in range(3):c.drawString(50.+100*j,600-(113.+40*i),'cell')
            c.drawString(40,515,'Table 1: Results')
        if kind=='short_caption':
            for y in [100,130,220]:c.line(40,600-y,340,600-y)
            for i in range(6):c.drawString(50,600-(113+20*i),'item')
            c.drawString(40,365,'Table 1: Results')
        if kind=='shaded_header':
            c.setFillColorRGB(.85,.85,.85)
            for j in range(3):c.rect(50+100*j,487,100,15,fill=1,stroke=0)
            c.setFillColorRGB(0,0,0)
            for j in range(3):c.drawString(50+100*j,487,'Header')
            for i in range(4):
                for j in range(3):c.drawString(50+100*j,600-(133+25*i),str(i*3+j))
        if kind=='blank_answer':
            for y in [100,220]:c.line(40,600-y,340,600-y)
            c.setFont('Helvetica-Bold',10);c.drawString(50,487,'Name                  Answer');c.setFont('Helvetica',10)
            for i in range(5):c.drawString(50,600-(133+20*i),'item')
            c.drawString(40,515,'Table 1: Results')
        if kind=='page_clipped':
            for y in [100,130,150,170,190,210,230]:c.line(40,600-y,400,600-y)
            for x in [40,140,240]:c.line(x,370,x,500)
            c.setFont('Helvetica',8)
            for i in range(2):
                for j in range(2):c.drawString(60+200*j,600-(111+12*i),'Header')
            for i in range(5):c.drawString(60,600-(143+20*i),'item')
        c.showPage()
    c.save();old=json.loads((ROOT/'build/migration/pdf-rule-interval-pipeline.json').read_text(encoding='utf-8'));inputs=[(pdf,i,k,k)for i,k in enumerate(kinds)]+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'],None)for c in old['cases']];cases=[]
    for path,i,label,kind in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+['visual_row_id','dominant_font_weight','paragraph_terminal','semantic_type','caption_start']}for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,excluded_bboxes=[]);expected=dict(page=reference(q,ordinary=True,supplemental=True))
            if kind:
                call=shaded if kind=='shaded_header'else forms if kind in ['blank_answer','page_clipped']else supplemental;expected['individual']=call(dict(q,kind=kind))
            cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,kind=kind,request=q,expected=expected))
    p='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py';write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())},cases=cases));print('Frozen',len(cases),'pages;',sum(bool(c['expected']['page']['candidates'])for c in cases),'combined positive;',[(c['kind'],len(c['expected']['individual']['candidates']))for c in cases if c['kind']])
if __name__=='__main__':main()
