"""Actual PDF raw geometry through five new producers and joint structural arbitration."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_physical_table_page import reference
from export_numeric_candidate_segments import reference as segments
from export_text_numeric_candidates import reference as text
EXTRA=['visual_row_id','dominant_font_weight','paragraph_terminal','semantic_type','caption_start','formula_candidate_only','compact_formula_cluster']
def main():
    target=ROOT/'build/migration/pdf-numeric-candidates.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-numeric-candidates';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'numeric.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);kinds=['partial_grid','year_header','metric_pair','unruled_numeric','captioned_pair']
    for kind in kinds:
        c.setFont('Helvetica',10)
        if kind=='partial_grid':
            c.line(40,500,340,500)
            for y in [140,180]:c.line(140,600-y,340,600-y)
            for x in [40,140,240,340]:c.line(x,360,x,500)
            for i in range(3):
                for j in range(3):c.drawString(60+100*j,600-(118+40*i),'cell')
        if kind=='year_header':
            c.line(40,490,380,490);c.drawString(50,495,'Metric')
            for j in range(3):c.drawRightString(240+60*j,495,str(2020+j))
            for i in range(3):
                c.drawString(50,600-(123+20*i),'label')
                for j in range(3):c.drawRightString(240+60*j,600-(123+20*i),str(10+i))
        if kind=='metric_pair':
            c.setFont('STSong-Light',15);c.drawString(50,505,'关键指标');c.line(110,510,350,510);c.setFont('STSong-Light',10)
            for i in range(5):c.drawString(50,600-(108+18*i),'指标一');c.drawRightString(280,600-(108+18*i),'10%')
        if kind=='unruled_numeric':
            c.setFont('Courier-Bold',10);c.drawString(50,490,'A'*40);c.setFont('Courier',10)
            for i in range(3):
                for x in [50,152,254]:c.drawString(x,600-(123+20*i),'10')
        if kind=='captioned_pair':
            for y in [100,220]:c.line(40,600-y,340,600-y)
            c.setFillColorRGB(.3,.3,.3);c.rect(40,480,300,20,fill=1,stroke=0);c.setFillColorRGB(1,1,1);c.setFont('Courier-Bold',10);c.drawString(50,487,'Header'*6);c.setFillColorRGB(0,0,0);c.setFont('Courier',10)
            for i in range(4):c.drawString(50,600-(138+25*i),'A'*22);c.drawString(200,600-(138+25*i),'Data')
            c.drawString(40,365,'Table 1: Data')
        c.showPage()
    c.save();old=json.loads((ROOT/'build/migration/pdf-structural-tables.json').read_text(encoding='utf-8'));inputs=[(pdf,i,k,k)for i,k in enumerate(kinds)]+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'],None)for c in old['cases']];cases=[]
    for path,i,label,kind in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,excluded_bboxes=[]);expected=dict(page=reference(q,ordinary=True,supplemental=True,numeric=True))
            if kind:expected['individual']=(segments if kind in ['partial_grid','year_header','metric_pair']else text)(dict(q,kind=kind))
            cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,kind=kind,request=q,expected=expected))
    refs=json.loads((ROOT/'build/migration/native-numeric-candidate-segments.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'pages;',sum(bool(c['expected']['page']['candidates'])for c in cases),'joint positive;',[(c['kind'],len(c['expected']['individual']['candidates']))for c in cases if c['kind']])
if __name__=='__main__':main()
