"""Full actual PDF source -> automatic closed physical candidate chain, immutable references."""
import json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_closed_rule_pipeline import references
from export_physical_table_page import reference
from export_pdf_table_footnotes import FIELDS
from export_native_table_candidate_merge import candidate

def main():
    target=ROOT/'build/migration/pdf-closed-rule-pipeline.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-closed-rule-pipeline';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'closed-pipeline.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    canvas=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for caption in [False,True]:
        for kind in ['full','two-boundaries','sparse','missing-side']:
            canvas.setFont('Helvetica',10)
            if caption:canvas.drawString(40,385,'Table 1: Results')
            for y in ([250,370]if kind in ['two-boundaries','sparse']else[250,290,330,370]):canvas.line(40,y,240,y)
            for x in ([40,140]if kind=='missing-side'else[40,140,240]):canvas.line(x,250,x,370)
            for y in [260,300,340]:
                for x in ([60]if kind=='sparse'else[60,160]):canvas.drawString(x,y,'cell')
            if caption:canvas.setFont('Helvetica',8);canvas.drawString(40,237,'Note: measured values')
            canvas.showPage();labels.append(str(caption)+'-'+kind)
    canvas.save()
    prior=json.loads((ROOT/'build/migration/pdf-closed-grids.json').read_text(encoding='utf-8'))
    inputs=[(pdf,i,label)for i,label in enumerate(labels)]+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'])for c in prior['cases'][::2]]
    cases=[]
    for path,i,label in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+['visual_row_id']}for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i))
            # Explicit existing candidate only for expansion; inspection/detection use raw source alone.
            seed=candidate([40.,270.,240.,310.],score=80.,ids=[],core=[40.,270.,240.,310.])
            q=dict(source=source,angle=0,median_height=None,excluded_bboxes=[],candidates=[seed]);ins,det,exp=references(q)
            cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=dict(inspect=ins,detect=det,expand=exp,page=reference(q))))
    refs=json.loads((ROOT/'build/migration/native-physical-table-page.json').read_text(encoding='utf-8'))['sources']
    write_json(target,dict(sources=refs,scope='24 live PDF pages. Original actual chars/paths/drawing lines/visual lines -> full automatic row grouping/captions/closed branch and physical-page arbitration. Explicit seed only for grid expansion. Form/obstacle input empty; no full general detector/default PDF claim.',cases=cases))
    print('Frozen',len(cases),'pages;',sum(bool(c['expected']['detect']['candidates'])for c in cases),'closed positive;',sum(bool(c['expected']['page']['candidates'])for c in cases),'physical page positive')
if __name__=='__main__':main()
