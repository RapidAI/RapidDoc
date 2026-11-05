"""Live actual PDF -> final two producers and all producer branches; fixed-CJK mismatch retained separately."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_final_candidate_producers import reference as individual
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/pdf-final-candidate-producers.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-final-candidate-producers';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'final.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    c=Canvas(str(pdf),pagesize=(500,600),pageCompression=0);labels=[]
    for kind in ['aligned_numeric','captioned_mixed']:
        for variant in ['base','note','spanning']:
            c.setFont('Helvetica',10)
            if kind=='aligned_numeric':
                for y in [90,180]:c.line(40,600-y,400,600-y)
                c.drawString(50,495,'Metric')
                for j in range(3):c.drawRightString(240+70*j,495,'Year'+chr(65+j))
                for i in range(3):
                    c.drawString(50,600-(123+20*i),'Alpha')
                    for j in range(3):c.drawRightString(240+70*j,600-(123+20*i),str(10+i))
                if variant=='note':c.drawString(50,425,'continued label')
                if variant=='spanning':c.drawString(250,515,'Amounts')
            else:
                c.drawString(50,515,'Table 1: Results')
                shift=10 if variant=='spanning'else 0
                for t,x in [('Item',50),('Amount',180),('Rate',280)]:c.drawString(x,600-(108+shift),t)
                if variant=='spanning':c.drawCentredString(250,505,'Group Amounts')
                for i in range(3):
                    y=133+shift+20*i;c.drawString(50,600-y,'Alpha');c.drawString(180,600-y,'10');c.drawString(280,600-y,'20')
                if variant=='note':c.drawString(50,411,'*a Note about measured values')
            c.showPage();labels.append((kind,variant))
    c.save();old=json.loads((ROOT/'build/migration/pdf-numeric-candidates.json').read_text(encoding='utf-8'));embedded=json.loads((ROOT/'build/migration/pdf-embedded-metric-candidate.json').read_text(encoding='utf-8'));inputs=[(pdf,i,k+'-'+v,k)for i,(k,v)in enumerate(labels)]+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'],None)for c in old['cases']if c['label']!='metric_pair']+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'],None)for c in embedded['cases']];cases=[]
    for path,i,label,kind in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in lines],chars=g.chars,drawing_lines=[asdict(p)for p in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,excluded_bboxes=[]);expected=dict(page=reference(q,ordinary=True,supplemental=True,numeric=True,final_producers=True))
            if kind:expected['individual']=individual(dict(q,kind=kind))
            cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,kind=kind,request=q,expected=expected))
    refs=json.loads((ROOT/'build/migration/native-final-candidate-producers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'live PDF queries;',sum(bool(c['expected']['page']['candidates'])for c in cases),'joint positive;',[(c['label'],len(c['expected']['individual']['candidates']))for c in cases if c['kind']])
if __name__=='__main__':main()
