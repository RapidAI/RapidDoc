"""Live PDF raw source -> full ordinary producer and combined ruled page reference."""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_rule_interval_pipeline import reference as ordinary
from export_physical_table_page import reference as page
def main():
    target=ROOT/'build/migration/pdf-rule-interval-pipeline.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-rule-interval-pipeline';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'ordinary.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,270]:
        for kind in ['ordinary','caption-note','prose','interrupted']:
            c.setPageRotation(rotation);c.setFont('Helvetica',10)
            for y in [250,290,330,370]:c.line(40,y,340,y)
            for y in [260,300,340]:
                if kind=='interrupted'and y==300:continue
                for x in [50,200]:c.drawString(x,y,'These ordinary words occupy wide columns.'if kind=='prose'else'cell')
            if kind=='caption-note':c.drawString(40,385,'Table 1: Data');c.setFont('Helvetica',8);c.drawString(40,237,'Note: measured values')
            c.showPage();labels.append(str(rotation)+'-'+kind)
    c.save();previous=json.loads((ROOT/'build/migration/pdf-closed-rule-pipeline.json').read_text(encoding='utf-8'));cases=[]
    inputs=[(pdf,i,label)for i,label in enumerate(labels)]+[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'])for c in previous['cases']]
    for path,i,label in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+['visual_row_id']}for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,angle=0,median_height=None,excluded_bboxes=[])
            cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=dict(ordinary=ordinary(q),page=page(q,ordinary=True))))
    refs=json.loads((ROOT/'build/migration/native-ruled-table-page.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'pages;',sum(bool(c['expected']['ordinary']['candidates'])for c in cases),'ordinary;',sum(bool(c['expected']['page']['candidates'])for c in cases),'combined positive')
if __name__=='__main__':main()
