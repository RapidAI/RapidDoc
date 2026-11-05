"""Additional true aligned positives; preserve earlier remote-bottom negative PDF fixtures."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_final_candidate_producers import reference as individual
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/pdf-aligned-final-positive.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    pdf=ROOT/'build/migration/pdf-final-candidate-producers/aligned-near-bottom.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    c=Canvas(str(pdf),pagesize=(500,600),pageCompression=0);labels=['base','tail-label','upper-prose']
    for kind in labels:
        c.setFont('Helvetica',10)
        for y in [90,170]:c.line(40,600-y,400,600-y)
        c.drawString(50,495,'Metric')
        for j in range(3):c.drawRightString(240+70*j,495,'Year'+chr(65+j))
        for i in range(3):
            c.drawString(50,600-(123+20*i),'Alpha')
            for j in range(3):c.drawRightString(240+70*j,600-(123+20*i),str(10+i))
        if kind=='tail-label':c.drawString(50,425,'continued label')
        if kind=='upper-prose':c.drawString(40,515,'These ordinary words cross all columns of the table.')
        c.showPage()
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in lines],chars=g.chars,drawing_lines=[asdict(p)for p in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i));q=dict(source=source,excluded_bboxes=[]);result=individual(dict(q,kind='aligned_numeric'))
            if i==0:assert result['candidates'],'base must exercise final regional grid confirmation'
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label='aligned-positive-'+label,kind='aligned_numeric',request=q,expected=dict(individual=result,page=reference(q,ordinary=True,supplemental=True,numeric=True,final_producers=True))))
    refs=json.loads((ROOT/'build/migration/native-final-candidate-producers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,cases=cases));print('Frozen',len(cases),'actual aligned queries;',sum(bool(c['expected']['individual']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
