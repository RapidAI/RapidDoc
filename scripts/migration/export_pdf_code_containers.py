"""Real PDF glyph/line/path inputs through complete old code builders; retain all rotation outcomes."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_native_code_containers import reference
def main():
    target=ROOT/'build/migration/pdf-code-containers.json';pdf=ROOT/'build/migration/code-containers.pdf'
    if target.exists()or pdf.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for variant,rotation in [('filled',0),('filled',90),('filled',180),('filled',270),('ruled',0),('algorithm',0),('prose',0),('table',0)]:
        c.setPageRotation(rotation);c.setFillColorRGB(0,0,0)
        if variant=='filled':c.setFillColorRGB(.94,.94,.94);c.rect(30,350,340,160,fill=1,stroke=0);c.setFillColorRGB(0,0,0)
        if variant in ['ruled','table']:
            for y in [100,240]:c.line(30,600-y,370,600-y)
        if variant=='table':
            for x in [100,200,300]:c.line(x,360,x,500)
        if variant=='algorithm':c.setFont('Helvetica-Bold',10);c.drawString(40,515,'Algorithm 1: Actual Listing')
        texts=['def function(x):','if x > 0:','return x + 1','return 0','else:','return value']if variant!='algorithm'else['input sequence of values','for each item in sequence','if item is valid','return item as result','end for each item','output result value']
        for i,t in enumerate(texts):
            c.setFont('Courier'if variant=='filled'else'Helvetica',10);x=40+[0,10,20,20,10,0][i]
            if variant=='prose':t='These ordinary words describe a separate prose paragraph'
            c.drawString(x,600-(118+i*20),t)
            if variant=='ruled':c.drawString(31,600-(118+i*20),str(i+1))
        c.showPage();labels.append((variant,rotation))
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,(variant,rotation)in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            s=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA+['median_glyph_width']}for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i))
            for kind in ['filled','ruled']:
                q=dict(source=s,kind=kind,excluded_bboxes=[],claimed_line_indices=[]);cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=variant+'-'+str(rotation)+'/'+kind,request=q,expected=reference(q)))
    assert any(c['expected']['code_blocks']for c in cases)
    refs=json.loads((ROOT/'build/migration/native-code-containers.json').read_text(encoding='utf-8'))['sources'];write_json(target,dict(sources=refs,scope='8 real PDF pages /16 complete filled+ruled builder queries; four rotations and real framed/algorithm/prose/table source evidence, all old acceptance/rejection outcomes retained.',cases=cases));print('Frozen actual code pages',8,'queries',len(cases),'positives',[(c['label'],len(c['expected']['code_blocks']))for c in cases if c['expected']['code_blocks']])
if __name__=='__main__':main()
