"""Real PDF source with explicit incoming candidates for positive final-policy branch coverage."""
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_native_table_candidate_merge import candidate
from export_native_table_page_policies import reference
from export_physical_table_page import reference as producers

def main():
    target=ROOT/'build/migration/pdf-table-page-policies-positive.json';pdf=ROOT/'build/migration/pdf-final-candidate-producers/page-policies-positive.pdf'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    if not pdf.exists():
        c=Canvas(str(pdf),pagesize=(500,600),pageCompression=0)
        def text(x,y,t,size=10,width=None):
            o=c.beginText(x,600-y);o.setFont('Helvetica',size)
            if width:o.setHorizScale(100*width/stringWidth(t,'Helvetica',size))
            o.textOut(t);c.drawText(o)
        for top,bottom in [(100,150),(160,220)]:
            for j in range(4):y=top+(bottom-top)*j/3;c.line(40,600-y,240,600-y)
            for x in [40,140,240]:c.line(x,600-top,x,600-bottom)
            for i in range(3):
                for x in [50,160]:text(x,top+(bottom-top)*(i+.65)/3,'10')
        text(40,235,'Table 1: Results');c.showPage()
        for i in range(3):
            for x in [45,330]:text(x,120+i*20,'10')
        for i in range(3):text(40,185+i*26,'These ordinary words describe a separate prose paragraph after the table data',12,width=350)
        c.showPage()
        text(40,90,'Table 1 continued')
        for y in [100,140,180,220]:c.line(40,600-y,240,600-y)
        for x in [40,140,240]:c.line(x,500,x,380)
        for i in range(3):
            for x in [50,160]:text(x,120+i*40,'10')
        c.showPage();c.save()
    data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,op in enumerate(['join','trim','externalize']):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            s=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(r)for r in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=i))
            ids=[l['source_index']for l in s['lines']]
            cs=[candidate([40.,100.,240.,150.],ids=ids),candidate([40.,160.,240.,220.],ids=[])]if i==0 else[candidate([40.,100. if i==1 else 80.,440. if i==1 else 240.,260. if i==1 else 220.],ids=ids)]
            q=dict(source=s,candidates=cs,operation=op);expected=reference(q)
            assert expected['candidates']!=cs,op+' must change incoming candidate'
            if op=='join':assert len(expected['candidates'])==1
            if op=='trim':assert expected['candidates'][0]['bbox'][3]<170.
            if op=='externalize':assert 99.<=expected['candidates'][0]['core_bbox'][1]<=101.
            fullq=dict(source=s,excluded_bboxes=[]);full=reference(fullq,True);before=producers(fullq,ordinary=True,supplemental=True,numeric=True,final_producers=True)
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=op,request=q,expected=expected,complete_expected=full,complete_changed=before!=full))
    import json
    refs=json.loads((ROOT/'build/migration/native-table-page-policies.json').read_text(encoding='utf-8'))['sources']
    write_json(target,dict(sources=refs,scope='3 live PDF sources, positive standalone policy calls with explicit incoming candidate boxes; complete detector also checked independently.',cases=cases));print('Frozen real policy positives',len(cases),'complete detector changes',sum(c['complete_changed']for c in cases))
if __name__=='__main__':main()
