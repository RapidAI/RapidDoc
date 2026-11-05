"""Real PDF full bar detection, images, outlines, captions and occupied grids."""
import io,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import reference
from export_native_bar_groups import reference as groups
FIELDS='text bbox angle source_index chars font_signature dominant_font_weight effective_height em_height style_scale_repaired semantic_type title_suppressed paragraph_group visual_row_id'.split()
def main():
    target=ROOT/'build/migration/pdf-bar-graphics.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from PIL import Image
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-bar-graphics';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'full-bars.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in ['caption','image','outline','outline-note','grid']:
            c.setPageRotation(rotation);c.setFillColorRGB(.2,.4,.7)
            for i in range(4):c.rect(50,400-i*20,80+i*20,10,fill=1,stroke=0)
            c.setLineWidth(2);c.line(50,335,50,415);c.setFillColorRGB(0,0,0);c.setFont('Helvetica',10)
            for i in range(4):c.drawString(20,400-i*20,str((i+1)*20))
            if kind=='caption':c.drawString(50,447,'Fig 1: Example Caption');c.drawString(50,432,'Second caption line.')
            if kind=='image':c.drawImage(ImageReader(Image.new('RGB',(60,80),(100,150,200))),170,335,60,80)
            if kind.startswith('outline'):
                p=c.beginPath();p.moveTo(80,330)
                for j in range(1,27):p.lineTo(80+(j%2)*20,330-j*.4)
                p.close();c.drawPath(p,stroke=0,fill=1)
                if kind=='outline-note':c.drawString(80,327,'Source: test')
            if kind=='grid':
                c.setLineWidth(1)
                for y in [340,370,400,410]:c.line(50,y,190,y)
                for x in [50,120,190]:c.line(x,340,x,410)
                for y in [342,375,402]:
                    for x in [65,140]:c.drawString(x,y,'cell')
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();cases=[]
    def extract(path,indices):
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            for i,label in indices:
                snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
                s=clean(dict(page_size=snap.page_size,chars=g.chars,path_infos=[asdict(p)for p in snap.path_infos],drawing_lines=[asdict(r)for r in snap.drawing_lines],lines=[{k:getattr(l,k)for k in FIELDS}for l in rows],image_bboxes=[im.bbox for im in snap.image_infos],form_bboxes=[]))
                expected=reference(s,10.);before=groups(s,10.)['groups'];cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=dict(source=s,em=10.),expected=expected,groups_before=len(before),grid_rejected=bool(before)and not expected['bar_bboxes']))
    extract(pdf,list(enumerate(labels)))
    old=json.loads((ROOT/'build/migration/pdf-bar-groups.json').read_text(encoding='utf-8'));paths={}
    for c in old['cases']:paths.setdefault(c['pdf'],[]).append((c['page_index'],'prior-'+c['label']))
    for name,indices in paths.items():extract(ROOT/name,indices)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/annotation_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/typography.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual full bar PDF chains;',sum(bool(c['expected']['bar_bboxes'])for c in cases),'positive;',sum(c['request']['source']!=c['expected']['source']for c in cases),'mutations;',sum(c['grid_rejected']for c in cases),'grid rejections')
if __name__=='__main__':main()
