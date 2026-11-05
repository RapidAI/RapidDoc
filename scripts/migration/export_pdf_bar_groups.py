"""Actual PDF bar groups and tiled-table rejection; no frozen intermediate decisions injected."""
import io,sys,json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_groups import reference,group_reference
def main():
    target=ROOT/'build/migration/pdf-bar-groups.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-bar-groups';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'bars.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for compound in [False,True]:
            c.setPageRotation(rotation);c.setFillColorRGB(.2,.4,.7)
            for h in [True,False]:
                p=c.beginPath()
                for i in range(4):
                    b=(50,400-i*20,80+i*20,10)if h else(230+i*20,330,10,60+i*20)
                    if compound:p.rect(*b)
                    else:c.rect(*b,stroke=0,fill=1)
                if compound:c.drawPath(p,stroke=0,fill=1)
            c.setLineWidth(2);c.line(50,335,50,415);c.line(225,330,305,330);c.setFillColorRGB(0,0,0);c.setFont('Helvetica',10)
            for i in range(4):c.drawString(20,400-i*20,str(20*(i+1)));c.drawString(230+i*20,310,str(20*(i+1)))
            c.showPage();labels.append(f'{rotation}-{compound}')
    c.save();cases=[]
    def extract(path,indices):
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            for i,label in indices:
                snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
                s=clean(dict(page_size=snap.page_size,chars=g.chars,path_infos=[asdict(p)for p in snap.path_infos],drawing_lines=[asdict(r)for r in snap.drawing_lines],lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,chars=l.chars)for l in rows]))
                cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=dict(source=s,em=10.),expected=reference(s,10.)))
    extract(pdf,list(enumerate(labels)))
    tiled=json.loads((ROOT/'build/migration/pdf-tiled-tables.json').read_text(encoding='utf-8'));path=ROOT/tiled['cases'][0]['pdf'];assert digest(path.read_bytes())==tiled['cases'][0]['pdf_sha256'];extract(path,[(c['page_index'],'tiled-'+c['label'])for c in tiled['cases']])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_graphic_geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},prefix_ast=group_reference()[1],cases=cases));print('Frozen',len(cases),'actual PDF bar-group chains;',sum(bool(c['expected']['groups'])for c in cases),'group-positive;',sum(len(c['expected']['groups'])for c in cases),'groups;',sum(bool(c['expected']['table_bboxes'])for c in cases),'table positives')
if __name__=='__main__':main()
