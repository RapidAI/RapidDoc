"""Freeze actual PDF concentric contour candidates and existing graphic arbitration."""
import ast,copy,io,json,sys
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-concentric-graphics.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _native_multicolumn_grid_bboxes as grids
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_detect_strong_graphic_bboxes');start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)and n.target.id=='output');wrapper=ast.parse('def _tail(source,candidates,median_height):\n pass').body[0];wrapper.body=copy.deepcopy(fn.body[start:]);ns=dict(old.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(path),'exec'),ns);tail=ns['_tail']
    folder=ROOT/'build/migration/pdf-concentric-graphics';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'concentric.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for enough in [True,False]:
            canvas.setPageRotation(rotation);canvas.setLineWidth(1);canvas.circle(130,450,50,stroke=1,fill=0);canvas.circle(130,450,30,stroke=1,fill=0)
            p=canvas.beginPath();p.moveTo(175,460);p.lineTo(190,470);p.lineTo(200,470);canvas.drawPath(p,stroke=1,fill=0)
            canvas.setFont('Helvetica',10);canvas.drawString(80,530,'Figure 1. Rings');canvas.drawString(90,450,'25%')
            if enough:canvas.drawString(145,425,'75 %')
            canvas.showPage();labels.append(f'{rotation}-{enough}')
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);s=SimpleNamespace(page_size=snap.page_size,path_infos=list(snap.path_infos),lines=lines,drawing_lines=snap.drawing_lines,form_bboxes=[])
            candidates=old._detect_captioned_concentric_path_graphics(s,10.);rejected=[]
            for j,b in enumerate(sorted(candidates,key=ns['_bbox_area'],reverse=True)):
                proof=grids(s,10.,b)
                if proof:rejected.append(dict(sorted_candidate_index=j,grid_bboxes=proof))
            source=clean(dict(page_size=s.page_size,path_infos=[asdict(p)for p in s.path_infos],lines=[dict(text=l.text,bbox=l.bbox,angle=l.angle,source_index=l.source_index,visual_row_id=l.visual_row_id)for l in lines],drawing_lines=[asdict(r)for r in snap.drawing_lines],form_bboxes=[]))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,source=source,expected=clean(dict(concentric_bboxes=candidates)),arbitration=clean(dict(graphic_bboxes=tail(s,candidates,10.),grid_rejections=rejected))))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual PDF concentric/arbitration chains;',sum(bool(c['expected']['concentric_bboxes'])for c in cases),'positive')
if __name__=='__main__':main()
