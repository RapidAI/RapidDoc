"""Freeze real PDF drawing/character -> component candidates -> original grid arbitration."""
import ast,copy,io,json,sys
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-graphic-arbitration.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _native_multicolumn_grid_bboxes as grids
    # Execute the exact same original tail AST frozen by the independent synthetic exporter.
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_detect_strong_graphic_bboxes');start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)and n.target.id=='output');wrapper=ast.parse('def _tail(source,candidates,median_height):\n pass').body[0];wrapper.body=copy.deepcopy(fn.body[start:]);ns=dict(old.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(path),'exec'),ns);tail=ns['_tail']
    folder=ROOT/'build/migration/pdf-graphic-arbitration';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'arbitration.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for table in [True,False]:
            canvas.setPageRotation(rotation);canvas.setLineWidth(1)
            for y in [350,380,410,440]:canvas.line(20,y,220,y)
            for x in [20,120,220]:canvas.line(x,350,x,440)
            if table:
                canvas.setFont('Helvetica',10)
                for y in [360,390,420]:
                    for x in [30,140]:canvas.drawString(x,y,'cell')
            p=canvas.beginPath();p.moveTo(40,360);p.lineTo(70,370);p.lineTo(80,390);p.lineTo(100,380);p.lineTo(140,400);p.lineTo(170,430);p.lineTo(190,420);canvas.drawPath(p,stroke=1,fill=0)
            labels.append(f'{rotation}-{table}');canvas.showPage()
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);s=SimpleNamespace(page_size=snap.page_size,lines=rows,drawing_lines=snap.drawing_lines,form_bboxes=[])
            candidates=old._detect_complex_drawing_components(snap.drawing_lines,snap.path_infos,snap.page_size,10.);rejected=[]
            for j,b in enumerate(sorted(candidates,key=ns['_bbox_area'],reverse=True)):
                proof=grids(s,10.,b)
                if proof:rejected.append(dict(sorted_candidate_index=j,grid_bboxes=proof))
            source=clean(dict(page_size=snap.page_size,lines=[dict(text=l.text,bbox=l.bbox,angle=l.angle,source_index=l.source_index,visual_row_id=l.visual_row_id)for l in rows],drawing_lines=[asdict(r)for r in snap.drawing_lines],form_bboxes=[]))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,source=source,path_infos=clean([asdict(p)for p in snap.path_infos]),candidate_bboxes=clean(candidates),expected=clean(dict(graphic_bboxes=tail(s,candidates,10.),grid_rejections=rejected))))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual PDF candidate/arbitration chains;',sum(bool(c['candidate_bboxes'])for c in cases),'candidate-positive;',sum(bool(c['expected']['grid_rejections'])for c in cases),'table rejection-positive')
if __name__=='__main__':main()
