"""Freeze actual drawing-line components and positive PDF component graphics."""
import io,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-drawing-graphics.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _summarize_drawing_components as summarize,_detect_complex_drawing_components as detect,_is_two_dimensional_complex_path as complex_path
    folder=ROOT/'build/migration/pdf-drawing-graphics';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'components.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for complex in [True,False]:
            canvas.setPageRotation(rotation);canvas.setLineWidth(1);canvas.rect(50,350,180,180,stroke=1,fill=0)
            if complex:
                p=canvas.beginPath();p.moveTo(65,380);p.lineTo(85,400);p.lineTo(110,390);p.lineTo(135,430);p.lineTo(170,410);p.lineTo(190,470);p.lineTo(210,490);canvas.drawPath(p,stroke=1,fill=0)
            canvas.showPage();labels.append(f'{rotation}-{complex}')
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            s=doc._extract_native_page(i);lines=list(s.drawing_lines);paths=list(s.path_infos);positions={id(l):i for i,l in enumerate(lines)}
            expected=clean(dict(tolerance=7.5,complex_path_indices=[j for j,p in enumerate(paths)if complex_path(p,10.)],components=[dict(line_indices=[positions[id(l)]for l in c.lines],bbox=c.bbox,horizontal_count=c.horizontal_count,vertical_count=c.vertical_count)for c in summarize(lines,7.5)],graphic_bboxes=detect(lines,paths,s.page_size,10.)))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,page_size=clean(s.page_size),drawing_lines=clean([asdict(l)for l in lines]),path_infos=clean([asdict(p)for p in paths]),expected=expected))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'actual PDF component chains;',sum(bool(c['expected']['graphic_bboxes'])for c in cases),'positive')
if __name__=='__main__':main()
