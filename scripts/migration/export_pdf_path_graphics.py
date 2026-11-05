"""Freeze actual PDFium complex-container/axis candidates with positive rotated PDFs."""
import io,json,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-path-graphics.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _is_two_dimensional_complex_path as complex_path,_detect_complex_path_containers as containers,_detect_axis_path_graphics as axes
    folder=ROOT/'build/migration/pdf-path-graphics';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'paths.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for container in [True,False]:
            canvas.setPageRotation(rotation);canvas.setLineWidth(1);canvas.setStrokeColorRGB(0,0,0)
            if container:
                canvas.setFillColorRGB(.9,.9,.9);canvas.rect(50,350,180,180,stroke=0,fill=1)
                for i in range(4):
                    x=65+i*35;p=canvas.beginPath();p.moveTo(x,365);p.lineTo(x+5,380);p.lineTo(x+25,390);p.lineTo(x+20,410);p.lineTo(x+10,420);p.lineTo(x,415);p.close();canvas.drawPath(p,stroke=1,fill=0)
            else:
                canvas.line(50,350,230,350);canvas.line(50,350,50,530)
                p=canvas.beginPath();p.moveTo(65,380);p.lineTo(85,400);p.lineTo(110,390);p.lineTo(135,430);p.lineTo(170,410);p.lineTo(190,470);p.lineTo(210,490);canvas.drawPath(p,stroke=1,fill=0)
            labels.append(f'{rotation}-'+('container'if container else'axis'));canvas.showPage()
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snapshot=doc._extract_native_page(i);paths=list(snapshot.path_infos);size=snapshot.page_size
            expected=clean(dict(complex_path_indices=[i for i,p in enumerate(paths)if complex_path(p,10.)],container_bboxes=containers(paths,size,10.),axis_bboxes=axes(paths,size,10.)))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,page_size=clean(size),path_infos=clean([asdict(p)for p in paths]),expected=expected))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'actual PDF path chains;',sum(bool(c['expected']['container_bboxes'])for c in cases),'container positives;',sum(bool(c['expected']['axis_bboxes'])for c in cases),'axis positives')
if __name__=='__main__':main()
