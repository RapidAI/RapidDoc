"""冻结实际旧DocVortex PDF绘图/路径读取与合并；不覆写参考或固定PDF。"""
import json,math,random,struct
from dataclasses import asdict
from io import BytesIO
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/pdf-vector-primitives.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from pypdf import PdfReader,PdfWriter
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.native_objects import _merge_collinear_drawing_lines
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFDrawingLine
    outdir=ROOT/'build/migration/pdf-vector-primitives';outdir.mkdir(parents=True,exist_ok=True);synthetic=outdir/'geometry.pdf'
    if synthetic.exists():raise FileExistsError(synthetic)
    buffer=BytesIO();c=Canvas(buffer,pagesize=(320.5,240.25),invariant=1)
    c.setFont('Helvetica',12);c.drawString(10,220,'Vector evidence')
    for i,width in enumerate([0.,.5,1.,2.0005]):c.setLineWidth(width);c.line(10,20+i*25,200,20+i*25)
    c.setFillColorRGB(.2,.5,.7);c.rect(30,130,70,2,fill=1,stroke=0);c.rect(130,130,70,40,fill=1,stroke=1)
    p=c.beginPath();p.moveTo(20,180);p.curveTo(50,200,70,160,100,185);c.drawPath(p)
    c.showPage();c.beginForm('inner',0,0,140,100);c.setLineWidth(.7);c.rect(2,2,110,60,fill=0,stroke=1);c.line(5,25,120,25);c.endForm()
    c.beginForm('outer',0,0,200,140);c.saveState();c.translate(15,15);c.scale(1.7,.6);c.rotate(1);c.doForm('inner');c.restoreState();c.endForm();c.translate(20,30);c.doForm('outer');c.translate(140,90);c.scale(.6,1.2);c.doForm('inner');c.showPage()
    c.saveState();clip=c.beginPath();clip.rect(35,45,130,90);c.clipPath(clip,stroke=0,fill=0);c.translate(20,20);c.doForm('outer');c.restoreState();c.setStrokeAlpha(0.);c.line(10,200,290,200);c.setFillAlpha(0.);c.rect(190,40,60,60,fill=1,stroke=0);c.showPage()
    c.setLineWidth(.5);c.line(20,80,80,80);c.line(81.9,80.2,160,80.2);c.line(162.01,80,230,80);c.line(50,20,50,70);c.line(50.25,71.99,50.25,160);c.save()
    reader=PdfReader(BytesIO(buffer.getvalue()));writer=PdfWriter()
    for page in reader.pages:writer.add_page(page)
    for rotation in (0,90,180,270):
        page=writer.add_page(reader.pages[0]);page.cropbox.lower_left=(10.125,5.25);page.cropbox.upper_right=(300.375,230.125);page.rotate(rotation)
    with synthetic.open('wb')as stream:writer.write(stream)
    files=[synthetic.relative_to(ROOT).as_posix(),'demo/pdfs/示例1-论文模板.pdf','demo/pdfs/比亚迪财报.pdf','tests/migration/inputs/pdf_native.pdf','tests/migration/inputs/pdf_mixed.pdf','tests/migration/inputs/pdf_scan.pdf','build/migration/model-document/five-pages.pdf']
    cases=[]
    for name in files:
        path=ROOT/name
        with PDFDocument(path.read_bytes())as doc:
            for index in range(len(doc)):
                page=doc[index];g=page.get_vector_geometry()
                # get_bbox属于底层同库证据，仍在旧进程完成，不把它留给Rust结果反推。
                with doc._open_page(index)as native:frame=list(native.get_bbox())
                cases.append(dict(pdf=name,pdf_sha256=digest(path.read_bytes()),page_index=index,expected=dict(page_size=list(page.size),page_bbox=frame,rotation=page.rotation,drawing_lines=[asdict(x)for x in g.drawing_lines],path_infos=[asdict(x)for x in g.path_infos])))
    rng=random.Random(782393);numerics=[]
    pairs=[(0.,-0.),(3.,4.),(1e-310,2e-310),(1e308,1e308),(2**-1024,2**-1024)]+[(rng.uniform(-1e8,1e8),rng.uniform(-1e8,1e8))for _ in range(500)]
    for x,y in pairs:numerics.append(dict(x=x,y=y,hypot_bits=struct.pack('>d',math.hypot(x,y)).hex()))
    rounds=[dict(value=x,expected=round(x,3))for x in [.0005,.0015,1.0005,2.675,-.0005,-.0001,1e15,*[rng.uniform(-1000,1000)for _ in range(500)]]]
    merges=[]
    for _ in range(180):
        lines=[]
        for i in range(rng.randrange(1,18)):
            y=rng.choice([30.,31.,32.,32.00001]);x=rng.randrange(1,100);end=x+rng.randrange(1,60);width=rng.choice([.1,.5,1.]);lines.append(PDFDrawingLine((float(x),y),(float(end),y),(float(x),y-width/2,float(end),y+width/2),width,'horizontal'))
        merges.append(dict(lines=[asdict(x)for x in lines],page_size=[200.,200.],expected=[asdict(x)for x in _merge_collinear_drawing_lines(lines,(200.,200.))]))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/'+p for p in ['native_objects.py','native_coordinates.py','native_contracts.py','_object_bridge.py']];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,numerics=numerics,rounds=rounds,merges=merges));print(f'Frozen {len(cases)} real PDF pages, {sum(len(c["expected"]["drawing_lines"])for c in cases)} lines/{sum(len(c["expected"]["path_infos"])for c in cases)} paths; {len(numerics)} hypot/{len(rounds)} round/{len(merges)} merge facts')
if __name__=='__main__':main()
