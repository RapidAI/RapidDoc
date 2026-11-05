"""Freeze actual raster-connected diagram/table and plain-frame PDF evidence."""
import io,json,sys
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    grid_positive='--grid-positive'in sys.argv
    protected='--protected-spans'in sys.argv
    if protected and not grid_positive:raise ValueError('--protected-spans requires --grid-positive')
    target=ROOT/('build/migration/pdf-raster-diagrams-protected.json'if protected else'build/migration/pdf-raster-diagrams-grid.json'if grid_positive else'build/migration/pdf-raster-diagrams.json')
    if target.exists():raise FileExistsError(target)
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_raster_connected_diagrams as detect,_is_plain_text_frame as plain,_drawing_component_summaries as summaries
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _native_multicolumn_grid_bboxes as grids
    folder=ROOT/'build/migration/pdf-raster-diagrams';folder.mkdir(parents=True,exist_ok=True);pdf=folder/('protected.pdf'if protected else'grids.pdf'if grid_positive else'diagrams.pdf');existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in (['table']if grid_positive else['diagram','table','text-frame','empty-frame']):
            canvas.setPageRotation(rotation);canvas.setLineWidth(1)
            if kind in ['diagram','table']:
                for y in [350,400,450,500]:canvas.line(20,y,220,y)
                for x in [20,70,120,170,220]:canvas.line(x,350,x,500)
                for y in [380,430,480]:canvas.line(50 if protected else 40,y,190 if protected else 200,y)
                for j,(x,y)in enumerate([(30,365),(140,365),(30,465),(140,465)]):canvas.drawImage(ImageReader(Image.new('RGB',(20,15),(j*40,120,180))),x,y,20,15)
                if kind=='table':
                    canvas.setFont('Helvetica',10)
                    for x in ([80,140]if grid_positive else[30,140]):
                        for y in [360,410,460]:canvas.drawString(x,y,'cell')
            else:
                canvas.rect(20,350,200,150,stroke=1,fill=0)
                if kind=='text-frame':canvas.setFont('Helvetica',10);canvas.drawString(40,420,'ordinary text')
            canvas.showPage();labels.append(f'{rotation}-{kind}')
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);s=SimpleNamespace(page_size=snap.page_size,lines=rows,drawing_lines=snap.drawing_lines,path_infos=snap.path_infos,image_bboxes=[p.bbox for p in snap.image_infos],form_bboxes=[],drawing_component_cache=[])
            frames=[c.bbox for c in summaries(s,7.5)];expected=clean(dict(raster_diagram_bboxes=detect(s,10.),plain_text_frame_flags=[plain(s,b,10.)for b in frames]));source=clean(dict(page_size=s.page_size,lines=[dict(text=l.text,bbox=l.bbox,angle=l.angle,source_index=l.source_index,visual_row_id=l.visual_row_id)for l in rows],drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],image_bboxes=s.image_bboxes,form_bboxes=[]))
            case=dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,source=source,frame_bboxes=clean(frames),expected=expected)
            if grid_positive:case['grid_bboxes']=clean([grids(s,10.,b)for b in frames])
            cases.append(case)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'real PDF chains;',sum(bool(c['expected']['raster_diagram_bboxes'])for c in cases),'diagram-positive;',sum(any(c['expected']['plain_text_frame_flags'])for c in cases),'plain-frame-positive')
if __name__=='__main__':main()
