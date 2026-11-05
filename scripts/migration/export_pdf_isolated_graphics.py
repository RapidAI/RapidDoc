"""Freeze real PDF vector/raster/text inputs and old isolated graphic decisions."""
import io,json,math,sys,statistics
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-isolated-graphics.json'
    if target.exists():raise FileExistsError(target)
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.isolated_graphics import _isolated_path_groups as group,isolated_vector_components as vectors,meaningful_small_raster as small
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_layout import _line_effective_height
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_area
    from types import SimpleNamespace
    folder=ROOT/'build/migration/pdf-isolated-graphics';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'isolated.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for cover in [False,True]:
            canvas.setPageRotation(rotation);canvas.setFillColorRGB(.2,.4,.7);p=canvas.beginPath()
            for j in range(40):
                a=j*math.tau/40;x=100+20*math.cos(a);y=450+20*math.sin(a)
                if j==0:p.moveTo(x,y)
                else:p.lineTo(x,y)
            p.close();canvas.drawPath(p,stroke=0,fill=1)
            canvas.drawImage(ImageReader(Image.new('RGB',(20,20),(100,200,150))),250,350,45,45)
            canvas.setFillColorRGB(0,0,0);canvas.setFont('Helvetica',10);canvas.drawString(20,550,'Body reference line')
            if cover:canvas.drawString(82,446,'covered vector');canvas.drawString(250,370,'covered raster')
            canvas.showPage();labels.append(f'{rotation}-{cover}')
    canvas.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            s=SimpleNamespace(page_size=snap.page_size,path_infos=list(snap.path_infos),image_bboxes=[p.bbox for p in snap.image_infos],lines=rows)
            heights=[_line_effective_height(l,l.bbox)for l in rows if l.angle==0];em=statistics.median(heights)if heights else 10.;area=s.page_size[0]*s.page_size[1]
            eligible=[p for p in s.path_infos if p.form_depth==0 and(p.fill_visible or p.stroke_visible)and not p.rectangle_bboxes and p.segment_count>=8 and 0<_bbox_area(p.bbox)<=.06*area];positions={id(p):j for j,p in enumerate(s.path_infos)}
            expected=clean(dict(em=em,path_groups=[[positions[id(p)]for p in g]for g in group(s,eligible,em)],vector_bboxes=vectors(s),meaningful_raster_indices=[j for j,b in enumerate(s.image_bboxes)if small(s,b)]))
            source=clean(dict(page_size=s.page_size,path_infos=[asdict(p)for p in s.path_infos],image_bboxes=s.image_bboxes,lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,angle=l.angle,effective_height=l.effective_height,em_height=l.em_height,style_scale_repaired=l.style_scale_repaired)for l in rows]))
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,source=source,expected=expected))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/isolated_graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual isolated PDF chains;',sum(bool(c['expected']['vector_bboxes'])for c in cases),'vector-positive;',sum(bool(c['expected']['meaningful_raster_indices'])for c in cases),'small-raster-positive')
if __name__=='__main__':main()
