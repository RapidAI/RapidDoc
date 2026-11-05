"""Freeze complete old real PDF source/image/noise/chart-description grouping chains."""
import io,json,sys
from dataclasses import fields,asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-raster-charts.json'
    if target.exists():raise FileExistsError(target)
    from PIL import Image
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_text import restore_drop_cap_paragraphs,group_hanging_letter_paragraphs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_noise import exclude_faint_native_noise
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _raster_axis_tick_groups as groups,_detect_native_raster_axis_graphics as detect,group_native_raster_chart_descriptions as group
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _detect_repeated_raster_watermark_fingerprints as watermarks,_filter_repeated_raster_watermark_bboxes as filter_images,_exclude_proven_blank_top_under_display_title as trim
    import re
    folder=ROOT/'build/migration/pdf-raster-charts';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'charts.pdf'
    existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    canvas=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0)
    specs=[(v,r,'plain')for v in [True,False]for r in [0,90,180,270]]+[(v,0,style)for v in [True,False]for style in ['font','wide','bullet','terminal']]
    for i,(vertical,rotation,style)in enumerate(specs):
        canvas.setPageRotation(rotation)
        im=Image.new('RGB',(50,50));im.putdata([((x*73+y*119+i*7)%256,(x*31+y*17+i)%256,(x*13+y*41+i*3)%256)for y in range(50)for x in range(50)])
        canvas.drawImage(ImageReader(im),100,200,200,200)
        canvas.setFont('Courier',10)
        for j in range(5):
            if vertical:canvas.drawRightString(80,395-j*47.5,str(j*10))
            else:canvas.drawCentredString(100+j*50,180,str(j*10))
        canvas.setFont('Helvetica',10)
        canvas.drawString(68,444,'This description continues above the chart'+('.'if style=='terminal'else''))
        canvas.setFont('Helvetica-Bold'if style=='font'else'Helvetica',10)
        canvas.drawString(68,430,'1) short continuation'if style=='bullet'else'A continuation with many more words than the first line'if style=='wide'else'short continuation')
        canvas.showPage()
    canvas.save();data=pdf.read_bytes();names=[f.name for f in fields(_LineItem)if not f.name.startswith('_')]
    pages=[];sources=[];infos=[];sizes=[]
    with PDFDocument(data)as doc:
        for i,spec in enumerate(specs):
            geometry=doc.get_page_chars_with_geometry(i);size=doc.page_size(i);rotation=doc.page_rotation(i)
            lines=_build_native_line_items_from_chars(geometry.chars,size,page_rotation=rotation);restore_drop_cap_paragraphs(lines);group_hanging_letter_paragraphs(lines)
            source=SimpleNamespace(lines=lines,chars=list(geometry.chars),image_bboxes=[],page_size=size)
            sources.append(source);infos.append(doc.get_page_image_infos(i));sizes.append(size)
            pages.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=str(spec),prepared_lines=clean([{n:getattr(l,n)for n in names}for l in lines]),image_infos=clean([asdict(x)for x in infos[-1]])))
        wm=watermarks(infos,sizes)
        for i,(p,s,img)in enumerate(zip(pages,sources,infos,strict=True)):
            if len(s.lines)>=3 and sum(len(l.text.strip())for l in s.lines)>=60:img=[x for x in img if not x.smooth_background]
            s.image_bboxes=filter_images(img,s.page_size,wm);s.image_bboxes=trim(s,img)
            exclude_faint_native_noise(s,lambda indices:doc._get_page_text_paint(i,indices),())
            before=dict(lines=[{n:clean(getattr(l,n))for n in names}for l in s.lines],chars=clean(s.chars),image_bboxes=s.image_bboxes,page_index=i)
            positions={id(l):j for j,l in enumerate(s.lines)}
            numeric=[l for l in s.lines if l.angle==0 and l.semantic_type is None and l.effective_height>0 and re.fullmatch(r'[+−-]?\d+(?:\.\d+)?',l.text.strip())and len(l.text.strip())<=8 and l.bbox[2]-l.bbox[0]<=4*l.effective_height]
            axis=[dict(vertical=v,line_indices=[positions[id(l)]for l in ls])for v,ls,_ in groups(numeric)];cores=detect(s);group(s)
            p['source']=clean(before);after=dict(before,lines=[{n:clean(getattr(l,n))for n in names}for l in s.lines]);p['expected']=clean(dict(source=after,graphic_bboxes=cores,axis_groups=axis))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/text_noise.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},watermark_fingerprints=sorted(wm),cases=pages))
    print('Frozen',len(pages),'actual raster PDF chains;',sum(bool(p['expected']['graphic_bboxes'])for p in pages),'core positive;',sum(p['source']['lines']!=p['expected']['source']['lines']for p in pages),'group positive')
if __name__=='__main__':main()
