"""Freeze real PDF chars/images/vectors -> local source lines -> raster decisions."""
import json
from dataclasses import fields,asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-image-sources.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument,PDFDrawingLine
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars,_coerce_pdf_drawing_lines,_extract_decorative_text_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_text import restore_drop_cap_paragraphs,group_hanging_letter_paragraphs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _detect_repeated_raster_watermark_fingerprints as detect,_filter_repeated_raster_watermark_bboxes as filter_watermark,_exclude_proven_blank_top_under_display_title as trim
    f=json.loads((ROOT/'build/migration/pdf-image-infos.json').read_text(encoding='utf-8'))
    chosen={'L-smooth','L-noise','L-blank-8','alpha-blank','repeated-0','repeated-1','repeated-2','L-blank-top-clipped','L-blank-form','L-blank-rotation-90'}
    names=[x.name for x in fields(_LineItem)if not x.name.startswith('_')]
    pages=[];sources=[];image_infos=[];sizes=[]
    for c in f['cases']:
        if c.get('label')not in chosen:continue
        data=(ROOT/c['pdf']).read_bytes()
        with PDFDocument(data)as doc:
            size=doc.page_size(c['page_index']);rotation=doc.page_rotation(c['page_index'])
            g=doc.get_page_chars_with_geometry(c['page_index'])
            rows=_build_native_line_items_from_chars(g.chars,size,page_rotation=rotation)
            restore_drop_cap_paragraphs(rows);group_hanging_letter_paragraphs(rows)
            infos=doc.get_page_image_infos(c['page_index'])
            rules=_coerce_pdf_drawing_lines(doc.get_page_drawing_lines(c['page_index']))
            heights=sorted(l.em_height for l in rows if l.em_height>0)
            if heights:
                em=heights[len(heights)//2]
                rules.extend(_coerce_pdf_drawing_lines([PDFDrawingLine((i.bbox[0],(i.bbox[1]+i.bbox[3])/2),(i.bbox[2],(i.bbox[1]+i.bbox[3])/2),i.bbox,i.bbox[3]-i.bbox[1],'horizontal')for i in infos if i.bbox[2]-i.bbox[0]>=2*em and 0<i.bbox[3]-i.bbox[1]<=.1*em]))
            rows,decor=_extract_decorative_text_rules(rows,size);rules.extend(decor)
            source=SimpleNamespace(lines=rows,page_size=size,image_bboxes=[]);sources.append(source);image_infos.append(infos);sizes.append(size)
            pages.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],expected=clean(dict(lines=[{n:getattr(l,n)for n in names}for l in rows],drawing_lines=[asdict(r)for r in rules]))))
    watermarks=detect(image_infos,sizes);boxes=[]
    for source,infos in zip(sources,image_infos,strict=True):
        if len(source.lines)>=3 and sum(len(l.text.strip())for l in source.lines)>=60:infos=[i for i in infos if not i.smooth_background]
        source.image_bboxes=filter_watermark(infos,source.page_size,watermarks);boxes.append(trim(source,infos))
    write_json(target,dict(pages=pages,expected=clean(dict(watermark_fingerprints=sorted(watermarks),image_bboxes=boxes))))
    print('Frozen',len(pages),'fully read PDF source/image chains')
if __name__=='__main__':main()
