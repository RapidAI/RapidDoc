"""PDF live geometry to explicit rule/core/caption/footnote preparation queries."""
import io,sys
from common import ROOT,digest,write_json
from export_native_rule_annotation_preparation import row,fragment,reference
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/pdf-rule-annotation-preparation.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-rule-annotation-preparation';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'annotations.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,270]:
        for kind in ['caption','footnote','both','neither']:
            c.setPageRotation(rotation);c.setFillColorRGB(.95,.95,.95);c.rect(40,100,200,120,stroke=0,fill=1);c.setLineWidth(1)
            for y in [100,140,180,220]:c.line(40,y,240,y)
            for x in [40,140,240]:c.line(x,100,x,220)
            c.setFillColorRGB(0,0,0);c.setFont('Helvetica',10)
            for y in [110,150,190]:
                for x in [60,160]:c.drawString(x,y,'cell')
            if kind in ['caption','both']:c.drawString(40,240,'Table 1: Results');c.drawString(40,225,'Second title line')
            if kind in ['footnote','both']:c.drawString(40,80,'Note: measured values')
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);frame=snap.path_infos[0].bbox
            rows=[row([fragment(l.source_index,l.bbox,text=l.text)],visual=l.visual_row_id)for l in lines];core=[j for j,r in enumerate(rows)if frame[0]<=(r['bbox'][0]+r['bbox'][2])/2<=frame[2]and frame[1]<=r['center_y']<=frame[3]];caption=next((l.source_index for l in lines if l.text.startswith('Table')),None);foot=[j for j,l in enumerate(lines)if l.text.startswith('Note')];q=clean(dict(rows=rows,rule_bboxes=[r.bbox for r in snap.drawing_lines],core_row_indices=core,median_height=10.,caption_source_index=caption,footnote_row_indices=foot));cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Each live visual line projected as a single-fragment row; local bbox equals original for these explicit queries. Core rows from actual first path frame; title/note seed via fixture text, not automatic title/note detection or table row clustering. 180deg not in this corpus.',cases=cases));print('Frozen',len(cases),'live PDF annotation queries;',sum(bool(c['expected']['caption_rows'])for c in cases),'caption positives;',sum(bool(c['expected']['annotations'])for c in cases),'annotation positives')
if __name__=='__main__':main()
