"""Real text/rules to explicit draft/grid queries; no generated draft verdict injection."""
import io,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_prose_rule_conflicts import reference
def main():
    target=ROOT/'build/migration/pdf-prose-rule-conflicts.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-prose-rule-conflicts';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'prose.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in ['prose','bold','two-rows','missing-vertical','caption-draft']:
            c.setPageRotation(rotation);c.setFillColorRGB(.9,.9,.95);c.rect(40,100,200,120,stroke=0,fill=1);c.setLineWidth(1)
            c.line(40,350,240,350)
            for x in ([40,240]if kind=='missing-vertical'else[40,140,240]):c.line(x,100,x,220)
            for y in [100,140,180,220]:c.line(40,y,240,y)
            c.setFillColorRGB(0,0,0);c.setFont('Helvetica-Bold'if kind=='bold'else'Helvetica',8)
            for i in range(2 if kind=='two-rows' else 3):c.drawString(40,300-i*15,'These ordinary words form a separate prose sentence.')
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);frame=snap.path_infos[0].bbox
            source=clean(dict(lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,angle=l.angle,caption_start=l.caption_start,formula_candidate_only=l.formula_candidate_only,semantic_type=l.semantic_type,dominant_font_weight=l.dominant_font_weight)for l in rows],drawing_lines=[asdict(r)for r in snap.drawing_lines]));boundaries=clean([r.bbox for r in snap.drawing_lines if r.orientation=='horizontal']);candidates=clean([dict(kind='draft',angle=0,median_height=10.,caption={}if label.endswith('caption-draft')else None,boundaries=boundaries,member_source_indices=[l.source_index for l in rows]),dict(kind='table',angle=0,bbox=frame,core_bbox=None)]);q=dict(source=source,candidates=candidates);cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Explicit draft/grid projection reconstructed from live first path frame, horizontal boundaries and visual-line IDs; median_height=10 and caption presence are query inputs, not automatic draft detection.',cases=cases));print('Frozen',len(cases),'live PDF prose queries;',sum(bool(c['expected']['rejected_indices'])for c in cases),'rejections')
if __name__=='__main__':main()
