"""Independent embedded-font metric PDF; retain unembedded-CID runtime mismatch evidence."""
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_pdf_table_footnotes import FIELDS
from export_pdf_numeric_candidates import EXTRA
from export_numeric_candidate_segments import reference as individual
from export_physical_table_page import reference
def main():
    target=ROOT/'build/migration/pdf-embedded-metric-candidate.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    font=ROOT/'rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf';pdf=ROOT/'build/migration/pdf-numeric-candidates/embedded-metric.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    pdfmetrics.registerFont(TTFont('EmbeddedMetric',str(font)));c=Canvas(str(pdf),pagesize=(400,600),pageCompression=0);c.setFont('EmbeddedMetric',15);c.drawString(50,505,'关键指标');c.line(110,510,350,510);c.setFont('EmbeddedMetric',10)
    for i in range(5):c.drawString(50,600-(108+18*i),'指标一');c.drawRightString(280,600-(108+18*i),'10%')
    c.save();data=pdf.read_bytes()
    with PDFDocument(data)as doc:
        snap=doc._extract_native_page(0);g=doc.get_page_chars_with_geometry(0);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
        source=clean(dict(page_size=snap.page_size,lines=[{k:getattr(l,k)for k in FIELDS+EXTRA}for l in lines],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_bboxes=[],image_bboxes=[],signature_bboxes=[],page_index=0));q=dict(source=source,excluded_bboxes=[]);result=individual(dict(q,kind='metric_pair'));assert result['candidates']
        case=dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=0,label='embedded-metric',kind='metric_pair',request=q,expected=dict(individual=result,page=reference(q,ordinary=True,supplemental=True,numeric=True)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py',str(font.relative_to(ROOT))];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=[case]));print('Frozen embedded metric PDF positive; unembedded CID first-char top mismatch retained separately')
if __name__=='__main__':main()
