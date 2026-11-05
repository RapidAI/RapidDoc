"""Live PDF lines/rules to explicit frame-and-source-ID queries; no candidate detector claimed."""
import io,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_closed_grids import reference
def main():
    target=ROOT/'build/migration/pdf-closed-grids.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-closed-grids';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'closed.pdf';existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for rotation in [0,90,180,270]:
        for kind in ['full','two-by-two','missing-top','one-column-text']:
            c.setPageRotation(rotation);c.setFillColorRGB(.95,.95,.95);c.rect(40,100,200,120,stroke=0,fill=1);c.setLineWidth(1)
            rows=2 if kind=='two-by-two' else 3
            for r in range(rows+1):
                if kind=='missing-top'and r==rows:continue
                y=100.+r*120./rows;c.line(40,y,240,y)
            for x in [40,140,240]:c.line(x,100,x,220)
            c.setFillColorRGB(0,0,0);c.setFont('Helvetica',10)
            for r in range(rows):
                for x in ([60]if kind=='one-column-text'else[60,160]):c.drawString(x,110.+r*120./rows,'cell')
            c.showPage();labels.append(f'{rotation}-{kind}')
    c.save();data=pdf.read_bytes();cases=[]
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);first=snap.path_infos[0];frame=first.rectangle_bboxes[0]if first.rectangle_bboxes else first.bbox
            source=clean(dict(form_bboxes=[],drawing_lines=[asdict(r)for r in snap.drawing_lines],lines=[dict(bbox=l.bbox,source_index=l.source_index,effective_height=l.effective_height)for l in rows]));candidates=clean([dict(bbox=frame,core_bbox=None,angle=0,line_indices=[l.source_index for l in rows])])
            for allow in [False,True]:
                q=dict(source=source,candidates=candidates,allow_two_by_two=allow);cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label+'-'+str(allow),request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},candidate_scope='Explicit frame from first actual path rectangle/bbox; member IDs from live visual lines; not full table candidate production. Form input empty; true Form extraction not covered.',cases=cases));print('Frozen',len(cases),'PDF query chains across',len(labels),'pages;',sum(any(c['expected']['closed_grid_flags'])for c in cases),'positive')
if __name__=='__main__':main()
