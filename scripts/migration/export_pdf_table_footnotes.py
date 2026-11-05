"""Live PDF geometry to automatic note collection; no note verdict or selected-row injection."""
import io,json,sys
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_rule_annotation_preparation import row,fragment
from export_native_table_footnotes import reference
FIELDS='text bbox source_index angle effective_height em_height style_scale_repaired font_signature font_coverage chars'.split()
def main():
    close_aux='--close-aux'in sys.argv
    target=ROOT/('build/migration/pdf-table-footnotes-closeaux.json'if close_aux else'build/migration/pdf-table-footnotes.json')
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    folder=ROOT/'build/migration/pdf-table-footnotes';folder.mkdir(parents=True,exist_ok=True);pdf=folder/('closeaux.pdf'if close_aux else'notes.pdf');existing='--existing-pdf'in sys.argv
    if pdf.exists()and not existing:raise FileExistsError(pdf)
    if existing and not pdf.is_file():raise FileNotFoundError(pdf)
    c=Canvas(io.BytesIO()if existing else str(pdf),pagesize=(400,600),pageCompression=0);labels=[]
    for kind in ['explicit','aux-reference','aux-missing','superscript']:
        c.setFillColorRGB(.9,.9,.95);c.rect(40,200,200,100,stroke=0,fill=1);c.setFillColorRGB(0,0,0);c.setFont('Helvetica',10)
        if kind=='superscript':c.drawString(50,250,'Long reference cell content');c.setFont('Helvetica',6);c.drawString(180,255,'a')
        else:c.drawString(50,250,'12 a'if kind=='aux-reference'else'12')
        c.setFont('Helvetica',8);c.drawString(40,189 if close_aux else 185,'Note: measured values'if kind=='explicit'else'a) measured values');c.drawString(40,178 if close_aux else 174,'continued explanatory text');c.showPage();labels.append(kind)
    c.save();prior=json.loads((ROOT/'build/migration/pdf-rule-annotation-preparation.json').read_text(encoding='utf-8'));inputs=[(pdf,i,label)for i,label in enumerate(labels)]+([]if close_aux else[(ROOT/c['pdf'],c['page_index'],'prior-'+c['label'])for c in prior['cases']]);cases=[]
    for path,i,label in inputs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(i);g=doc.get_page_chars_with_geometry(i);lines=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation);frame=snap.path_infos[0].bbox;rows=[row([fragment(l.source_index,l.bbox,text=l.text)],visual=l.visual_row_id)for l in lines];core=[l.source_index for l in lines if frame[0]<=(l.bbox[0]+l.bbox[2])/2<=frame[2]and frame[1]<=(l.bbox[1]+l.bbox[3])/2<=frame[3]];q=clean(dict(rows=rows,lines=[{k:getattr(l,k)for k in FIELDS}for l in lines],rule_bbox=frame,median_height=10.,core_line_indices=core,page_size=snap.page_size,angle=0));cases.append(dict(pdf=str(path.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,request=q,expected=reference(q)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_constants.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Each actual visual line projected to one local=original fragment; first actual fill frame and line centres supply explicit rule/core query. Note classification, marker references and final selection are original full rules; no selected footnote-row input. 0/90/270 page rotations, angle query=0.',cases=cases));print('Frozen',len(cases),'live PDF automatic-note queries;',sum(bool(c['expected']['footnote_rows'])for c in cases),'positive')
if __name__=='__main__':main()
