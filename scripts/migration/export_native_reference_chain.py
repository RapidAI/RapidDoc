"""Original raw PDF source -> complete document text -> reference/compact pass."""
import copy,json
from dataclasses import asdict
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_document_auxiliary_chain import reference as auxiliary
from export_native_document_titles import reference as titles
from export_native_references import reference as references
from export_native_late_image_text import project
from export_native_inline_clusters import clean

def reference(q):
    out=auxiliary(q);out.update(titles(out));out.update(references(out));return out

def main():
    target=ROOT/'build/migration/native-reference-chain.json';pdf=ROOT/'build/migration/native-reference-runtime.pdf';runtime=ROOT/'build/migration/native-reference-runtime.json'
    if any(p.exists()for p in [target,pdf,runtime]):raise FileExistsError('immutable reference chain exists')
    from loguru import logger
    logger.remove();cases=[]
    for c in json.loads((ROOT/'build/migration/native-document-text-chain.json').read_text(encoding='utf-8'))['cases']:
        q=copy.deepcopy(c['request']);out=dict(label=c['label'],request=q)
        try:out['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:out.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(out)
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    buf=BytesIO();c=Canvas(buf,pagesize=(400,600));c.setFont('Helvetica',11)
    for i in range(5):c.drawString(30,530-i*22,'Original body paragraph with complete native words')
    c.setFont('Helvetica-Bold',13);c.drawString(30,390,'References');c.setFont('Helvetica',9)
    for i in range(4):
        c.drawString(30,355-i*40,f'[{i+1}] Original author, Native Journal, 202{i}.')
        c.drawString(42,342-i*40,'Original hanging continuation with source details.')
    c.showPage();c.setFont('Helvetica',9)
    for i in range(4):
        c.drawString(30,530-i*40,f'[{i+5}] Original author, Native Journal, 202{i}.')
        c.drawString(42,517-i*40,'Original continuation across the reference page.')
    c.showPage();c.setFont('Helvetica-Bold',13);c.drawString(30,500,'Appendix A');c.setFont('Helvetica',11)
    for i in range(5):c.drawString(30,460-i*25,'Original appendix body words remain ordinary prose.')
    c.save();data=buf.getvalue();pages=[]
    with PDFDocument(data)as doc:
        for index in range(3):
            snap=doc._extract_native_page(index);g=doc.get_page_chars_with_geometry(index);ls=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            source=clean(dict(page_size=snap.page_size,page_index=index,lines=[project(l)for l in ls],chars=g.chars,drawing_lines=[asdict(r)for r in snap.drawing_lines],path_infos=[asdict(p)for p in snap.path_infos],form_infos=[asdict(f)for f in snap.form_infos],form_bboxes=snap.form_bboxes,image_bboxes=[info.bbox for info in snap.image_infos],signature_bboxes=[]))
            geometry=clean(dict(chars=g.chars,tight_bboxes=[dict(char_idx=i,bbox=b)for i,b in sorted(g.tight_bboxes.items())],origins=[dict(char_idx=i,origin=o)for i,o in sorted(g.origins.items())],loose_bboxes=[dict(char_idx=i,bbox=b)for i,b in sorted(g.loose_bboxes.items())]));pages.append(dict(source=source,geometry=geometry))
    q=dict(pages=pages);expected=reference(q)
    for p in expected['pages'][:2]:
        assert p['numbered_references'];assert p['reference_regions'];assert any(l['paragraph_group']is not None and l['title_suppressed']for l in p['remaining_lines'])
    assert not expected['pages'][2].get('reference_regions');assert not any(l['paragraph_group']is not None for l in expected['pages'][2]['remaining_lines'])
    cases.append(dict(label='live-reference-document',request=q,expected=expected));pdf.write_bytes(data);write_json(runtime,dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),request=q,expected=expected));write_json(target,dict(cases=cases));print(len(cases),'raw-source reference chains; 3 actual PDFium pages, 2 numbered regions/pages and appendix barrier')
if __name__=='__main__':main()
