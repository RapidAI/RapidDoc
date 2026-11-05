"""Freeze old script sidecars and generated actual PDF superscript/subscript chains."""
import copy,json
from types import SimpleNamespace
from dataclasses import asdict,fields
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-script-payloads.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import detect_pdf_text_script_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.materialize import apply_pdf_inline_evidence
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.local_text import restore_drop_cap_paragraphs,group_hanging_letter_paragraphs
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from loguru import logger
    logger.remove();cases=[];excluded=[];real=[]
    old=json.loads((ROOT/'build/migration/script-refinement.json').read_text(encoding='utf-8'))
    for i,c in enumerate(old['lines']):
        q=c['request']
        if q['fraction_members']:excluded.append(dict(index=i,reason='preclassified fraction members belong to the independent existing 560-line refinement reference; this new detector computes members from raw primitives'));continue
        l=copy.deepcopy(q['line']);raw=l['chars'];l['bbox']=[min(ch['bbox'][0]for ch in raw),min(ch['bbox'][1]for ch in raw),max(ch['bbox'][2]for ch in raw),max(ch['bbox'][3]for ch in raw)];l['text']=''.join(ch['char']for ch in raw);l['source_index']=i
        request=dict(lines=[l],page_size=q['page_size'],tight_bboxes=q['tight_bboxes'],origins=q['origins'],all_chars=[],drawing_lines=[])
        payload=detect_pdf_text_script_lines([SimpleNamespace(**copy.deepcopy(l))],tuple(q['page_size']),{int(k):v for k,v in q['tight_bboxes'].items()},{int(k):v for k,v in q['origins'].items()},all_chars=[],drawing_lines=[])
        cases.append(dict(label=c['label'],request=request,expected=clean([asdict(p)for p in payload])))
    folder=ROOT/'build/migration/native-script-payloads';folder.mkdir(parents=True,exist_ok=True);pdf=folder/'scripts.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase.pdfmetrics import stringWidth
    cn=canvas.Canvas(str(pdf),pagesize=(260.,190.))
    for angle in [0,90,180,270]:
        cn.setPageRotation(angle);cn.setFont('Helvetica',12);cn.drawString(20.,140.,'Energy mc');x=20.+stringWidth('Energy mc','Helvetica',12);cn.setFont('Helvetica',7);cn.drawString(x,145.,'2');cn.setFont('Helvetica',12);cn.drawString(x+5.,140.,' is constant')
        cn.drawString(20.,100.,'Water H');x=20.+stringWidth('Water H','Helvetica',12);cn.setFont('Helvetica',7);cn.drawString(x,96.,'2');cn.setFont('Helvetica',12);cn.drawString(x+5.,100.,'O is liquid');cn.showPage()
    cn.save();data=pdf.read_bytes();names=[f.name for f in fields(_LineItem)if not f.name.startswith('_')]
    with PDFDocument(data)as doc:
        for page_index in range(4):
            g=doc.get_page_chars_with_geometry(page_index)
            with doc._open_page(page_index)as page:size=tuple(page.get_size());rotation=page.get_rotation()
            lines=_build_native_line_items_from_chars(g.chars,size,page_rotation=rotation,supported_angles=(0.,90.,180.,270.));restore_drop_cap_paragraphs(lines);group_hanging_letter_paragraphs(lines)
            payload=detect_pdf_text_script_lines(lines,size,g.tight_bboxes,g.origins,all_chars=g.chars,drawing_lines=[])
            blocks=[dict(type='text',content=l.text,bbox=l.bbox)for l in lines];diagnostics=[];apply_pdf_inline_evidence(blocks,[],[],payload,size,materialized_diagnostics=diagnostics)
            real.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=page_index,page_size=size,rotation=rotation,expected=clean([asdict(p)for p in payload]),output_expected=clean(dict(blocks=blocks,materialized_ranges=diagnostics))))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/scripts.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/common.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,excluded_preclassified_fraction_inputs=excluded,real=clean(real)))
    print('Frozen',len(cases),'payload cases;',len(excluded),'preclassified fraction references retained independently;',len(real),'actual PDF angles;',sum(bool(l['script_ranges'])for c in real for l in c['expected']),'script-positive lines;',sum(len(c['output_expected']['materialized_ranges'])for c in real),'materialized ranges')
if __name__=='__main__':main()
