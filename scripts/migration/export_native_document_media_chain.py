"""Original raw-source -> final page -> spacing/inline -> rendered nested cell media."""
import copy,json,sys
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_reference_chain import reference as prepare
from export_native_final_text_lines import reference as lines
from export_native_text_block_chain import page as blocks
from export_native_refined_block_chain import reference as refine
from export_native_final_page_output import reference as final
from export_native_document_inline import reference as inline,request as inline_request
from export_native_table_media_png import projection
def complete(q):
    e=prepare(q);out=[]
    for i,p in enumerate(e['pages']):
        request=dict(page=p,page_index=i,document_body_profile=e.get('body_profile'),canonical_body_profile=e.get('canonical_body_profile'),document_title_profile=e.get('title_profile'))
        b=blocks(request,lines(request));b.update(refine(request));b['model_list']=final(request);out.append(b)
    e['block_pages']=out;e['finalized_pages']=inline(inline_request([b['model_list']for b in out],e['pages'],q['pages']))['pages'];return e
def main():
    positive='--positive'in sys.argv;suffix='-positive'if positive else''
    if '--canonical'in sys.argv:suffix+='-canonical'
    if '--full-source'in sys.argv:suffix+='-source'
    target=ROOT/f'build/migration/native-document-media-chain{suffix}.json';pdf=ROOT/f'build/migration/native-document-media-chain{suffix}.pdf'
    if target.exists():raise FileExistsError('immutable document media references exist')
    from loguru import logger
    logger.remove()
    base=next(c for c in json.loads((ROOT/'build/migration/native-final-page-output-chain.json').read_text(encoding='utf-8'))['cases']if c['label']=='grid/positive')['request']
    if positive:
        source=next(c for c in json.loads((ROOT/'build/migration/native-table-block-commit.json').read_text(encoding='utf-8'))['cases']if c['events'].get('media'))['request']['source']
        base=dict(pages=[dict(source=copy.deepcopy(source),geometry=dict(chars=source['chars'],tight_bboxes=[],origins=[],loose_bboxes=[]))])
    if '--canonical'in sys.argv:
        from export_native_final_text_lines import typed_lines
        from export_native_late_image_text import project
        for p in base['pages']:p['source']['lines']=[project(l)for l in typed_lines(p['source']['lines'])]
    if '--full-source'in sys.argv:
        from export_native_table_page_policies import empty
        for p in base['pages']:p['source']=dict(empty(),**p['source'])
    from reportlab.pdfgen.canvas import Canvas
    width,height=base['pages'][0]['source']['page_size'];stream=BytesIO();canvas=Canvas(stream,pagesize=(width,height),pageCompression=0,invariant=1)
    for p in base['pages']:
        for l in p['source']['lines']:canvas.setFont('Helvetica',10);canvas.drawString(l['bbox'][0],height-l['bbox'][3],l['text'])
        canvas.setFillColorRGB(.15,.5,.8);canvas.rect(100,180,12,12,fill=1,stroke=0);canvas.showPage()
    canvas.save()
    if pdf.exists():assert pdf.read_bytes()==stream.getvalue(),'immutable PDF differs'
    else:pdf.write_bytes(stream.getvalue())
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_media import materialize_table_media
    cases=[]
    variants=[('cell',[[100.,408.,112.,420.]]),('two-cell-images',[[100.,408.,112.,420.],[115.,408.,127.,420.]]),('cross-cell',[[135.,408.,145.,420.]])]
    if positive:
        box=base['pages'][0]['source']['image_bboxes'][0];variants=[('original-vector-cell',[box]),('duplicate-vector-cell',[box,box.copy()]),('vector-cell-and-table',[box,[18.,38.,222.,162.]])]
    for mode,boxes in variants:
        q=copy.deepcopy(base);q.update(materialize_inline=True,finalize_output=True,refine_blocks=True);q['pages'][0]['source']['image_bboxes']=boxes
        expected=complete(q);pages=copy.deepcopy(expected['finalized_pages'])
        with PDFDocument(pdf.read_bytes())as doc:materialize_table_media(pages,doc)
        class Failed:
            def render_page(self,*args,**kw):raise RuntimeError('deliberate original render fallback')
        failed=copy.deepcopy(expected['finalized_pages']);materialize_table_media(failed,Failed())
        cases.append(dict(label=mode,request=q,expected=expected,media_expected=projection(pages),failure_expected=projection(failed)))
    write_json(target,dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=digest(pdf.read_bytes()),cases=cases));print(len(cases),'raw-source media chains;',sum(len(c['media_expected']['images'])for c in cases),'rendered images; original failure fallback frozen')
if __name__=='__main__':main()
