"""独立冻结旧PDF读取→代理恢复→完整去重→几何索引；不注入字符。"""
import json
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/pdf-native-geometry.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.native_text_geometry import _extract_page_text_geometry
    cases=[]
    for name in ('pdf-character-primitives','pdf-character-edges'):
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            with PDFDocument((ROOT/c['pdf']).read_bytes())as doc:
                with doc._open_page(c['page_index'])as page:
                    g=_extract_page_text_geometry(page,include_extended_geometry=c['extended'])
                    cases.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],extended=c['extended'],expected=dict(page_size=list(page.get_size()),page_bbox=list(page.get_bbox()),rotation=page.get_rotation(),raw_char_count=c['expected']['raw_char_count'],chars=encode(g.chars),tight_bboxes=encode(g.tight_bboxes),loose_bboxes=encode(g.loose_bboxes),origins=encode(g.origins))))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/native_text_geometry.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/extract.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/dedup.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual full PDF/geometry cases;',sum(len(c['expected']['chars'])for c in cases),'retained chars')
if __name__=='__main__':main()
