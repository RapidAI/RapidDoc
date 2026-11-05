"""冻结旧DocVortex去重前原始字符；独立读取真实PDF，不注入字符或几何。"""
import json
from contextlib import closing
from common import ROOT,digest,write_json

def encode(x):
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    if isinstance(x,Bbox):return list(x.bbox)
    if isinstance(x,dict):return {k:encode(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encode(v)for v in x]
    return x

def main():
    target=ROOT/'build/migration/pdf-character-primitives.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.text.extract import get_chars
    from rapid_doc._vendor.docvortex_core.document.pdf.native_text_geometry import _restore_pdfium_surrogate_pairs
    vectors=json.loads((ROOT/'build/migration/pdf-vector-primitives.json').read_text(encoding='utf-8'))
    cases=[]
    for c in vectors['cases']:
        name,index=c['pdf'],c['page_index']
        if name.endswith('示例1-论文模板.pdf')and index not in (0,3,10,13):continue
        if name.endswith('比亚迪财报.pdf')and index not in (0,1,8):continue
        with PDFDocument((ROOT/name).read_bytes())as doc:
            with doc._open_page(index)as page:
                with closing(page.get_textpage())as textpage:
                    for extended in (False,True):
                        chars=get_chars(textpage,list(page.get_bbox()),page.get_rotation(),include_geometry=extended)
                        raw=encode(chars)
                        restored=_restore_pdfium_surrogate_pairs(chars,textpage,raw_codes={x['char_idx']:x['raw_code']for x in chars})
                        cases.append(dict(pdf=name,pdf_sha256=c['pdf_sha256'],page_index=index,extended=extended,expected=dict(page_size=list(page.get_size()),page_bbox=list(page.get_bbox()),rotation=page.get_rotation(),raw_char_count=textpage.count_chars(),chars=raw,restored_chars=encode(restored))))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/text/extract.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/_contracts.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_text_geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'actual PDF/mode cases,',sum(len(c['expected']['chars'])for c in cases),'raw chars')
if __name__=='__main__':main()
