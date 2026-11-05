"""Actual PDFium source/geometry through the original calibrated page sequence."""
import copy,json
from dataclasses import asdict
from common import ROOT,write_json,digest
from export_native_document_pages import reference
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/native-document-pages-runtime.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from loguru import logger
    logger.remove();f=json.loads((ROOT/'build/migration/native-marginal-runtime.json').read_text(encoding='utf-8'));data=(ROOT/f['pdf']).read_bytes();pages=[]
    with PDFDocument(data)as doc:
        for i,s in enumerate(f['request']['sources']):
            g=doc.get_page_chars_with_geometry(i)
            geometry=dict(chars=g.chars,tight_bboxes=[dict(char_idx=j,bbox=b)for j,b in sorted(g.tight_bboxes.items())],origins=[dict(char_idx=j,origin=o)for j,o in sorted(g.origins.items())],loose_bboxes=[dict(char_idx=j,bbox=b)for j,b in sorted(g.loose_bboxes.items())])
            pages.append(clean(dict(source=copy.deepcopy(s),geometry=geometry)))
    q=dict(pages=pages);write_json(target,dict(pdf=f['pdf'],pdf_sha256=digest(data),request=q,expected=reference(q)));print('3 actual PDF pages through original containers and calibrated text')
if __name__=='__main__':main()
