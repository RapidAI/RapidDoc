"""Actual original detached-page and guarded region recovery references."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-host-tables.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page,recover_table_region
    from rapid_doc.model.docvortex_adapter import _table_text_angle
    refs=json.loads((ROOT/'build/migration/native-pdf-acceptance.json').read_text(encoding='utf-8'))
    cases=[];pages=[]
    for doc in refs['cases']:
        data=(ROOT/doc['pdf']).read_bytes()
        with PDFDocument(data)as pdf:
            for index,blocks in enumerate(doc['expected']['pages']):
                tables=[b for b in blocks if b['type']=='table']
                if not tables:continue
                page=prepare_table_page(pdf[index]);g=page.geometry
                expected=clean(dict(page_size=page.page_size,chars=g.chars,tight_bboxes=g.tight_bboxes,origins=g.origins,drawing_lines=page.drawing_lines,rectangles=page.rectangles))
                pages.append(dict(pdf=doc['pdf'],pdf_sha256=digest(data),page_index=index,expected=expected))
                for b in tables:
                    w,h=page.page_size;bb=[b['bbox'][0]*w,b['bbox'][1]*h,b['bbox'][2]*w,b['bbox'][3]*h];angle=_table_text_angle(page,bb)
                    result=recover_table_region(page,tuple(bb),angle=angle)
                    cases.append(dict(label=f'{doc["label"]}/{index}/{len(cases)}',pdf=doc['pdf'],page_index=index,bbox=bb,angle=angle,expected=None if result is None else clean(asdict(result))))
    write_json(target,dict(pages=pages,cases=cases));print(len(pages),'actual detached pages;',len(cases),'original region recoveries;',sum(c['expected']is not None for c in cases),'positives')
if __name__=='__main__':main()
