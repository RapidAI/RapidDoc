"""Freeze actual page-wide fraction evidence exceeding diagnostic line capacity."""
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-page-fractions.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline import scripts
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel
    original=scripts._fraction_member_indices;cases=[];seen=set()
    def record(page_size,all_chars,tight_bboxes,drawing_lines,angle,prepared_rules=None):
        out=original(page_size,all_chars,tight_bboxes,drawing_lines,angle,prepared_rules)
        if len(all_chars)>1500:
            q=clean(dict(page_size=page_size,chars=all_chars,tight_bboxes=tight_bboxes,drawing_lines=[asdict(r)for r in drawing_lines],angle=angle))
            key=digest(__import__('json').dumps(q,sort_keys=True).encode())
            if key not in seen:seen.add(key);cases.append(dict(label=key,request=q,expected=sorted(out)))
        return out
    scripts._fraction_member_indices=record
    with PDFDocument((ROOT/'demo/pdfs/比亚迪财报.pdf').read_bytes())as doc:PdfModel().predict(doc)
    assert cases
    write_json(target,dict(cases=cases));print(len(cases),'actual page-wide fraction requests; sizes',[len(c['request']['chars'])for c in cases])
if __name__=='__main__':main()
