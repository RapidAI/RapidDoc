"""Freeze large materialized candidate sets from actual Python PDF analysis."""
from dataclasses import asdict
from common import ROOT,write_json
from export_native_table_candidate_merge import encode
def main():
    target=ROOT/'build/migration/native-pdf-candidate-capacity-canonical-members.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.models import PdfModel
    original=table_rules._merge_table_candidates;cases=[]
    def packed(c):
        raw=asdict(c);raw['line_indices']=sorted(c.line_indices)
        for a,original_annotation in zip(raw['annotations'],c.annotations,strict=True):a['line_indices']=sorted(original_annotation.line_indices)
        return encode(raw)
    def record(candidates):
        materialized=[c.materialize()if isinstance(c,table_rules._RuleCandidateDraft)else c for c in sorted(candidates,key=lambda c:c.score,reverse=True)]
        raw=[packed(c)for c in materialized]
        total=sum(len(c['line_indices'])+sum(len(a['line_indices'])+len(a['line_bboxes'])for a in c['annotations'])for c in raw)
        out=original(materialized)
        if total>20000:cases.append(dict(label=f'actual-{len(cases)}',member_count=total,request=dict(candidates=raw),expected=dict(candidates=[packed(c)for c in out])))
        return out
    table_rules._merge_table_candidates=record
    with PDFDocument((ROOT/'demo/pdfs/比亚迪财报.pdf').read_bytes())as doc:PdfModel().predict(doc)
    assert cases
    write_json(target,dict(cases=cases));print([(c['member_count'],len(c['request']['candidates']))for c in cases])
if __name__=='__main__':main()
