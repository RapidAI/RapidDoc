"""Independent old preparation -> native recovery -> semantic table HTML facts."""
import json
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    target=ROOT/'build/migration/native-table-block-chains.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_materialization import recover_table_result
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle
    f=json.loads((ROOT/'build/migration/native-table-block.json').read_text(encoding='utf-8'));cases=[]
    for meta in f['real']:
        c=f['cases'][meta['case_index']];v=c['request'];p=c['expected'];s=v['source']
        ti=NativeTableInput(tuple(p['body_bbox']),tuple(s['page_size']),p['candidate']['angle'],tuple(p['native_chars']),tuple(NativeTableRule(**(r|dict(bbox=tuple(r['bbox']))))for r in s['drawing_lines']),tuple(NativeTableRectangle(**(r|dict(bbox=tuple(r['bbox']))))for r in meta['rectangles']))
        tight={q['char_idx']:q.get('tight_bbox')for q in s['chars']};origins={q['char_idx']:q.get('origin')for q in s['chars']}
        r=recover_table_result(ti,tight,origins)
        cases.append(dict(case_index=meta['case_index'],expected=None if r is None else dict(content=r[0],result=asdict(r[1]))))
    prefix='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'
    modules=['table_materialization.py','table_text_styles.py','_table_recovery/engine.py']
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in modules},cases=clean(cases)))
    print('Frozen',len(cases),'old block preparation -> full recovery/HTML chains;',sum(c['expected']is not None for c in cases),'nonempty')
if __name__=='__main__':main()
