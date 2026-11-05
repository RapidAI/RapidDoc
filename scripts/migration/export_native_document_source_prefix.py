"""Old document prefix: geometry, raw marginals, Form roles, header separators."""
import copy,json
from common import ROOT,write_json,digest
from export_native_geometry_plans import reference as geometry
from export_native_geometry import apply_ref,lines
from export_native_late_image_text import project
from export_form_runtime import typed,normalized_reference

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.auxiliary_text import _classify_raw_page_marginals
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _detect_repeated_header_separator_bboxes
    result=geometry(q);pages=copy.deepcopy(q['pages'])
    for i,p in enumerate(pages):p['source']['lines']=apply_ref(dict(lines=p['source']['lines'],plan=result['plan'],page_index=i,allow_y_trim=False));p['geometry']['loose_bboxes']=result['remaining_loose_bboxes'][i]
    raw=[typed(p['source'])for p in pages];_classify_raw_page_marginals(raw)
    for p,s in zip(pages,raw):p['source']['lines']=[project(l)for l in s.lines]
    normalized=normalized_reference([p['source']for p in pages]);separators=_detect_repeated_header_separator_bboxes([typed(s)for s in normalized])
    for p,s in zip(pages,normalized):p['source']=s
    return dict(pages=pages,plan=result['plan'],header_separator_bboxes=[[list(b)for b in sorted(bs)]for bs in separators])
def main():
    target=ROOT/'build/migration/native-document-source-prefix.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q)
        for p in q['pages']:p['source']['lines']=[project(l)for l in lines(p['source']['lines'])]
        cases.append(dict(label=label,request=q,expected=reference(q)))
    f=json.loads((ROOT/'build/migration/native-marginals.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        add('marginals/'+c['label'],dict(pages=[dict(source=s,geometry=dict(chars=s.get('chars',[]),tight_bboxes=[],loose_bboxes=[],origins=[]))for s in c['request']['sources']]))
    for name in ['native-geometry-plans','native-geometry-plan-edges','native-geometry-plan-proofs']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:add(name+'/'+c['label'],c['request'])
    forms=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text(encoding='utf-8'));real=json.loads((ROOT/'build/migration/native-geometry-real-document.json').read_text(encoding='utf-8'))['case']['request'];pages=copy.deepcopy(real['pages'])
    for p,s in zip(pages,forms['request']['sources']):p['source'].update({k:v for k,v in s.items()if k!='lines'})
    add('live-form-geometry-document',dict(pages=pages))
    write_json(target,dict(cases=cases,sources={k:digest((ROOT/f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{k}').read_bytes())for k in ['char_geometry.py','auxiliary_text.py','form_roles.py','pipeline.py']}));print(len(cases),'complete document source prefixes')
if __name__=='__main__':main()
