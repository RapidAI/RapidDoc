"""Original full generator followed by allow_y_trim=False, before marginal classification."""
import copy,json
from common import ROOT,write_json
from export_native_geometry_plans import reference
from export_native_geometry import apply_ref
def main():
    target=ROOT/'build/migration/native-geometry-prefix.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    for name in ['native-geometry-plans','native-geometry-plan-edges','native-geometry-plan-proofs','native-geometry-real-document','native-geometry-font-contract']:
        data=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'));rows=[data['case']]if'case'in data else data['cases']
        for c in rows:
            if 'error'in c:continue
            q=copy.deepcopy(c['request']);plan_result=reference(q);pages=copy.deepcopy(q['pages'])
            for i,p in enumerate(pages):
                p['source']['lines']=apply_ref(dict(lines=p['source']['lines'],plan=plan_result['plan'],page_index=i,allow_y_trim=False));p['geometry']['loose_bboxes']=plan_result['remaining_loose_bboxes'][i]
            cases.append(dict(label=name+'/'+c['label'],request=q,expected=dict(pages=pages,plan=plan_result['plan'])))
    write_json(target,dict(cases=cases));print(len(cases),'full document geometry prefixes')
if __name__=='__main__':main()
