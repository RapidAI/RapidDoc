"""冻结显式已有角色条件下的旧数值上标补充。"""
import copy,json,random
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/script-numeric-extra.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._script_geometry import build_script_features,_numeric_superscript_indices
    f=json.loads((ROOT/'build/migration/script-geometry.json').read_text(encoding='utf-8'));rng=random.Random(83271);cases=[]
    selected=[c for i,c in enumerate(f['cases'])if c['expected']['numeric_indices']or i%23==0]
    for c in selected:
        for mode in ('classified','masked'):
            r=copy.deepcopy(c['request']);roles=copy.deepcopy(c['expected']['roles'])if mode=='classified'else[rng.choice(['body','body','sup','sub'])for _ in r['chars']];r['roles']=roles
            tight={int(k):v for k,v in r['tight_bboxes'].items()};origins={int(k):v for k,v in r['origins'].items()};protected=set(r['protected_body_indices'])
            expected=sorted(_numeric_superscript_indices(build_script_features(r['chars'],tight,origins,protected),roles,protected))
            cases.append(dict(label=c['label']+':'+mode,request=r,expected=expected))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_script_geometry.py';write_json(target,dict(sources={source:digest((ROOT/source).read_bytes())},cases=cases));print('Frozen',len(cases),'numeric cases with explicit roles,',sum(bool(c['expected'])for c in cases),'positive')
if __name__=='__main__':main()
