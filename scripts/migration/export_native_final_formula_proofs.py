"""Positive canonical replacement, index and unmapped-ink evidence."""
import copy
from common import ROOT,write_json
from export_native_final_formula import reference
from export_native_page_containers import index_base
from export_native_late_image_text import project
from export_native_geometry import lines
from export_native_title_layout import line

def main():
    target=ROOT/'build/migration/native-final-formula-proofs.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,p,check):
        p=copy.deepcopy(p)
        for key in ['remaining_lines','canonical_formula_source_lines','formula_candidate_lines']:
            if key in p:p[key]=[project(l)for l in lines(p[key])]
        q=dict(page=p);out=reference(q);check(out);cases.append(dict(label=label,request=q,expected=out))
    for kind in ['leaders','paged','unpaged']:
        q=index_base(kind);p=dict(page_size=q['page_size'],remaining_lines=q['lines'],table_bboxes=[],fixed_blocks=[],drawing_lines=[])
        add('index/'+kind,p,lambda out:True if out['index_blocks']else (_ for _ in ()).throw(AssertionError('index missing')))
    ls=[line('Original complete body paragraph',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])]+[dict(line('x = y + z (1)',[130.,190.,370.,200.],20,font='Math'),style_scale_repaired=True)]
    p=dict(page_size=[400.,600.],remaining_lines=ls,table_bboxes=[],fixed_blocks=[],drawing_lines=[],canonical_formula_geometry=True,canonical_formula_source_lines=copy.deepcopy(ls))
    p['canonical_formula_source_lines'][-1]['text']='a = b + c (1)'
    add('canonical/replaces-equal-inventory',p,lambda o:None if 'a = b + c'in o['formula_blocks'][0]['content']else (_ for _ in ()).throw(AssertionError('canonical not substituted')))
    p['canonical_formula_source_lines'][-1]['bbox'][0]=140.
    add('canonical/rejects-different-inventory',p,lambda o:None if 'x = y + z'in o['formula_blocks'][0]['content']else (_ for _ in ()).throw(AssertionError('normal not retained')))
    p.pop('canonical_formula_source_lines');p['canonical_formula_geometry']=False;p['formula_ink_bboxes']=[[128.,190.,130.,200.]]
    add('ink/uniquely-attaches',p,lambda o:None if o['formula_blocks'][0]['bbox'][0]==128. else (_ for _ in ()).throw(AssertionError('ink not attached')))
    write_json(target,dict(cases=cases));print(len(cases),'positive final-formula/index/canonical/ink proofs')
if __name__=='__main__':main()
