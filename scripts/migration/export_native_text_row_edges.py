"""Positive CJK paragraph and Python Unicode boundary/source identity contracts."""
import copy,json
from common import ROOT,write_json
from export_native_text_rows import reference
from export_native_geometry import lines
from export_native_late_image_text import project
def main():
    target=ROOT/'build/migration/native-text-row-edges.json'
    if target.exists():raise FileExistsError(target)
    old=json.loads((ROOT/'build/migration/native-text-rows.json').read_text(encoding='utf-8'))['cases'];cases=[]
    for name in ['cjk-prose','cjk-label','italic','prose','hanging','structured','front','explicit']:
        base=copy.deepcopy(next(c['request']for c in old if c['label']=='direct/'+name))
        for mode in ['base','exact-threshold','below-threshold','word-mark','unicode-case','source-collision','source-negative','ink-overlap','semantic-barrier']:
            q=copy.deepcopy(base)
            if name.startswith('cjk-'):
                for i,l in enumerate(q['lines']):l.update(bbox=[30.,180.+22*i,370.,190.+22*i])
            if mode in ['exact-threshold','below-threshold']and name.startswith('cjk-'):
                gap=10. if name=='cjk-prose' else 9.;gap-=1e-6 if mode=='below-threshold' else 0.
                for i,l in enumerate(q['lines']):l['bbox'][1]=180.+(10.+gap)*i;l['bbox'][3]=l['bbox'][1]+10.
            if mode=='word-mark':
                for l in q['lines']:l['text']=l['text'].replace('Original','Original\u0301').replace('Following','Following\u0301')
            if mode=='unicode-case':
                for l in q['lines']:l['text']=l['text'].replace('Keywords','Keywords').replace('Original','Orıginal')
            if mode=='source-collision':
                for l in q['lines']:l['source_index']=7
            if mode=='source-negative':
                for i,l in enumerate(q['lines']):l['source_index']=[-1,-2,-17,31][i%4]
            if mode=='ink-overlap':
                for l in q['lines']:l['ink_bbox']=[30.,180.,370.,230.]
            if mode=='semantic-barrier'and len(q['lines'])>1:q['lines'][1]['semantic_type']='paragraph_title'
            q['lines']=[project(l)for l in lines(q['lines'])];cases.append(dict(label=name+'/'+mode,request=q,expected=reference(q)))
    write_json(target,dict(cases=cases));print(len(cases),'row edge contracts;',sum(bool(c['expected']['cjk_prose'])for c in cases),'positive CJK paragraph cases')
if __name__=='__main__':main()
