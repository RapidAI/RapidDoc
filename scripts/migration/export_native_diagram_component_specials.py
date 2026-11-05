"""Separate positive timeline/prose corpus; retain primary old missing-tight errors."""
import copy,json
from common import ROOT,write_json
from export_native_diagram_components import reference,base
def main():
    target=ROOT/'build/migration/native-diagram-component-specials.json'
    if target.exists():raise FileExistsError(target)
    original=json.loads((ROOT/'build/migration/native-diagram-components.json').read_text(encoding='utf-8'));cases=[];errors=[]
    def add(label,q):
        try:out=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=out))
    for c in original['errors']:
        q=copy.deepcopy(c['request'])
        for l in q['source'].get('lines',[]):l['chars']=[dict(tight_bbox=l['bbox'])]
        add('tight-'+c['label'],q)
    timeline=next(c['request']for c in cases if c['label']=='tight-timeline')
    for color in [[229,255,255,255],[230,10,10,0],[230,230,230,0],[255,255,255,0],None]:
        q=copy.deepcopy(timeline);q['source']['path_infos'][-1]['fill_rgba']=color;add('strip-color-'+str(color),q)
    for y in [160.,200.,359.9999,360.,360.0001,400.]:
        q=copy.deepcopy(timeline);q['source']['lines'][-1]['bbox']=[340.,y,370.,y+10.];q['source']['lines'][-1]['chars']=[dict(tight_bbox=q['source']['lines'][-1]['bbox'])];add('label-depth-'+str(y),q)
    for count in [24,25,26]:
        q=copy.deepcopy(timeline)
        for l in q['source']['lines']:l['text']='x'*count
        add('label-length-'+str(count),q)
    for n in [4,8,16]:
        q=base();q['source']['path_infos']*=n;add('duplicate-paths-'+str(n),q)
    write_json(target,dict(sources=original['sources'],scope='Additional tight glyph-backed timeline and prose paths; primary old missing-tight error facts remain immutable.',cases=cases,errors=errors));print('Specials',len(cases),'errors',len(errors),'timeline',next(c['expected']for c in cases if c['label']=='tight-timeline'))
if __name__=='__main__':main()
