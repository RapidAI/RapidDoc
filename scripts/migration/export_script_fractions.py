"""冻结旧分式保护；原索引分支与prepared规则直接调用，不替换候选。"""
import copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/script-fractions.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _fraction_member_indices,_prepare_fraction_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    cases=[]
    def add(label,chars,tight,rules,size,angle,prepared=False,override=None):
        drawings=[SimpleNamespace(bbox=r['bbox'])if isinstance(r,dict)else r for r in rules]
        values=_prepare_fraction_rules(drawings,tuple(size),angle)if prepared else None
        if override is not None:values=override
        expected=sorted(_fraction_member_indices(tuple(size),chars,tight,drawings,angle,values))
        cases.append(dict(label=label,request=dict(chars=chars,tight_bboxes=tight,drawing_lines=rules,page_size=size,angle=angle,prepared_rules=values),expected=expected))
    rng=random.Random(167283)
    for n in range(420):
        size=[300.,400.];angle=(0,90,180,270)[n%4];h=rng.choice([4.,6.,8.,10.]);y=60.;chars=[];tight={}
        local=[(0,'a',[12.,y-h-1.,18.,y-1.]),(1,'2',[12.,y+1.,18.,y+h+1.])]
        for i in range(40 if n%3==0 else 3):
            x=rng.uniform(70,180);yy=rng.uniform(10,140);local.append((i+2,rng.choice('ABC12 †'),[x,yy,x+5.,yy+h]))
        for i,t,b in local:
            box=list(_rotate_bbox_from_upright(tuple(b),tuple(size),angle));chars.append(dict(char=t,char_idx=i-2,bbox=box));tight[i-2]=box
        width=rng.choice([2.,.45*h,.45*h-.000001,4.*h,8.*h,8.*h+.000001,13.*h])
        thick=rng.choice([.25,1.25,.25*h,.25*h+.000001,3.]);rule=[10.,y-thick/2,10.+width,y+thick/2]
        rules=[dict(bbox=list(_rotate_bbox_from_upright(tuple(rule),tuple(size),angle)))]
        if n%3==0:
            for z in range(4):rules.append(dict(bbox=list(_rotate_bbox_from_upright((90.,float(z*20+10),100.,float(z*20+10.5)),tuple(size),angle))))
        for prepared in (False,True):add(f'fraction:{n}:{prepared}',chars,tight,rules,size,angle,prepared)
    for prepared in (False,True):
        chars=[dict(char='a',char_idx=0,bbox=[12.,50.,18.,56.]),dict(char='b',char_idx=1,bbox=[12.,64.,18.,70.])];tight={c['char_idx']:c['bbox']for c in chars}
        add(f'degenerate:{prepared}',chars,tight,[[10.,60.,30.,60.]],[300.,400.],0,prepared,[[10.,60.,30.,60.]]if prepared else None)
    add('empty',[],{},[],[300.,400.],0)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/scripts.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'fraction cases,',sum(bool(c['expected'])for c in cases),'positive; indexed and all four rotations included')
if __name__=='__main__':main()
