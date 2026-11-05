"""Freeze complete original coverage replacement arbitration, preserving candidate metadata/order."""
import copy,random
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _prefer_complete_table_replacements
    names=['base','recovered','replacements'];groups=[[SimpleNamespace(**dict(c,bbox=tuple(c['bbox']),core_bbox=tuple(c['core_bbox'])if c.get('core_bbox')is not None else None))for c in q[k]]for k in names];raw={id(c):raw for group,k in zip(groups,names)for c,raw in zip(group,q[k])};result=_prefer_complete_table_replacements(*groups);return clean({k:[raw[id(c)]for c in group]for k,group in zip(names,result)})
def candidate(b=(50.,100.,250.,300.),tag='old',core=None):return dict(bbox=list(b),core_bbox=core,angle=0,score=100.,line_indices=[1,2],annotations=[dict(kind='caption',bbox=[50.,80.,150.,90.],line_indices=[3])],tag=tag)
def main():
    target=ROOT/'build/migration/native-table-replacements.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,base=(),recovered=(),replacements=()):
        q=clean(copy.deepcopy(dict(base=base,recovered=recovered,replacements=replacements)));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('empty');add('only-old',[candidate()]);add('only-new',replacements=[candidate(tag='new')]);add('equal',[candidate()],replacements=[candidate(tag='new')])
    add('partial',[candidate()],replacements=[candidate([50.,100.,250.,200.],tag='half')]);add('duplicate-half',[candidate()],replacements=[candidate([50.,100.,250.,200.],tag='half')]*3)
    add('two-halves',[candidate()],replacements=[candidate([50.,100.,250.,200.],tag='top'),candidate([50.,200.,250.,300.],tag='bottom')])
    for bottom in [279.99999,280.,280.00001]:add('coverage-'+str(bottom),[candidate()],replacements=[candidate([50.,100.,250.,bottom],tag='new')])
    for left in [69.99999,70.,70.00001]:add('overlap-'+str(left),[candidate()],replacements=[candidate([left,100.,left+200.,300.],tag='shift')])
    for core in [None,[],[50.,100.,250.,300.]]:add('core-'+str(core),[candidate(core=core)],replacements=[candidate(tag='new')])
    add('core-priority',[candidate([0.,0.,400.,600.],core=[50.,100.,250.,300.])],replacements=[candidate(tag='new')])
    add('recovered',recovered=[candidate()],replacements=[candidate(tag='new')]);add('same-in-two-old',[candidate()],recovered=[candidate()],replacements=[candidate(tag='new')])
    add('global-rejection',[candidate()],recovered=[candidate([50.,100.,250.,500.],tag='long')],replacements=[candidate(tag='new')])
    add('rejected-pieces-still-in-next-coverage',[candidate([50.,100.,250.,500.])],recovered=[candidate()],replacements=[candidate(tag='top'),candidate([50.,300.,250.,500.],tag='bottom')])
    add('independent',[candidate()],[candidate([300.,100.,400.,300.])],[candidate(tag='new')])
    for scale in [.01,1.,1000000.]:
        old=candidate();new=[candidate([50.,100.,250.,200.],tag='top'),candidate([50.,200.,250.,300.],tag='bottom')]
        for c in [old,*new]:c['bbox']=[v*scale for v in c['bbox']]
        add('scale-'+str(scale),[old],replacements=new)
    add('integer',[candidate([0,0,100,100])],replacements=[candidate([0,0,100,90],tag='new')]);add('mixed',[candidate([0,0,100,100])],replacements=[candidate([0.,0,100.,90],tag='new')])
    rng=random.Random(261114)
    for i in range(130):
        old=[candidate(tag='old'),candidate([300.,100.,400.,300.],tag='other')]if i%3 else[candidate()];new=[]
        for j in range(rng.randrange(1,7)):
            top=rng.choice([100.,120.,180.,200.]);bottom=rng.choice([220.,260.,280.,300.]);left=rng.choice([50.,50.001,60.,70.]);new.append(candidate([left,top,250.,bottom],tag=f'n{j}'))
        if i%5==0:new+=copy.deepcopy(new)
        rng.shuffle(new);add('pieces-'+str(i),old[:1],old[1:],new)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'old replacement cases;',sum(c['request']!=c['expected']for c in cases),'changed;',sum(len(c['request']['replacements'])>len(c['expected']['replacements'])for c in cases),'replacement-rejection cases')
if __name__=='__main__':main()
