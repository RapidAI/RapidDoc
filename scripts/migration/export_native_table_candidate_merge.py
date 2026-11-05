"""Freeze old materialized-candidate merge and annotation ordering/conflict semantics."""
import copy,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def encode(c):
    c=copy.deepcopy(c)
    for a in c.get('annotations',[]):
        if isinstance(a['line_bboxes'],dict):a['line_bboxes']=[[int(k),v]for k,v in a['line_bboxes'].items()]
    return clean(c)
def objects(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TableCandidate,_TableAnnotation
    out=[]
    for c in copy.deepcopy(raw):
        c['line_indices']=set(c['line_indices']);c['bbox']=tuple(c['bbox']);c['local_bbox']=tuple(c['local_bbox']);c['core_bbox']=tuple(c['core_bbox'])if c.get('core_bbox')is not None else None
        c['annotations']=[_TableAnnotation(a['kind'],tuple(a['bbox']),set(a['line_indices']),{int(k):tuple(b)for k,b in a['line_bboxes']})for a in c['annotations']];out.append(_TableCandidate(**c))
    return out
def reference(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _merge_table_candidates
    return dict(candidates=[encode(asdict(c))for c in _merge_table_candidates(objects(raw))])
def candidate(b=(50.,100.,150.,200.),score=100.,ids=(1,),angle=0,core=None,annotations=()):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TableCandidate
    c=asdict(_TableCandidate(tuple(b),tuple(b),angle,score,core_bbox=core,line_indices=set(ids)));c['annotations']=copy.deepcopy(list(annotations));return encode(c)
def annotation(kind='caption',ids=(8,9),reverse=False):
    pairs=[[8,[50.,70.,90.,80.]],[9,[90.,70.,130.,80.]]]
    if reverse:pairs.reverse()
    return dict(kind=kind,bbox=[50.,70.,130.,80.],line_indices=list(ids),line_bboxes=pairs)
def main():
    target=ROOT/'build/migration/native-table-candidate-merge.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,cs):
        raw=clean(copy.deepcopy(cs));cases.append(dict(label=label,request=dict(candidates=raw),expected=reference(raw)))
    add('empty',[]);add('single',[candidate()]);add('duplicate',[candidate(),candidate()])
    for score in [99.99999,100,100.,100.00001,101.]:add('weak-barrier-'+str(score),[candidate(),candidate([100.,100.,200.,200.],score=score,ids=[2])])
    for x in [129.99999,130.,130.00001]:add('overlap-'+str(x),[candidate(),candidate([x,100.,x+100.,200.],score=100.,ids=[2])])
    for angle in [0,90,180,270,15]:add('angle-'+str(angle),[candidate(),candidate([100.,100.,200.,200.],ids=[2],angle=angle)])
    for core in [None,[60.,110.,140.,190.]]:
        for other in [None,[100.,110.,190.,190.]]:add('core-'+str(core)+str(other),[candidate(core=core),candidate([100.,100.,200.,200.],ids=[2],core=other)])
    add('first-target-not-transitive',[candidate([0.,0.,100.,100.],score=103.,ids=[1]),candidate([160.,0.,260.,100.],score=102.,ids=[2]),candidate([80.,0.,180.,100.],score=101.,ids=[3])])
    add('dynamic-expanded-target',[candidate([0.,0.,100.,100.],score=103.),candidate([80.,0.,180.,100.],score=102.),candidate([160.,0.,260.,100.],score=101.)])
    for body in [[],[8],[9],[8,9]]:
        for reverse in [False,True]:add('body-conflict-'+str(body)+str(reverse),[candidate(annotations=[annotation(reverse=reverse)]),candidate(ids=body,annotations=[annotation('footnote',ids=[10])])])
    add('same-kind-union',[candidate(annotations=[annotation()]),candidate(annotations=[dict(kind='caption',bbox=[30.,60.,100.,80.],line_indices=[8,10],line_bboxes=[[8,[30.,60.,60.,70.]],[10,[60.,60.,100.,70.]]])])])
    add('no-line-boxes',[candidate(annotations=[dict(kind='caption',bbox=[50.,60.,130.,80.],line_indices=[8],line_bboxes=[])]),candidate(ids=[9])])
    add('duplicate-kinds-first-match',[candidate(annotations=[annotation(),annotation(ids=[10])]),candidate(annotations=[annotation()])])
    for reverse in [False,True]:
        a=dict(kind='caption',bbox=[0,0,10,10],line_indices=[8,9],line_bboxes=[[8,[0,0,10,10]],[9,[0.,0.,10.,10.]]]);
        if reverse:a['line_bboxes'].reverse()
        add('bbox-tie-types-'+str(reverse),[candidate(annotations=[a]),candidate(ids=[])])
    add('unmerged-no-conflict-cleanup',[candidate(ids=[8],annotations=[annotation()])])
    for flag in ['inferred_grid_authoritative','preserve_inline_font_styles']:
        a=candidate();b=candidate();b[flag]=True;b['inferred_grid']=[dict(bbox=[50,100,150,100],width=1,orientation='horizontal')];add('target-extra-fields-'+flag,[a,b])
    rng=random.Random(261115)
    for i in range(120):
        cs=[]
        for j in range(rng.randrange(2,9)):
            x=rng.choice([0.,50.,80.,100.,130.,150.]);y=rng.choice([0.,50.,80.]);cs.append(candidate([x,y,x+100.,y+100.],score=rng.choice([99.,100.,101.]),ids=[rng.randrange(-3,10)],angle=rng.choice([0,0,90]),annotations=[annotation(reverse=bool(j%2))]if j%2 else[]))
        rng.shuffle(cs);add('combined-'+str(i),cs)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Materialized _TableCandidate only; annotations.line_bboxes encoded as ordered [id,bbox] pairs.',cases=cases));print('Frozen',len(cases),'old materialized merge cases;',sum(c['request']!=c['expected']for c in cases),'changed')
if __name__=='__main__':main()
