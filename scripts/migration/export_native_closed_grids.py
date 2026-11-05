"""Freeze full original closed-grid proof, source-ID semantics and rounding/errors."""
import copy,math,random,statistics
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import has_native_closed_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TableCandidate
    s=q['source'];source=SimpleNamespace(form_bboxes=[tuple(b)for b in s.get('form_bboxes',[])],drawing_lines=[SimpleNamespace(**dict(r,bbox=tuple(r['bbox'])))for r in s.get('drawing_lines',[])],lines=[SimpleNamespace(**dict(l,bbox=tuple(l['bbox']),source_index=l.get('source_index',0),effective_height=l.get('effective_height',0.)))for l in s.get('lines',[])])
    flags=[]
    for c in q['candidates']:
        candidate=_TableCandidate(bbox=tuple(c['bbox']),local_bbox=tuple(c['bbox']),angle=c.get('angle',0),score=0,core_bbox=tuple(c['core_bbox'])if c.get('core_bbox')is not None else None,line_indices=set(c.get('line_indices',[])))
        flags.append(has_native_closed_grid(source,candidate,allow_two_by_two=q.get('allow_two_by_two',False)))
    return dict(closed_grid_flags=flags)
def base(rows=3,cols=2):
    b=[40.,100.,240.,220.]
    rules=[dict(bbox=[40.,100.+r*120./rows,240.,100.+r*120./rows],orientation='horizontal')for r in range(rows+1)]+[dict(bbox=[40.+c*200./cols,100.,40.+c*200./cols,220.],orientation='vertical')for c in range(cols+1)]
    lines=[dict(bbox=[50.+c*200./cols,110.+r*120./rows,70.+c*200./cols,120.+r*120./rows],source_index=r*cols+c,effective_height=10.)for r in range(rows)for c in range(cols)]
    return dict(source=dict(lines=lines,drawing_lines=rules,form_bboxes=[]),candidates=[dict(bbox=b,angle=0,core_bbox=None,line_indices=[l['source_index']for l in lines])],allow_two_by_two=False)
def main():
    target=ROOT/'build/migration/native-closed-grids.json'
    if target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    def add(label,q=None):
        q=clean(copy.deepcopy(q or base()))
        try:expected=reference(q)
        except statistics.StatisticsError as e:errors.append(dict(label=label,request=q,old_error=type(e).__name__,expected_code='invalid_input'));return
        cases.append(dict(label=label,request=q,expected=expected))
    add('base')
    for rows in [1,2,3,4]:
        for cols in [1,2,3]:
            for allow in [False,True]:q=base(rows,cols);q['allow_two_by_two']=allow;add(f'layout-{rows}-{cols}-{allow}',q)
    for angle in [0.,90,180,270,15,.00001]:q=base();q['candidates'][0]['angle']=angle;add('angle-'+str(angle),q)
    for core in [None,[],[0.,0.,0.,0.],[40.,100.,240.,220.]]:q=base();q['candidates'][0]['core_bbox']=core;add('core-'+str(core),q)
    q=base();q['candidates'][0]['bbox']=[0,0,1,1];q['candidates'][0]['core_bbox']=[40.,100.,240.,220.];add('prefer-core',q)
    for bbox in [[40.,100.,340.,220.],[40.,100.,340.00001,220.],[0.,0.,400.,600.]]:q=base();q['source']['form_bboxes']=[bbox];add('larger-form-'+str(bbox),q)
    for left in [49.99999,50.,50.00001]:q=base();q['source']['form_bboxes']=[[left,80.,400.,240.]];add('form-95percent-'+str(left),q)
    for count in [0,3,4,5,6]:
        for allow in [False,True]:q=base();q['source']['lines']=q['source']['lines'][:count];q['allow_two_by_two']=allow;add(f'members-{count}-{allow}',q)
    for height in [0.,-1.,1.,5.,10.,20.]:
        q=base()
        for l in q['source']['lines']:l['effective_height']=height
        add('effective-'+str(height),q)
    q=base()
    for l in q['source']['lines']:l['effective_height']=0.
    q['source']['lines'][0]['effective_height']=10.;add('median-positive-only',q)
    q=base();q['source']['lines']*=2;add('duplicate-lines',q)
    q=base()
    for l in q['source']['lines']:l['source_index']=0
    q['candidates'][0]['line_indices']=[0];add('duplicate-source-member-count',q)
    q=base();q['candidates'][0]['line_indices']=[999];add('nonmember-ids',q)
    q=base();q['candidates'][0]['line_indices']*=2;add('duplicate-request-ids',q)
    q=base()
    for l in q['source']['lines']:l['source_index']=-l['source_index']-1
    q['candidates'][0]['line_indices']=[l['source_index']for l in q['source']['lines']];add('signed-source-ids',q)
    for direction in ['horizontal','vertical']:
        for shift in [1.99999,2.,2.00001]:
            q=base()
            for r in q['source']['drawing_lines']:
                if r['orientation']==direction:r['bbox'][0 if direction=='horizontal'else 1]+=shift
            add('span-'+direction+str(shift),q)
    for edge in [40.004999,40.005,40.005001]:
        q=base();q['source']['drawing_lines'][-2]['bbox'][0]=q['source']['drawing_lines'][-2]['bbox'][2]=edge
        for l in q['source']['lines'][::2]:l['bbox'][0]=39.995;l['bbox'][2]=40.015
        add('rounded-track-'+str(edge),q)
    q=base()
    for i,l in enumerate(q['source']['lines']):l['bbox'][0]=-50. if i%2==0 else 300.;l['bbox'][2]=l['bbox'][0]+20.
    add('outside-member-centres-retained',q)
    q=base()
    for l in q['source']['lines']:l['bbox'][0]=50.;l['bbox'][2]=70.
    add('single-occupied-column',q)
    for scale in [1.,1000000.]:
        q=base()
        for l in q['source']['lines']:l['bbox']=[int(x*scale)for x in l['bbox']];l['effective_height']*=scale
        for r in q['source']['drawing_lines']:r['bbox']=[int(x*scale)for x in r['bbox']]
        q['candidates'][0]['bbox']=[int(x*scale)for x in q['candidates'][0]['bbox']];q['source']['form_bboxes']=[[int(40*scale),int(100*scale),int(340*scale+1),int(220*scale)]];add('large-int-form-'+str(scale),q)
    q=base();q['candidates']*=2;add('batch-duplicate-candidates',q)
    rng=random.Random(261113)
    for i in range(100):
        q=base(rows=rng.choice([2,3,4]));q['allow_two_by_two']=bool(i%2);rng.shuffle(q['source']['lines']);rng.shuffle(q['source']['drawing_lines']);r=rng.choice(q['source']['drawing_lines']);r['bbox'][rng.randrange(4)]+=rng.choice([-.005,.005,-2.,2.,2.001]);add('geometry-'+str(i),q)
    rounding=[]
    for x in [0.005,.015,1.005,1.015,2.675,40.005,-1.005,1e8+.005]:
        for v in [math.nextafter(x,-math.inf),x,math.nextafter(x,math.inf)]:rounding.append(dict(value=v,expected=round(v,2)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,errors=errors,rounding=rounding));print('Frozen',len(cases),'closed-grid cases;',sum(any(c['expected']['closed_grid_flags'])for c in cases),'positive;',len(errors),'old median errors;',len(rounding),'rounding facts')
if __name__=='__main__':main()
