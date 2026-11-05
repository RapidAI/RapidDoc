"""Freeze original prose-spanning draft filter on explicit upstream draft projections."""
import copy,random
from types import SimpleNamespace as NS
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _exclude_prose_spanning_rule_drafts_python
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _RuleCandidateDraft
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TableCandidate
    s=q['source'];source=NS(lines=[NS(**dict(l,angle=l.get('angle',0),caption_start=l.get('caption_start',False),formula_candidate_only=l.get('formula_candidate_only',False),semantic_type=l.get('semantic_type'),dominant_font_weight=l.get('dominant_font_weight')))for l in s.get('lines',[])],drawing_lines=[NS(**r)for r in s.get('drawing_lines',[])])
    cs=[]
    for c in q['candidates']:
        if c['kind']=='table':cs.append(_TableCandidate(tuple(c['bbox']),tuple(c['bbox']),c.get('angle',0),100.,core_bbox=tuple(c['core_bbox'])if c.get('core_bbox')is not None else None))
        else:cs.append(_RuleCandidateDraft(NS(angle=c.get('angle',0),median_height=c.get('median_height')), [NS(bbox=b)for b in c.get('boundaries',[])],[NS(fragments=[NS(line_index=i)for i in c.get('member_source_indices',[])])],c.get('caption'),[],False,1.))
    kept=_exclude_prose_spanning_rule_drafts_python(source,cs);indices={id(c)for c in kept};return dict(candidates=[c for c,o in zip(q['candidates'],cs)if id(o)in indices],rejected_indices=[i for i,o in enumerate(cs)if id(o)not in indices])
def base():
    lines=[dict(text='These ordinary words form a separate prose sentence.',bbox=[50.,120.+i*15.,240.,130.+i*15.],source_index=i,angle=0)for i in range(3)]
    rules=[dict(bbox=[x,220.,x,320.],orientation='vertical')for x in [50.,150.,250.]]+[dict(bbox=[50.,y,250.,y],orientation='horizontal')for y in [220.,320.]]
    return dict(source=dict(lines=lines,drawing_lines=rules),candidates=[dict(kind='draft',angle=0,median_height=10.,caption=None,boundaries=[[50.,100.,250.,100.],[50.,320.,250.,320.]],member_source_indices=[0,1,2]),dict(kind='table',angle=0,bbox=[50.,220.,250.,320.],core_bbox=None)])
def main():
    target=ROOT/'build/migration/native-prose-rule-conflicts.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q=None):q=clean(copy.deepcopy(q or base()));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('base')
    for kind in ['no-grid','no-draft','caption','nonzero-draft','nonzero-grid','no-boundary','zero-em','negative-em','missing-members']:
        q=base()
        if kind=='no-grid':q['candidates']=q['candidates'][:1]
        if kind=='no-draft':q['candidates']=q['candidates'][1:]
        if kind=='caption':q['candidates'][0]['caption']={}
        if kind=='nonzero-draft':q['candidates'][0]['angle']=90
        if kind=='nonzero-grid':q['candidates'][1]['angle']=90
        if kind=='no-boundary':q['candidates'][0]['boundaries']=[]
        if kind in ['zero-em','negative-em']:q['candidates'][0]['median_height']=0. if kind=='zero-em'else -1.
        if kind=='missing-members':q['candidates'][0]['member_source_indices']=[]
        add(kind,q)
    for weight in [None,0.,400.,599.999,600.,700.]:
        q=base()
        for l in q['source']['lines']:l['dominant_font_weight']=weight
        add('weight-'+str(weight),q)
    for key,value in [('angle',90),('caption_start',True),('formula_candidate_only',True),('semantic_type','paragraph_title'),('semantic_type','')]:
        q=base()
        for l in q['source']['lines']:l[key]=value
        add(key+str(value),q)
    for text in ['one two three four five','one two three four five six','_one two three four five six','éone two three four five six','ab_cd two three four five six','one two three four five １２','one two three four five six。']:
        q=base()
        for l in q['source']['lines']:l['text']=text
        add('words-'+repr(text),q)
    for count in [0,1,2,3,4]:
        q=base();q['source']['lines']=(q['source']['lines']+[copy.deepcopy(q['source']['lines'][0])])[:count];add('line-count-'+str(count),q)
    for gap in [5.99999,6.,6.00001]:
        q=base()
        for i,l in enumerate(q['source']['lines']):l['bbox'][1]=120.+i*gap;l['bbox'][3]=130.+i*gap
        add('prose-row-gap-'+str(gap),q)
    for x in [152.99999,153.,153.00001]:
        q=base();q['source']['drawing_lines'][1]['bbox'][0]=q['source']['drawing_lines'][1]['bbox'][2]=x;q['source']['drawing_lines'][2]['bbox'][0]=q['source']['drawing_lines'][2]['bbox'][2]=150.;add('track-gap-'+str(x),q)
    for y in [214.99999,215.,215.00001]:q=base();q['source']['drawing_lines'][-2]['bbox']=[50.,y,250.,y];add('top-border-'+str(y),q)
    for y in [169.99999,170.,170.00001]:q=base();q['candidates'][0]['boundaries'][0]=[50.,y,250.,y];add('first-gap-'+str(y),q)
    for x in [179.99999,180.,180.00001]:q=base();q['candidates'][1]['bbox'][2]=50.+x;add('grid-width-'+str(x),q)
    q=base();q['candidates'][0]['member_source_indices']=[0]
    for l in q['source']['lines']:l['source_index']=0
    add('duplicate-source-id',q)
    q=base();q['candidates'][0]['member_source_indices']=[-i-1 for i in range(3)]
    for i,l in enumerate(q['source']['lines']):l['source_index']=-i-1
    add('signed-members',q)
    q=base();q['candidates']=[q['candidates'][0],copy.deepcopy(q['candidates'][0]),q['candidates'][1]];add('duplicate-drafts',q)
    rng=random.Random(261116)
    for i in range(100):
        q=base();rng.shuffle(q['source']['lines']);rng.shuffle(q['source']['drawing_lines']);rng.shuffle(q['candidates']);r=rng.choice(q['source']['drawing_lines']);r['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-5.,5.,5.001]);add('geometry-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Explicit draft projection of context angle/em, caption presence, boundary boxes and row fragment line_index membership; no draft generation/materialization.',cases=cases));print('Frozen',len(cases),'old prose conflict cases;',sum(bool(c['expected']['rejected_indices'])for c in cases),'rejections')
if __name__=='__main__':main()
