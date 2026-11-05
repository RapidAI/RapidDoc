"""Freeze full old ordinary horizontal-rule producer, including deferred materialization parity."""
import copy,json,random
from common import ROOT,digest,write_json
from export_closed_rule_pipeline import setup,dumped,base
from export_native_inline_clusters import clean
from export_native_tiled_tables import old_source

def reference(q):
    old,source,size,angle,lines,rows,axis,em,excluded=setup(q)
    paths=old_source(dict(source,chars=[])).path_infos
    kwargs=dict(path_infos=paths,excluded_bboxes=excluded)
    plain=old._build_rule_table_candidates(rows,lines,size,angle,em,axis,**kwargs)
    deferred=old._build_rule_table_candidates(rows,lines,size,angle,em,axis,defer_materialization=True,**kwargs)
    result=clean(dict(candidates=[dumped(c)for c in plain]));lazy=clean(dict(candidates=[dumped(c.materialize())for c in deferred]))
    assert result==lazy,'old ordinary/deferred materialization differ'
    return result

def main():
    target=ROOT/'build/migration/native-rule-interval-pipeline.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):
        q=clean(copy.deepcopy(q));q.pop('candidates',None);cases.append(dict(label=label,request=q,expected=reference(q)))
    prior=json.loads((ROOT/'build/migration/native-rule-page.json').read_text(encoding='utf-8'))
    for c in prior['cases']:add('prior-'+c['label'],c['request'])
    for height in [15.,25.,40.,60.,60.00001,80.]:
        for cols in [1,2,3,4]:
            q=base();s=q['source'];s['drawing_lines']=[dict(bbox=[40.,100.+i*height,340.,100.+i*height],orientation='horizontal',width=0.)for i in range(5)];s['lines']=[dict(text='value',bbox=[45.+j*75.,105.+i*height,70.+j*75.,115.+i*height],source_index=i*cols+j,angle=0,chars=[],effective_height=10.)for i in range(4)for j in range(cols)];add(f'rows-{height}-{cols}',q)
    for width in [40.,100.,145.,150.]:
        q=base();s=q['source'];s['drawing_lines']=s['drawing_lines'][:4];s['lines']=s['lines'][:6]
        for l in s['lines']:l['bbox'][2]=l['bbox'][0]+width
        add('prose-width-'+str(width),q)
    for y in [100.,110.,125.,139.99999,140.,140.00001,180.,220.]:
        q=base();q['source']['lines'][0]['bbox'][1:4:2]=[y-5.,y+5.];add('boundary-center-'+str(y),q)
    for fills in [0,1,2,3,9]:
        q=base();s=q['source'];s['drawing_lines']=s['drawing_lines'][:4];s['path_infos']=[]
        for i in range(fills):
            b=[40.,105.+i*10.,240.,110.+i*10.];s['path_infos'].append(dict(bbox=b,rectangle_bboxes=[b],fill_visible=True,stroke_visible=False,segment_count=5,form_depth=0,source_index=i))
        add('fills-'+str(fills),q)
    rng=random.Random(271027)
    for i in range(60):
        q=base();s=q['source'];s['drawing_lines']=s['drawing_lines'][:rng.choice([2,3,4])];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines'])
        for l in s['lines']:
            if rng.randrange(3)==0:l['bbox'][rng.choice([0,2])]+=rng.choice([-3.,3.,7.5])
        add('perturbed-'+str(i),q)
    refs=prior['sources'];write_json(target,dict(sources=refs,scope='Full old ordinary producer and deferred materialization outputs agree on all frozen inputs; raw page source, no interval/core/caption/score verdict injection.',cases=cases));print('Frozen',len(cases),'facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',sum(len(c['expected']['candidates'])for c in cases),'candidates')
if __name__=='__main__':main()
