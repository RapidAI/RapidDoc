"""Freeze complete partial-grid/year-header/metric-pair producers, not accepted-region projections."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_tiled_tables import old_source
from export_native_bar_graphics import objects
from export_supplemental_rule_tables import line
from export_closed_rule_pipeline import dumped
def base(kind):
    s=dict(page_size=[400.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[])
    if kind=='partial_grid':
        s['drawing_lines']=[dict(bbox=[40.,100.,340.,100.],width=0.,orientation='horizontal')]+[dict(bbox=[140.,y,340.,y],width=0.,orientation='horizontal')for y in [140.,180.]]+[dict(bbox=[x,100.,x,240.],width=0.,orientation='vertical')for x in [40.,140.,240.,340.]]
        s['lines']=[line('cell',[60.+100*j,110.+40*i,100.+100*j,120.+40*i],i*3+j)for i in range(3)for j in range(3)]
    elif kind=='year_header':
        s['drawing_lines']=[dict(bbox=[40.,110.,380.,110.],width=0.,orientation='horizontal')]
        s['lines']=[line('Metric',[50.,95.,120.,105.],0)]+[line(str(2020+j),[200.+60*j,95.,240.+60*j,105.],j+1)for j in range(3)]
        for i in range(3):s['lines']+=[line('label',[50.,115.+20*i,120.,125.+20*i],4+i*4)]+[line(str(10+i),[210.+60*j,115.+20*i,240.+60*j,125.+20*i],5+i*4+j)for j in range(3)]
    else:
        h=line('关键指标',[50.,80.,130.,95.],0);h['effective_height']=15.;s['lines']=[h];s['drawing_lines']=[dict(bbox=[130.,90.,350.,90.],width=0.,orientation='horizontal')]
        for i in range(5):s['lines']+=[line('指标一',[50.,100.+18*i,120.,110.+18*i],1+i*2),line('10%',[250.,100.+18*i,280.,110.+18*i],2+i*2)]
    return dict(kind=kind,source=s,excluded_bboxes=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_year_header_numeric_tables,_detect_ruled_metric_pair_tables
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_open_grid import detect_partial_rule_grids
    source=old_source(q['source']);source.lines=objects(q['source']['lines']);source.form_bboxes=q['source'].get('form_bboxes',[])
    call=dict(partial_grid=detect_partial_rule_grids,year_header=_detect_year_header_numeric_tables,metric_pair=_detect_ruled_metric_pair_tables)[q['kind']]
    return clean(dict(candidates=[dumped(c)for c in call(source,q.get('excluded_bboxes',[]))]))
def main():
    target=ROOT/'build/migration/native-numeric-candidate-segments.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    for kind in ['partial_grid','year_header','metric_pair']:
        q=base(kind);assert reference(q)['candidates'],kind;add(kind,q)
        for count in [0,1,2,3,4,6,8,12]:q=base(kind);q['source']['lines']=q['source']['lines'][:count];add(kind+'-count-'+str(count),q)
        for semantic in [None,'header','footer','text','']:
            q=base(kind)
            for l in q['source']['lines']:l['semantic_type']=semantic
            add(kind+'-semantic-'+str(semantic),q)
        for angle in [90,180,270]:
            q=base(kind)
            for l in q['source']['lines']:l['angle']=angle
            add(kind+'-angle-'+str(angle),q)
        for scale in [5.,9.99999,10.,10.00001,20.]:
            q=base(kind)
            for l in q['source']['lines']:l['effective_height']=scale*(1.5 if kind=='metric_pair'and l['source_index']==0 else 1.)
            add(kind+'-scale-'+str(scale),q)
        for ex in [[],[[0.,0.,400.,600.]],[[40.,100.,340.,240.]]]:q=base(kind);q['excluded_bboxes']=ex;add(kind+'-exclude-'+str(ex),q)
    values=['0','-1','−1','+12','1.2','1,000','(12)','(12','12)','12 %','12‰','١٢','10-20','10 / 20','10kg','10公斤','10abcdef','10abcdefg','1e3','1.','1,','12 $','None','',' 10 ','1\n%','10-','+1/-2']
    for kind in ['year_header','metric_pair']:
        for token in values:
            q=base(kind)
            for l in q['source']['lines']:
                if kind=='year_header'and l['source_index']>=4 and l['source_index']%4!=0 or kind=='metric_pair'and l['source_index']>0 and l['source_index']%2==0:l['text']=token
            add(kind+'-numeric-'+repr(token),q)
    for years in [['2020','2021','2022'],['1000','2000','3000'],['2022','2021','2020'],['2020A','2021B','2022C'],['٢٠٢٠','٢٠٢١','٢٠٢٢'],['2020AA','2021','2022']]:
        q=base('year_header')
        for l,t in zip(q['source']['lines'][1:4],years):l['text']=t
        add('year-tokens-'+repr(years),q)
    for mode in ['missing-track','outer-end-spread','existing-bottom','containing-form','late-track','duplicate-track','integer-coordinates','next-year-header','one-text-value','formula-gate']:
        kind='year_header'if mode=='next-year-header'else'metric_pair'if mode in ['one-text-value','formula-gate']else'partial_grid';q=base(kind);s=q['source']
        if mode=='missing-track':s['drawing_lines']=s['drawing_lines'][:-1]
        if mode=='outer-end-spread':s['drawing_lines'][-1]['bbox'][3]+=2.
        if mode=='existing-bottom':s['drawing_lines'].append(dict(bbox=[40.,240.,340.,240.],width=0.,orientation='horizontal'))
        if mode=='containing-form':s['form_bboxes']=[[0.,0.,400.,600.]]
        if mode=='late-track':s['drawing_lines'][-2]['bbox'][1]=155.
        if mode=='duplicate-track':s['drawing_lines'].append(copy.deepcopy(s['drawing_lines'][-2]))
        if mode=='integer-coordinates':
            for l in s['lines']:l['bbox']=[int(x)for x in l['bbox']]
            for r in s['drawing_lines']:r['bbox']=[int(x)for x in r['bbox']]
        if mode=='next-year-header':
            s['drawing_lines'].append(dict(bbox=[40.,170.,380.,170.],width=0.,orientation='horizontal'))
            for i,l in enumerate(copy.deepcopy(s['lines'][:4])):l['bbox']=[l['bbox'][0],155.,l['bbox'][2],165.];l['source_index']=100+i;s['lines'].append(l)
        if mode=='one-text-value':s['lines'][2]['text']='合格'
        if mode=='formula-gate':s['lines'][0]['formula_candidate_only']=True
        add(mode,q)
    rng=random.Random(271032)
    for i in range(75):
        q=base(rng.choice(['partial_grid','year_header','metric_pair']));s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines']);l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_open_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'facts;',[(k,sum(bool(c['expected']['candidates'])for c in cases if c['request']['kind']==k))for k in ['partial_grid','year_header','metric_pair']])
if __name__=='__main__':main()
