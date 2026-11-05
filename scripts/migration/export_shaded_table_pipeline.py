"""Freeze full shaded-header producer, including positive native clipping stage grids."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects
from export_native_tiled_tables import old_source
from export_closed_rule_pipeline import dumped
from export_supplemental_rule_tables import line
def path(b,fill=True):return dict(bbox=b,rectangle_bboxes=[b],fill_visible=fill,stroke_visible=False,fill_rgba=[200,200,200,255]if fill else None,segment_count=5,form_depth=0,source_index=0)
def base(mode='numeric'):
    s=dict(page_size=[400.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[])
    for j in range(3):
        s['lines'].append(line('Header',[50.+100*j,100.,85.+100*j,110.],j));s['path_infos'].append(path([50.+100*j,98.,150.+100*j,113.]))
    count=9 if mode=='stage'else 4
    for i in range(count):
        for j in range(3):
            if mode=='stage'and j==0 and i%3:continue
            text=str(i*3+j)if mode=='numeric'else('Detail'if j else f'{i//3+1}. Prepare')
            s['lines'].append(line(text,[50.+100*j,125.+25*i,85.+100*j,135.+25*i],3+i*3+j))
    if mode!='numeric':s['drawing_lines']=[dict(bbox=[50.,y,350.,y],width=0.,orientation='horizontal')for y in [100.,150.,200.]]
    if mode=='stage':
        s['image_bboxes']=[[50.,95.,350.,400.]]
        for l in s['lines'][:3]:l['effective_height']=12.
        for j in range(3):
            s['path_infos'].append(path([50.+100*j,98.,110.+100*j,115.],False))
            for i in range(9):s['path_infos'].append(path([50.+100*j,121.+25*i,110.+100*j,141.+25*i],False))
    return dict(kind='shaded_header',source=s,excluded_bboxes=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_shaded_header_tables
    source=old_source(q['source']);source.lines=objects(q['source']['lines']);source.image_bboxes=q['source'].get('image_bboxes',[])
    return clean(dict(candidates=[dumped(c)for c in _detect_shaded_header_tables(source,q.get('excluded_bboxes',[]))]))
def main():
    target=ROOT/'build/migration/native-shaded-table-pipeline.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['numeric','multiline','stage']:
        add(mode,base(mode))
        for count in [0,1,2,3,4]:
            q=base(mode);q['source']['lines']=q['source']['lines'][:3+count*3];add(mode+'-count-'+str(count),q)
        for color in [179,180,200,255]:
            q=base(mode)
            for p in q['source']['path_infos']:
                if p['fill_visible']:p['fill_rgba']=[color,color,color,255]
            add(mode+'-color-'+str(color),q)
        for semantic in [None,'header','footer','text','']:
            q=base(mode)
            for l in q['source']['lines']:l['semantic_type']=semantic
            add(mode+'-semantic-'+str(semantic),q)
        for ex in [[],[[0.,0.,400.,600.]],[[40.,95.,355.,117.]]]:
            q=base(mode);q['excluded_bboxes']=ex;add(mode+'-excluded-'+str(ex),q)
    for mode in ['stage-missing-clips','stage-width','stage-number','stage-all-stages','stage-gap','stage-overlay']:
        q=base('stage');s=q['source']
        if mode=='stage-missing-clips':s['path_infos']=s['path_infos'][:8]
        if mode=='stage-width':
            for p in s['path_infos'][3:]:p['bbox'][2]=p['bbox'][0]+10.
        if mode=='stage-number':s['lines'][3]['text']='Unnumbered'
        if mode=='stage-all-stages':
            for i in range(9):
                if i%3:s['lines'].append(line(str(i+1)+'. Prepare',[50.,125.+25*i,85.,135.+25*i],100+i))
        if mode=='stage-gap':s['lines'][3]['bbox'][1]+=30.
        if mode=='stage-overlay':
            for p in s['path_infos'][3:]:p['bbox'][3]+=30.
        add(mode,q)
    rng=random.Random(271029)
    for i in range(60):
        q=base(rng.choice(['numeric','multiline','stage']));s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['path_infos']);l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'shaded facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',sum(any(c['inferred_grid_authoritative']for c in x['expected']['candidates'])for x in cases),'stage grid positive')
if __name__=='__main__':main()
