"""Freeze full captioned blank-answer and page-clipped form candidate producers."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_tiled_tables import old_source
from export_native_bar_graphics import objects
from export_supplemental_rule_tables import line,base as supplemental_base,reference as supplemental_reference
from export_closed_rule_pipeline import dumped
def base(kind):
    s=dict(page_size=[400.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[])
    if kind=='blank_answer':
        s['drawing_lines']=[dict(bbox=[40.,y,340.,y],width=0.,orientation='horizontal')for y in [100.,220.]]
        header=line('Name    Answer',[50.,105.,260.,115.],0,font_signature=['Arial-Bold',0]);header['chars']=[dict(char='a',bbox=[x,105.,x+5.,115.])for x in [50.,57.,64.,100.,107.,114.]];s['lines']=[header]+[line('item',[50.,125.+20*i,90.,135.+20*i],i+1)for i in range(5)]+[line('Table 1: Data',[40.,75.,200.,85.],10)]
    else:
        s['drawing_lines']=[dict(bbox=[40.,y,400.,y],width=0.,orientation='horizontal')for y in [100.,130.,150.,170.,190.,210.,230.]]+[dict(bbox=[x,100.,x,230.],width=0.,orientation='vertical')for x in [40.,140.,240.]]
        s['lines']=[line('Header',[60.+200*j,103.+12*i,110.+200*j,112.+12*i],i*2+j)for i in range(2)for j in range(2)]+[line('item',[60.,135.+20*i,100.,145.+20*i],4+i)for i in range(5)]
    return dict(kind=kind,source=s,excluded_bboxes=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_captioned_blank_answer_tables,_detect_page_clipped_blank_form_tables
    source=old_source(q['source']);source.lines=objects(q['source']['lines']);source.image_bboxes=[]
    f=_detect_captioned_blank_answer_tables if q['kind']=='blank_answer'else _detect_page_clipped_blank_form_tables
    return clean(dict(candidates=[dumped(c)for c in f(source,q.get('excluded_bboxes',[]))]))
def main():
    target=ROOT/'build/migration/native-form-table-pipeline.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q,call=reference):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=call(q)))
    for kind in ['blank_answer','page_clipped']:
        add(kind,base(kind))
        for count in range(0,10):q=base(kind);q['source']['lines']=q['source']['lines'][:count];add(kind+'-count-'+str(count),q)
        for ex in [[],[[0.,0.,400.,600.]],[[40.,100.,340.,220.]]]:q=base(kind);q['excluded_bboxes']=ex;add(kind+'-exclude-'+str(ex),q)
        for em in [5.,9.99999,10.,10.00001,20.]:
            q=base(kind)
            for l in q['source']['lines']:l['effective_height']=em
            add(kind+'-scale-'+str(em),q)
        for angle in [90,180,270]:
            q=base(kind)
            for l in q['source']['lines']:l['angle']=angle
            add(kind+'-angle-'+str(angle),q)
    for mode in ['not-bold','no-signature','no-chars','two-gaps','terminal','wide-key','one-word','unicode-digits','duplicate-rule','no-left','extra-vertical','misaligned-header']:
        q=base('blank_answer'if mode not in ['no-left','extra-vertical','misaligned-header']else'page_clipped');s=q['source']
        if mode=='not-bold':s['lines'][0]['font_signature']=['Arial',0]
        if mode=='no-signature':s['lines'][0]['font_signature']=None
        if mode=='no-chars':s['lines'][0]['chars']=[]
        if mode=='two-gaps':s['lines'][0]['chars']=[dict(char='a',bbox=[x,105.,x+5.,115.])for x in [50.,57.,100.,107.,160.,167.]]
        if mode=='terminal':s['lines'][1]['paragraph_terminal']=True
        if mode=='wide-key':s['lines'][1]['bbox'][2]=200.
        if mode=='one-word':s['lines'][0]['chars']=[dict(char='a',bbox=[50.,105.,60.,115.])]
        if mode=='unicode-digits':s['lines'][-1]['text']='Tab. ١. Data'
        if mode=='duplicate-rule':s['drawing_lines'].insert(0,copy.deepcopy(s['drawing_lines'][0]))
        if mode=='no-left':s['drawing_lines']=s['drawing_lines'][:-3]+s['drawing_lines'][-2:]
        if mode=='extra-vertical':s['drawing_lines'].append(dict(bbox=[100.,100.,100.,230.],width=0.,orientation='vertical'))
        if mode=='misaligned-header':s['lines'][1]['bbox'][1]+=3.
        add(mode,q)
    def ints(x):
        if isinstance(x,list):return[ints(v)for v in x]
        if isinstance(x,dict):return{k:ints(v)if k in ['bbox','page_size','rectangle_bboxes','source','lines','drawing_lines','chars','path_infos']else v for k,v in x.items()}
        return int(x)if isinstance(x,float)and x.is_integer()else x
    for kind in ['blank_answer','page_clipped']:add(kind+'-integer-coordinates',ints(base(kind)))
    for kind in ['open_vertical','short_caption']:
        add(kind+'-integer-coordinates',ints(supplemental_base(kind)),supplemental_reference)
        q=supplemental_base(kind);q['source']['drawing_lines'][0]['orientation']='vertical';add(kind+'-source-orientation',q,supplemental_reference)
    rng=random.Random(271030)
    for i in range(55):
        q=base(rng.choice(['blank_answer','page_clipped']));s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines']);l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'form/boundary facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
