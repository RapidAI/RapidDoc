"""Freeze full open vertical and short caption producers with automatic caption annotations."""
import copy,json,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects
from export_native_tiled_tables import old_source
from export_closed_rule_pipeline import dumped

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_open_vertical_tables,_detect_short_caption_ruled_tables
    s=old_source(q['source']);s.lines=objects(q['source'].get('lines',[]));s.image_bboxes=q['source'].get('image_bboxes',[])
    f=_detect_open_vertical_tables if q['kind']=='open_vertical'else _detect_short_caption_ruled_tables
    return dict(candidates=[dumped(c)for c in f(s,q.get('excluded_bboxes',[]))])
def line(text,b,i,**kw):return dict(text=text,bbox=b,source_index=i,angle=0,effective_height=10.,chars=[],**kw)
def base(kind):
    if kind=='open_vertical':
        rules=[dict(bbox=[40.,y,340.,y],width=0.,orientation='horizontal')for y in [100.,140.,180.,220.,260.]]+[dict(bbox=[x,100.,x,260.],width=0.,orientation='vertical')for x in [140.,240.]]
        lines=[line(str(i+1)if j==0 else'cell',[50.+100.*j,105.+40.*i,80.+100.*j,115.+40.*i],i*3+j)for i in range(4)for j in range(3)]
    else:
        rules=[dict(bbox=[40.,y,340.,y],width=0.,orientation='horizontal')for y in [100.,130.,220.]]
        lines=[line('item',[50.,105.+20.*i,110.,115.+20.*i],i)for i in range(6)]+[line('Table 1: Data',[40.,225.,180.,235.],20)]
    return dict(kind=kind,source=dict(page_size=[400.,600.],lines=lines,drawing_lines=rules,path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[]),excluded_bboxes=[])
def main():
    target=ROOT/'build/migration/native-supplemental-rule-tables.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=clean(reference(q))))
    for kind in ['open_vertical','short_caption']:
        add(kind,base(kind))
        for marker in ['Table 1: Data','Table: Data','Tab. ١. Data','TABLE IX: Data','Table1中文','表 1：数据','figure 1','Table 1 Data','Table 1.2 Results']:
            q=base(kind);q['source']['lines'].append(line(marker,[60.,80.,220.,90.],30));add(kind+'-caption-'+marker,q)
        for count in [0,1,2,3,4,5,6]:
            q=base(kind);q['source']['lines']=q['source']['lines'][:count*3 if kind=='open_vertical'else count];add(kind+'-rows-'+str(count),q)
        for exclusion in [[0.,0.,400.,600.],[50.,105.,100.,115.],[40.,100.,340.,220.]]:
            q=base(kind);q['excluded_bboxes']=[exclusion];add(kind+'-excluded-'+str(exclusion),q)
        for angle in [90,180,270]:
            q=base(kind)
            for l in q['source']['lines']:l['angle']=angle
            add(kind+'-angle-'+str(angle),q)
        for em in [5.,9.99999,10.,10.00001,20.]:
            q=base(kind)
            for l in q['source']['lines']:l['effective_height']=em
            add(kind+'-scale-'+str(em),q)
    for mode in ['header','crossing','closed','spread','short-grid','long-prose','fills','bold-caption','caption-continuation','bold-name']:
        q=base('open_vertical');s=q['source']
        if mode in ['header','short-grid']:s['lines']+=[line('head',[50.+100.*j,82.,90.+100.*j,92.],30+j)for j in range(3)]
        if mode=='short-grid':s['drawing_lines'].append(dict(bbox=[90.,100.,90.,260.],width=0.,orientation='vertical'));s['lines']=[l for l in s['lines']if l['source_index']>=30 or l['bbox'][1]<160.]
        if mode=='crossing':s['lines'][0]['bbox'][2]=250.
        if mode=='closed':s['drawing_lines']+=[dict(bbox=[x,100.,x,260.],width=0.,orientation='vertical')for x in [40.,340.]]
        if mode=='spread':s['drawing_lines'][-1]['bbox'][3]+=15.00001
        if mode=='long-prose':s['lines'][0]['text']='These ordinary words form a separate long prose sentence.';s['lines'][0]['bbox'][2]=330.
        if mode=='fills':
            for y in [120.,160.]:
                b=[40.,y,340.,y+20.];s['path_infos'].append(dict(bbox=b,rectangle_bboxes=[b],fill_visible=True,stroke_visible=False,fill_rgba=[200,200,200,255],segment_count=5,form_depth=0,source_index=0))
        if mode in ['bold-caption','bold-name']:s['lines'].append(line('Results',[140.,87.,240.,97.],30,dominant_font_weight=700 if mode=='bold-caption'else 400,font_signature=['Arial-Bold'if mode=='bold-name'else'Arial',0]))
        if mode=='caption-continuation':s['lines']+=[line('Table 1: Data',[40.,70.,180.,80.],30),line('continued title',[40.,82.,180.,92.],31)]
        add(mode,q)
    rng=random.Random(271028)
    for i in range(80):
        kind=rng.choice(['open_vertical','short_caption']);q=base(kind);s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines'])
        l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'supplemental facts;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()

