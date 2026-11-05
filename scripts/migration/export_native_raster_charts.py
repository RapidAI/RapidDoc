"""Freeze old raster-axis detection and complete chart-description grouping side effects."""
import copy,json,random,sys
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    edges='--edges'in sys.argv
    target=ROOT/('build/migration/native-raster-charts-edges.json'if edges else'build/migration/native-raster-charts.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _raster_axis_tick_groups as groups,_detect_native_raster_axis_graphics as detect,group_native_raster_chart_descriptions as group
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    import re
    def line(text,b,h=10.,**kw):return dict(text=text,bbox=b,effective_height=h,angle=0,source_index=0,font_signature=['Body',0],**kw)
    def base(vertical=True):
        ticks=[line(str(i*10),[20.,100.+i*20.,40.,110.+i*20.]if vertical else[60.+i*30.,200.,80.+i*30.,210.])for i in range(5)]
        return [line('A long description above the chart',[20.,68.,200.,78.]),line('short continuation',[20.,82.,100.,92.]),*ticks]
    def images(vertical=True):return [[60.,100.,200.,190.]]if vertical else[[60.,100.,210.,190.]]
    cases=[]
    def add(label,rows,boxes=()):
        rows=copy.deepcopy(rows);request=clean(dict(source=dict(lines=rows,image_bboxes=boxes,chars=[],metadata={'keep':True})))
        old=[]
        for r in copy.deepcopy(rows):
            if r.get('font_signature')is not None:r['font_signature']=tuple(r['font_signature'])
            old.append(_LineItem(**r))
        numeric=[l for l in old if l.angle==0 and l.semantic_type is None and l.effective_height>0 and re.fullmatch(r'[+−-]?\d+(?:\.\d+)?',l.text.strip())and len(l.text.strip())<=8 and l.bbox[2]-l.bbox[0]<=4*l.effective_height]
        positions={id(l):i for i,l in enumerate(old)};axis=[dict(vertical=v,line_indices=[positions[id(l)]for l in ls])for v,ls,_ in groups(numeric)]
        source=SimpleNamespace(lines=old,image_bboxes=boxes);cores=detect(source);group(source)
        result=copy.deepcopy(request['source'])
        for i,l in enumerate(old):
            if l.paragraph_group!=rows[i].get('paragraph_group'):result['lines'][i]['paragraph_group']=l.paragraph_group
        cases.append(dict(label=label,request=request,expected=clean(dict(source=result,graphic_bboxes=cores,axis_groups=axis))))
    add('empty',[])
    for vertical in [True,False]:
        add('positive-'+str(vertical),base(vertical),images(vertical))
        add('no-image-'+str(vertical),base(vertical))
        for count in [0,4,5,6]:
            rows=base(vertical);rows=rows[:2]+[line(str(i*10),[20.,100.+i*20.,40.,110.+i*20.]if vertical else[60.+i*30.,200.,80.+i*30.,210.])for i in range(count)];add(f'ticks-{vertical}-{count}',rows,images(vertical))
        for value in ['1','1.','1e2','.1','1,000','１２','+12','−12','-12','١٢',' 12　','123456789','1.234567']:
            rows=base(vertical);rows[-1]['text']=value;add(f'grammar-{vertical}-{value}',rows,images(vertical))
        for vals in [[40,30,20,10,0],[0,0,0,0,0],[0,10,20,30,40.09],[0,10,20,30,40.11],[0,1e-9,2e-9,3e-9,4e-9]]:
            rows=base(vertical)
            for r,value in zip(rows[2:],vals):r['text']=str(value)
            add('progression-'+str(vertical)+str(vals),rows,images(vertical))
        for row_index,field,values in [(0,'paragraph_terminal',[False,True]),(0,'caption_start',[False,True]),(1,'caption_start',[False,True]),(0,'semantic_type',[None,'text','title']),(1,'font_signature',[None,[],['Other',0],['Body',False]]),(1,'angle',[0,90,180,270]),(1,'paragraph_group',[None,-5,0,7]),(1,'effective_height',[8.99,9.,10.,11.,11.01]),(3,'effective_height',[0.,7.99,8.,12.,12.01])]:
            for value in values:
                rows=base(vertical);rows[row_index][field]=value;add(f'field-{vertical}-{row_index}-{field}-{value}',rows,images(vertical))
        for text in ['• item','▪ item','1. item','１２) item','1、item',' 2)　body','- item','1 item','plain']:
            rows=base(vertical);rows[1]['text']=text;add(f'list-{vertical}-{text}',rows,images(vertical))
        for value in [99.99,100.,100.01,108.,108.01]:
            rows=base(vertical);rows[0]['bbox'][2]=20.+value;add(f'width-{vertical}-{value}',rows,images(vertical))
        for value in [7.49,7.5,7.51,-.01,0.]:
            rows=base(vertical);rows[1]['bbox']=[20.,78.+value,100.,88.+value];add(f'gap-{vertical}-{value}',rows,images(vertical))
        for value in [40.,40.001]:
            rows=base(vertical);rows[2]['bbox'][2]=rows[2]['bbox'][0]+value;add(f'tick-width-{vertical}-{value}',rows,images(vertical))
        for value in [-5,0,9]:
            rows=base(vertical)+[line('existing',[300.,300.,350.,310.],paragraph_group=value)];add(f'group-base-{vertical}-{value}',rows,images(vertical))
    rows=[line('long first',[20.,88.,220.,98.]),line('middle',[20.,102.,130.,112.]),line('tail',[20.,116.,80.,126.])]+[line(str(i*10),[20.,130.+i*20.,40.,140.+i*20.])for i in range(5)]
    add('adjacent-overwrite',rows,[[60.,130.,200.,220.]])
    add('duplicate-cores',rows,[[60.,130.,200.,220.],[60.,130.,200.,220.]])
    add('equal-area-first',base(),images()+[[65.,100.,205.,190.]])
    for box in [[215.,120.,250.,180.],[215.,120.,315.,180.],[215.,120.,315.01,180.],[215.,120.,250.,180.01]]:add('legend-'+str(box),base(False),images(False)+[box])
    for field in ['caption_start','semantic_type','effective_height']:
        rows=base()+[line('label',[60.,90.,150.,100.])];rows[-1][field]=True if field=='caption_start' else 'text' if field=='semantic_type' else 18.01;add('label-'+field,rows,images())
    rng=random.Random(261030)
    for i in range(100):
        vertical=i%2==0;rows=base(vertical);r=rng.choice(rows);r['bbox'][rng.randrange(4)]+=rng.choice([-.01,.01,-4.,4.]);add('geometry-'+str(i),rows,images(vertical))
    if edges:
        cases=[]
        rows=[line(str(i*10),[20_000_000,100_000_000+i*200_000_000,40_000_000,110_000_000+i*200_000_000],10_000_000)for i in range(5)]
        smaller=[60_000_000,100_000_000,859_999_999,900_000_002]
        larger=[60_000_000,100_000_000,860_000_000,900_000_001]
        add('exact-large-integer-area',rows,[smaller,larger])
        add('mixed-int-float-area',rows,[smaller,list(map(float,larger))])
        add('mixed-float-int-area',rows,[list(map(float,larger)),smaller])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'raster chart cases;',sum(bool(c['expected']['graphic_bboxes'])for c in cases),'core positive;',sum(c['request']['source']['lines']!=c['expected']['source']['lines']for c in cases),'group positive')
if __name__=='__main__':main()
