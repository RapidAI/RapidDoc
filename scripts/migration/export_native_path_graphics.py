"""Freeze old complex-path/container/axis candidates and actual PDF vector reads."""
import copy,json,math,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-path-graphics.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _is_two_dimensional_complex_path as complex_path,_detect_complex_path_containers as containers,_detect_axis_path_graphics as axes
    def p(b,index=0,segments=5,depth=0,fill=False,stroke=False):return dict(bbox=b,source_index=index,segment_count=segments,form_depth=depth,fill_visible=fill,stroke_visible=stroke)
    def container():return [p([20.,20.,200.,200.],0,fill=True),p([40.,40.,80.,80.],1,segments=6)]+[p([50.+i*20.,100.,60.+i*20.,130.],i+2)for i in range(3)]
    def axis():return [p([50.,180.,200.,180.],0,stroke=True),p([50.,50.,50.,180.],1,stroke=True),p([60.,70.,180.,160.],2,segments=6,stroke=True)]
    cases=[]
    def add(label,paths,size=(400.,600.),em=10.,pdf=None):
        request=clean(dict(path_infos=copy.deepcopy(paths),page_size=size,median_height=em))
        old=[PDFPathInfo(**dict(x,bbox=tuple(x['bbox'])))for x in copy.deepcopy(paths)]
        expected=clean(dict(complex_path_indices=[i for i,x in enumerate(old)if complex_path(x,em)],container_bboxes=containers(old,size,em),axis_bboxes=axes(old,size,em)))
        c=dict(label=label,request=request,expected=expected)
        if pdf:c['pdf_ref']=pdf
        cases.append(c)
    add('empty',[]);add('container',container());add('axis',axis());add('both',container()+axis())
    for depth in [-1,0,1,5]:
        for segments in [0,5,6,8]:
            for width in [14.99,15.,15.01]:add(f'complex-{depth}-{segments}-{width}',[p([10.,10.,10.+width,25.],segments=segments,depth=depth)])
    for count in [3,4,5]:add('inner-count-'+str(count),container()[:1+count])
    for field,values in [('form_depth',[-1,0,1]),('fill_visible',[False,True]),('source_index',[0,1,2])]:
        for value in values:
            paths=container();paths[0][field]=value;add(f'container-{field}-{value}',paths)
    for index in [1,2]:
        for frac in [.89999,.9,.90001]:
            paths=container();paths[index]['bbox']=[20.-(1-frac)*100.,30.,20.+frac*100.,130.];add(f'inner-overlap-{index}-{frac}',paths)
    for ratio in [.0049999,.005,.0050001,.4999999,.5,.5000001]:
        paths=container();add('page-area-'+str(ratio),paths,size=(400.,32400./ratio/400.))
    for width in [39.99,40.,40.01]:
        paths=container();paths[0]['bbox'][2]=20.+width;add('container-width-'+str(width),paths)
    for delta in [-10.000001,-10.,-9.999999,0.,9.999999,10.,10.000001]:
        for endpoint in [50.,200.]:
            paths=axis();paths[1]['bbox'][0]=paths[1]['bbox'][2]=endpoint+delta;add(f'axis-end-{endpoint}-{delta}',paths)
    for top in [40.,170.,180.,190.,190.000001]:
        paths=axis();paths[1]['bbox'][3]=top;add('axis-y-'+str(top),paths)
    for width in [59.999,60.,60.001]:
        paths=axis();paths[0]['bbox'][2]=50.+width;add('axis-length-'+str(width),paths)
    for thickness in [4.9999,5.,5.0001]:
        paths=axis();paths[0]['bbox'][3]+=thickness;add('axis-thin-'+str(thickness),paths)
    for size in [(230.,600.),(230.76923076923077,600.),(400.,259.999),(400.,260.),(400.,260.001),(0.,600.),(-400.,600.)]:add('axis-size-'+str(size),axis(),size=size)
    for visibility in [False,True]:
        paths=axis();paths[-1]['fill_visible']=visibility;paths[-1]['stroke_visible']=visibility;add('complex-invisible-'+str(visibility),paths)
    for order in [[0,1,2],[1,0,2],[2,1,0]]:add('order-'+str(order),[axis()[i]for i in order])
    paths=axis();paths.insert(2,copy.deepcopy(paths[1]));add('duplicate-axes',paths)
    paths=container();paths[2]['source_index']=paths[0]['source_index'];add('duplicate-source-id',paths)
    paths=axis();paths.append(p([45,45,210,190],3,segments=10));add('union-int-float',paths)
    paths=axis();paths[0]['bbox']=[50,180,200,180];paths[1]['bbox']=[50.0,50,50,180.0];add('ties-types',paths)
    for delta in [math.nextafter(10.,-math.inf),10.,math.nextafter(10.,math.inf)]:
        paths=axis();paths[1]['bbox'][0]=paths[1]['bbox'][2]=50.+delta;add('ulp-'+str(delta),paths)
    rng=random.Random(261101)
    for i in range(100):
        paths=container()if i%2 else axis();path=rng.choice(paths);path['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-15.,15.]);add('geometry-'+str(i),paths)
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    vectors=json.loads((ROOT/'build/migration/pdf-vector-primitives.json').read_text(encoding='utf-8'));seen=set()
    for c in vectors['cases']:
        key=(c['pdf'],c['page_index'])
        if key in seen:continue
        seen.add(key);data=(ROOT/c['pdf']).read_bytes()
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(c['page_index'])
            add('pdf-'+str(len(seen)),[asdict(x)for x in snap.path_infos],snap.page_size,pdf=dict(pdf=c['pdf'],pdf_sha256=digest(data),page_index=c['page_index']))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_contracts.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'old path facts;',sum(bool(c['expected']['container_bboxes'])for c in cases),'container-positive;',sum(bool(c['expected']['axis_bboxes'])for c in cases),'axis-positive;',sum('pdf_ref'in c for c in cases),'actual PDF inputs')
if __name__=='__main__':main()
