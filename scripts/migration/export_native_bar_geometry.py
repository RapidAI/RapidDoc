"""Freeze original bar-admission AST, compound flags and zero-axis extension rules."""
import ast,copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def reference():
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as old
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_detect_native_bar_graphics')
    end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='outputs'for t in n.targets))
    body=[copy.deepcopy(n)for n in fn.body[:end]if not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='table_bounds'for t in n.targets))]
    loop=next(n for n in fn.body if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='horizontal')
    wrapper=ast.parse('def _prepare(source,em,table_bounds):\n pass').body[0];wrapper.body=body+ast.parse('directions=[]').body
    mini=ast.parse('for horizontal in (True,False):\n pass').body[0];mini.body=[copy.deepcopy(loop.body[0])]+ast.parse('directions.append(bars)').body
    wrapper.body+=[mini]+ast.parse('return paths,directions,baseline_flags').body;namespace=dict(old.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(path),'exec'),namespace)
    return namespace['_prepare'],old._bar_zero_axis_bboxes,ast.dump(wrapper,include_attributes=False)
def main():
    target=ROOT/'build/migration/native-bar-geometry.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    prepare,axes,tree=reference()
    def p(b=(50.,50.,150.,60.),rects=(),fill=True,color=(0,100,200,255),segments=5):return dict(bbox=b,rectangle_bboxes=rects,fill_visible=fill,fill_rgba=color,stroke_visible=False,segment_count=segments,form_depth=0,source_index=0)
    def rects():return [[50.,50.+i*20.,100.+i*20.,60.+i*20.]for i in range(4)]
    def q(horizontal=True):return dict(bounds=[50.,50.,200.,130.],horizontal=horizontal,baseline=50. if horizontal else 130.)
    def rules():return [dict(bbox=[50.,40.,50.,140.],orientation='vertical',width=2.),dict(bbox=[40.,130.,210.,130.],orientation='horizontal',width=2.)]
    cases=[]
    def add(label,paths,em=10.,tables=(),queries=(),drawing=()):
        request=clean(dict(path_infos=copy.deepcopy(paths),em=em,table_bboxes=tables,axis_queries=queries,drawing_lines=drawing));ps=[PDFPathInfo(**dict(x,bbox=tuple(x['bbox']),rectangle_bboxes=tuple(tuple(b)for b in x.get('rectangle_bboxes',[])),fill_rgba=tuple(x['fill_rgba'])if x.get('fill_rgba')is not None else None))for x in paths];rs=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in drawing];flags=[];admitted=[]
        for i,path in enumerate(ps):
            s=SimpleNamespace(path_infos=[path],drawing_lines=rs,compound_baseline_cache={});accepted,directions,f=prepare(s,em,[tuple(b)for b in tables]);flags.append(f.get(id(path.rectangle_bboxes)))
            for bar in accepted:
                ri=next((j for j,b in enumerate(path.rectangle_bboxes)if b is bar.bbox),None)if len(path.rectangle_bboxes)>1 else None
                admitted.append(dict(path_index=i,rectangle_index=ri,bbox=bar.bbox,horizontal=any(bar is b for b in directions[0]),vertical=any(bar is b for b in directions[1])))
        expected=clean(dict(compound_baselines=flags,admitted_bars=admitted,zero_axis_bboxes=[axes(SimpleNamespace(drawing_lines=rs),tuple(x['bounds']),x['horizontal'],x['baseline'],em)for x in queries]));cases.append(dict(label=label,request=request,expected=expected))
    add('empty',[]);add('single',[p()],queries=[q(),q(False)],drawing=rules());add('compound',[p(rects=rects())]);add('singleton-rectangle-uses-path-bbox',[p(rects=[[0.,0.,1.,1.]])])
    for count in [1,2,3,4,5]:
        for shift in [3.49999,3.5,3.50001,50.]:
            rs=rects()+[[50.,130.,180.,140.]];rs=rs[:count]
            if rs:rs[0][0]+=shift
            add(f'flags-{count}-{shift}',[p(rects=rs)])
    for n in [1,2]:
        rs=rects()
        for b in rs[:n]:b[0]+=10.
        add('support-75-'+str(n),[p(rects=rs)])
    for seg in [3,4,5,24,25]:
        for compound in [False,True]:add(f'seg-{seg}-{compound}',[p(rects=rects()if compound else[],segments=seg)])
    for fill in [False,True]:
        for color in [None,(230,230,230,255),(229,240,240,255),(0,100,200,0)]:add('color-'+str(fill)+str(color),[p(fill=fill,color=color,rects=rects())])
    for thickness in [1.9999,2.,2.0001,50.,50.0001]:add('thickness-'+str(thickness),[p(b=[50.,50.,150.,50.+thickness])])
    for length in [19.999,20.,20.001]:add('minimum-length-'+str(length),[p(b=[50.,50.,50.+length,60.])])
    for width in [12.499,12.5,12.501]:add('aspect-'+str(width),[p(b=[50.,50.,50.+width,60.])])
    for box in [[50.,50.,145.,60.],[50.,50.,144.999,60.],[0.,0.,400.,600.],[200.,200.,250.,250.]]:add('table-overlap-'+str(box),[p()],tables=[box])
    for em in [0.,1.,10.,20.,-1.]:add('em-'+str(em),[p(rects=rects())],em,queries=[q(),q(False)],drawing=rules())
    for horizontal in [True,False]:
        for width in [-2.,0.,1.,2.,10.]:
            rs=rules()
            for r in rs:r['width']=width
            add(f'padding-{horizontal}-{width}',[],queries=[q(horizontal)],drawing=rs)
        for shift in [3.4999,3.5,3.5001]:
            query=q(horizontal);query['baseline']+=shift;add(f'baseline-{horizontal}-{shift}',[],queries=[query],drawing=rules())
        for frac in [.79999,.8,.80001,2.,2.00001]:
            rs=rules();r=rs[0]if horizontal else rs[1];axis=1 if horizontal else 0;lo=50.;length=80. if horizontal else 150.;r['bbox'][axis]=lo;r['bbox'][axis+2]=lo+length*frac;add(f'axis-length-{horizontal}-{frac}',[],queries=[q(horizontal)],drawing=rs)
    rng=random.Random(261107)
    for i in range(80):
        rs=rects();rs[rng.randrange(4)][rng.randrange(4)]+=rng.choice([-.001,.001,-4.,4.]);add('geometry-'+str(i),[p(rects=rs)],queries=[q(),q(False)],drawing=rules())
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_graphic_geometry.py']
    write_json(target,dict(sources={x:digest((ROOT/x).read_bytes())for x in refs},admission_ast=tree,cases=cases));print('Frozen',len(cases),'old bar geometry cases;',sum(bool(c['expected']['admitted_bars'])for c in cases),'admission-positive')
if __name__=='__main__':main()
