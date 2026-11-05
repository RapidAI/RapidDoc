"""Freeze old drawing-line components and complex drawing graphic decisions."""
import copy,json,math,random,itertools
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-drawing-graphics.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _summarize_drawing_components as summarize,_detect_complex_drawing_components as detect,_is_two_dimensional_complex_path as complex_path
    def line(b,orientation):return dict(bbox=b,orientation=orientation,width=1.)
    def frame():return [line([50.,50.,200.,50.],'horizontal'),line([50.,180.,200.,180.],'horizontal'),line([50.,50.,50.,180.],'vertical'),line([200.,50.,200.,180.],'vertical')]
    def path(b=(60.,70.,180.,160.),segments=6,depth=0):return dict(bbox=b,segment_count=segments,form_depth=depth,source_index=0,fill_visible=False,stroke_visible=True)
    cases=[]
    def add(label,lines,paths=(),size=(400.,600.),em=10.,pdf=None):
        request=clean(dict(drawing_lines=copy.deepcopy(lines),path_infos=copy.deepcopy(paths),page_size=size,median_height=em))
        old=[_AxisLine(tuple(l['bbox']),l.get('width',1.),l['orientation'])for l in lines];ps=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox'])))for p in paths];positions={id(l):i for i,l in enumerate(old)};tolerance=max(2.,.75*em)
        summaries=summarize(old,tolerance);expected=clean(dict(tolerance=tolerance,complex_path_indices=[i for i,p in enumerate(ps)if complex_path(p,em)],components=[dict(line_indices=[positions[id(l)]for l in s.lines],bbox=s.bbox,horizontal_count=s.horizontal_count,vertical_count=s.vertical_count)for s in summaries],graphic_bboxes=detect(old,ps,size,em)))
        c=dict(label=label,request=request,expected=expected)
        if pdf:c['pdf_ref']=pdf
        cases.append(c)
    add('empty',[]);add('positive',frame(),[path()]);add('frame-only',frame());add('isolated-path',[],[path()])
    for order in itertools.permutations(range(4)):add('order-'+str(order),[frame()[i]for i in order],[path()])
    for gap in [0.,2.,7.499999,7.5,7.500001,10.]:
        for em in [0.,1.,2.6666666666666665,10.,20.]:
            add(f'gap-{gap}-{em}',[line([0.,0.,10.,0.],'horizontal'),line([10.+gap,0.,20.+gap,0.],'horizontal')],em=em)
    for xy in [(3.,4.),(4.5,6.),(math.nextafter(4.5,math.inf),6.),(math.nextafter(4.5,-math.inf),6.),(.1,.2),(1e-200,1e-200)]:
        add('hypot-'+str(xy),[line([0.,0.,0.,0.],'horizontal'),line([xy[0],xy[1],xy[0],xy[1]],'vertical')],em=math.hypot(*xy)/.75)
    # Connected through bridges, duplicate geometry, components encountered in raw order rather than root order.
    add('bridge',[line([100.,0.,110.,0.],'horizontal'),line([0.,0.,10.,0.],'horizontal'),line([7.,0.,103.,0.],'horizontal')],[path()])
    add('raw-component-order',[line([300.,0.,310.,0.],'horizontal'),line([0.,0.,10.,0.],'horizontal'),line([7.,0.,12.,0.],'horizontal'),line([307.,0.,320.,0.],'horizontal')])
    add('duplicates',frame()+frame(),[path()]);add('reversed',[line([20.,0.,10.,0.],'horizontal'),line([10.,0.,30.,0.],'vertical')])
    for count in [1,2,3,4]:add('count-'+str(count),frame()[:count],[path()])
    for em in [0.,1.,2.,10.,32.5,37.5,37.5001,43.333333333333336]:add('height-'+str(em),frame(),[path()],em=em)
    for size in [(230.76923076923077,600.),(230.769,600.),(400.,260.),(400.,259.999),(0.,600.),(-400.,600.)]:add('page-'+str(size),frame(),[path()],size=size)
    for segments in [5,6,8]:
        for depth in [0,1]:add(f'path-{segments}-{depth}',frame(),[path(segments=segments,depth=depth)])
    for shift in [0.,100.,125.,130.,130.001]:add('overlap-'+str(shift),frame(),[path((60.+shift,70.,180.+shift,160.))])
    lines=frame();lines[0]['bbox']=[50,50,200,50];lines[2]['bbox']=[50.,50,50,180.];add('mixed-int-float',lines,[path()])
    rng=random.Random(261102)
    for i in range(100):
        lines=frame();l=rng.choice(lines);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-8.,8.]);add('geometry-'+str(i),lines,[path()])
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    prior=json.loads((ROOT/'build/migration/native-path-graphics.json').read_text(encoding='utf-8'));seen=set()
    for c in prior['cases']:
        if'pdf_ref'not in c:continue
        ref=c['pdf_ref'];key=(ref['pdf'],ref['page_index'])
        if key in seen:continue
        seen.add(key);data=(ROOT/ref['pdf']).read_bytes();assert digest(data)==ref['pdf_sha256']
        with PDFDocument(data)as doc:
            snapshot=doc._extract_native_page(ref['page_index']);lines=[dict(bbox=l.bbox,width=l.width,orientation=l.orientation)for l in snapshot.drawing_lines];paths=[asdict(p)for p in snapshot.path_infos]
            add('pdf-'+str(len(seen)),lines,paths,snapshot.page_size,pdf=ref)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'old drawing component cases;',sum(bool(c['expected']['graphic_bboxes'])for c in cases),'graphic positive;',sum('pdf_ref'in c for c in cases),'actual PDF inputs')
if __name__=='__main__':main()
