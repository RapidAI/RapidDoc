"""Freeze old isolated vector grouping and meaningful small raster decisions."""
import copy,json,math,random
from types import SimpleNamespace
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-isolated-graphics.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.isolated_graphics import _isolated_path_groups as group,isolated_vector_components as vectors,meaningful_small_raster as small
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.line_layout import _line_effective_height
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_area
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    import statistics
    assert get_native()is None
    def path(b=(50.,50.,90.,90.),segments=30,color=(200,0,0,255),**kw):return dict(bbox=b,segment_count=segments,fill_visible=True,stroke_visible=False,form_depth=0,source_index=0,rectangle_bboxes=[],fill_rgba=color,**kw)
    def line(b=(200.,200.,300.,210.),h=10.,**kw):return dict(dict(text='body',bbox=b,source_index=0,effective_height=h,angle=0),**kw)
    def source(paths=None,lines=None,rasters=()):return dict(page_size=[400.,600.],path_infos=[path()]if paths is None else paths,lines=[]if lines is None else lines,image_bboxes=rasters)
    cases=[]
    def add(label,raw,pdf=None):
        request=clean(dict(source=copy.deepcopy(raw)));s=copy.deepcopy(raw)
        s['path_infos']=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox']),fill_rgba=tuple(p['fill_rgba'])if p.get('fill_rgba')is not None else None,rectangle_bboxes=tuple(tuple(b)for b in p.get('rectangle_bboxes',[]))))for p in s['path_infos']]
        s['lines']=[_LineItem(**l)for l in s['lines']];s=SimpleNamespace(**s)
        heights=[_line_effective_height(l,l.bbox)for l in s.lines if l.angle==0];em=statistics.median(heights)if heights else 10.
        area=s.page_size[0]*s.page_size[1];eligible=[p for p in s.path_infos if p.form_depth==0 and(p.fill_visible or p.stroke_visible)and not p.rectangle_bboxes and p.segment_count>=8 and 0<_bbox_area(p.bbox)<=.06*area]
        positions={id(p):i for i,p in enumerate(s.path_infos)};groups=[[positions[id(p)]for p in g]for g in group(s,eligible,em)]
        expected=clean(dict(em=em,path_groups=groups,vector_bboxes=vectors(s),meaningful_raster_indices=[i for i,b in enumerate(s.image_bboxes)if small(s,b)]))
        c=dict(label=label,request=request,expected=expected)
        if pdf:c['pdf_ref']=pdf
        cases.append(c)
    add('positive',source());add('empty',source(paths=[]));add('small-raster',source(paths=[],lines=[line()],rasters=[[50.,50.,90.,90.]]))
    for gap in [3.499999,3.5,3.500001]:
        add('gap-'+str(gap),source(paths=[path([50.,50.,90.,90.],20),path([90.+gap,50.,130.+gap,90.],20)]))
    for segments in [7,8,19,20,29,30,31]:add('segments-'+str(segments),source(paths=[path(segments=segments)]))
    for width in [14.99,15.,24.99,25.,25.01,120.]:add('width-'+str(width),source(paths=[path([50.,50.,50.+width,90.])]))
    for field,values in [('form_depth',[-1,0,1]),('fill_visible',[False,True]),('stroke_visible',[False,True]),('rectangle_bboxes',[[],[[50.,50.,90.,90.]]])]:
        for value in values:
            p=path();p[field]=value;add(f'eligibility-{field}-{value}',source(paths=[p]))
    for ratio in [.00049999,.0005,.00050001,.05999999,.06,.06000001]:
        s=source();s['page_size']=[400.,1600./ratio/400.];add('area-'+str(ratio),s)
    for height in [0.,-1.,5.,10.,16.,16.0001]:add('height-'+str(height),source(lines=[line(h=height)]))
    for em in [0.,5.,10.,16.]:
        for repaired in [False,True]:add(f'repaired-{em}-{repaired}',source(lines=[line(h=30.,em_height=em,style_scale_repaired=repaired)]))
    for angle in [0,90,180,270]:add('angle-'+str(angle),source(lines=[line(h=20.,angle=angle)]))
    for count in [1,2,3,8,9]:
        for colors in [1,2,3]:
            ps=[path([0.,0.,30.,30.],20,color=((i%colors)*60,0,0,255))for i in range(count)];add(f'corner-{count}-{colors}',source(paths=ps))
    ps=[path([0.,0.,30.,30.],20,color=None)for _ in range(8)];add('none-color',source(paths=ps))
    for width in [.9999,1.,1.0001,9.9999,10.,10.0001]:add('text-overlap-'+str(width),source(lines=[line([90.-width,50.,100.-width,90.])]))
    for x in [41.9999,42.,42.0001,82.,82.0001]:add('raster-overlap-'+str(x),source(rasters=[[x,50.,x+40.,90.]]))
    for w in [29.9999,30.,30.0001]:add('small-width-'+str(w),source(paths=[],lines=[line()],rasters=[[50.,50.,50.+w,90.]]))
    for width in [.09999,.1,.10001]:add('small-text-overlap-'+str(width),source(paths=[],lines=[line([90.-width,50.,100.-width,90.])],rasters=[[50.,50.,90.,90.]]))
    add('merge-order',source(paths=[path([50.,50.,80.,80.],20),path([120.,50.,150.,80.],20),path([82.,50.,118.,80.],20)]))
    add('duplicate-geometry',source(paths=[path(),path()]));add('mixed-types',source(paths=[path([50,50,90,90]),path([50.,50,90.,90])]))
    rng=random.Random(261103)
    for i in range(100):
        p=path();p['bbox']=list(p['bbox']);p['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-15.,15.]);add('geometry-'+str(i),source(paths=[p],lines=[line()]))
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
    prior=json.loads((ROOT/'build/migration/native-drawing-graphics.json').read_text(encoding='utf-8'))
    for c in prior['cases']:
        if'pdf_ref'not in c:continue
        ref=c['pdf_ref'];data=(ROOT/ref['pdf']).read_bytes();assert digest(data)==ref['pdf_sha256']
        with PDFDocument(data)as doc:
            snap=doc._extract_native_page(ref['page_index']);g=doc.get_page_chars_with_geometry(ref['page_index']);rows=_build_native_line_items_from_chars(g.chars,snap.page_size,page_rotation=snap.rotation)
            s=dict(page_size=snap.page_size,path_infos=[asdict(p)for p in snap.path_infos],image_bboxes=[p.bbox for p in snap.image_infos],lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,angle=l.angle,effective_height=l.effective_height,em_height=l.em_height,style_scale_repaired=l.style_scale_repaired)for l in rows]);add('pdf-'+str(ref['page_index'])+ref['pdf'],s,pdf=ref)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/isolated_graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'old isolated facts;',sum(bool(c['expected']['vector_bboxes'])for c in cases),'vector-positive;',sum(bool(c['expected']['meaningful_raster_indices'])for c in cases),'small-raster-positive;',sum('pdf_ref'in c for c in cases),'actual PDF inputs')
if __name__=='__main__':main()
