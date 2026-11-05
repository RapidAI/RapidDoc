"""Freeze old CJK scale evidence and stacked-column candidates without replacing decisions."""
import copy,json,random
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-stacked-graphics.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _native_graphic_em as scale,_detect_stacked_column_graphics as detect
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import has_inflated_native_cjk_metrics as inflated
    from rapid_doc._vendor.docvortex_core.document.pdf.native_contracts import PDFPathInfo,PDFDrawingLine
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    def chars(count=20,width=10.,loose=30.,ink=10.):return [dict(char='中',bbox=[0.,0.,width,loose],tight_bbox=[0.,0.,width,ink])for _ in range(count)]
    def path(bars,b=None,fill=True,segments=5):return dict(bbox=b or [min(x[0]for x in bars),min(x[1]for x in bars),max(x[2]for x in bars),max(x[3]for x in bars)],segment_count=segments,fill_visible=fill,stroke_visible=False,form_depth=0,source_index=0,rectangle_bboxes=bars)
    def source():return dict(chars=chars(),path_infos=[path([[50.+i*20.,100.-i*10.,60.+i*20.,150.]for i in range(4)])],lines=[dict(text=str(i*10),bbox=[20.,80.+i*25.,40.,90.+i*25.],source_index=i,angle=0)for i in range(3)],drawing_lines=[dict(bbox=[45.,150.,130.,150.],width=1.,orientation='horizontal')])
    cases=[]
    def add(label,s,em=10.,fallback=30.,pdf=None):
        request=clean(dict(source=copy.deepcopy(s),em=em,fallback_em=fallback));old=copy.deepcopy(s)
        for c in old.get('chars',[]):
            for k in ['bbox','tight_bbox']:
                if c.get(k):c[k]=Bbox(c[k])
        old['path_infos']=[PDFPathInfo(**dict(p,bbox=tuple(p['bbox']),rectangle_bboxes=tuple(tuple(b)for b in p.get('rectangle_bboxes',[]))))for p in old.get('path_infos',[])]
        old['lines']=[_LineItem(**r)for r in old.get('lines',[])]
        old['drawing_lines']=[PDFDrawingLine((0.,0.),(0.,0.),tuple(r['bbox']),r['width'],r['orientation'])for r in old.get('drawing_lines',[])]
        old=SimpleNamespace(**old)
        expected=clean(dict(graphic_em=scale(old,fallback),inflated_cjk_metrics=inflated(old.chars),stacked_column_bboxes=detect(old,em)))
        c=dict(label=label,request=request,expected=expected)
        if pdf:c['pdf_ref']=pdf
        cases.append(c)
    add('positive',source());add('empty',dict(chars=[],path_infos=[],lines=[],drawing_lines=[]))
    for n in [0,1,19,20,21]:
        for width in [0.,5.,10.]:
            for ink in [4.99,5.,10.,14.,14.01]:
                s=source();s['chars']=chars(n,width,30.,ink);add(f'anchors-{n}-{width}-{ink}',s)
    for loose in [21.99,22.,22.01,30.]:
        for fallback in [21.99,22.,22.01,30.]:
            s=source();s['chars']=chars(loose=loose);add(f'loose-fallback-{loose}-{fallback}',s,fallback=fallback)
    for char in ['㐀','一','中','鿿','ꀀ','中中','中 ','']:
        s=source()
        for c in s['chars']:c['char']=char
        add('unicode-'+str(ord(char[0])if char else 0)+str(len(char)),s)
    for n in [3,4,5]:
        for width in [2.49,2.5,10.,20.,20.01]:
            s=source();s['path_infos']=[path([[50.+i*30.,100.-i*10.,50.+i*30.+width,150.]for i in range(n)])];add(f'bar-{n}-{width}',s)
    for gap in [1.99,2.,2.01]:
        s=source();s['path_infos']=[path([[50.+i*(10.+gap),100.-i*10.,60.+i*(10.+gap),150.]for i in range(4)])];add('gap-'+str(gap),s)
    for delta in [1.99,2.,2.01,10.]:
        for count in [1,2]:
            s=source()
            for b in s['path_infos'][0]['rectangle_bboxes'][:count]:b[3]+=delta
            add(f'baseline-{delta}-{count}',s)
    for height in [29.99,30.,30.01,1000.]:
        s=source()
        for b in s['path_infos'][0]['rectangle_bboxes']:b[1]=150.-height
        add('height-'+str(height),s)
    for fill in [True,False]:
        s=source();s['path_infos'][0]['fill_visible']=fill;add('fill-'+str(fill),s)
    for offset in [.99,1.,1.01]:
        for bottom in [151.99,152.,152.01]:
            s=source();s['path_infos'].append(path([[50.+offset,10.,60.+offset,bottom]]));add(f'peer-{offset}-{bottom}',s)
    for text in ['１２','١٢','1,2','1.2','-12','+12','1e2','1,','1.',' 12　','12 3']:
        s=source()
        for l in s['lines']:l['text']=text
        add('tick-'+text,s)
    for count in [2,3,4]:
        s=source();s['lines']=(s['lines']*2)[:count];add('tick-count-'+str(count),s)
    for field,values in [('x',[9.99,10.,40.,50.,50.01]),('y',[49.99,50.,160.,160.01])]:
        for value in values:
            s=source()
            for l in s['lines']:
                if field=='x':l['bbox'][2]=value
                else:l['bbox'][1]=value-5;l['bbox'][3]=value+5
            add(f'tick-{field}-{value}',s)
    for width in [29.99,30.,30.01]:
        for top in [149.99,150.,190.,190.01]:
            s=source();s['path_infos'].append(path([],b=[45.,top,45.+width,top+20.],segments=8));add(f'label-{width}-{top}',s)
    for orientation in ['horizontal','vertical']:
        for ratio in [.8999,.9,.9001]:
            s=source();s['drawing_lines']=[dict(bbox=[50.,150.,50.+70.*ratio,150.],width=1.,orientation=orientation)];add(f'axis-{orientation}-{ratio}',s)
    rng=random.Random(261031)
    for i in range(80):
        s=source();b=rng.choice(s['path_infos'][0]['rectangle_bboxes']);b[rng.randrange(4)]+=rng.choice([-.01,.01,-4.,4.]);add('geometry-'+str(i),s)
    # Existing frozen live PDF vector+character inputs, no result changes or regenerated PDFs.
    from rapid_doc._vendor.docvortex_core.document.pdf._document import PDFDocument
    vectors=json.loads((ROOT/'build/migration/pdf-vector-primitives.json').read_text(encoding='utf-8'))
    seen=set()
    for c in vectors.get('cases',[]):
        key=(c['pdf'],c['page_index'])
        if key in seen:continue
        seen.add(key);data=(ROOT/c['pdf']).read_bytes()
        with PDFDocument(data)as doc:
            geometry=doc.get_page_chars_with_geometry(c['page_index']);snapshot=doc._extract_native_page(c['page_index'])
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _build_native_line_items_from_chars
            from dataclasses import asdict,fields
            rows=_build_native_line_items_from_chars(geometry.chars,snapshot.page_size,page_rotation=snapshot.rotation)
            s=dict(chars=clean(geometry.chars),lines=[dict(text=l.text,bbox=l.bbox,source_index=l.source_index,angle=l.angle)for l in rows],path_infos=[asdict(p)for p in snapshot.path_infos],drawing_lines=[asdict(l)for l in snapshot.drawing_lines])
            add('pdf-'+str(len(seen)),s,pdf=dict(pdf=c['pdf'],pdf_sha256=digest(data),page_index=c['page_index']))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'old stacked/scale cases;',sum(bool(c['expected']['stacked_column_bboxes'])for c in cases),'stacked positive;',sum('pdf_ref'in c for c in cases),'actual PDF cases')
if __name__=='__main__':main()
