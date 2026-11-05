"""Freeze complete old lane inference with observational route accounting, without mocks."""
import copy,json,random,sys
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/native-text-lanes.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import line_layout as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_to_upright
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    cases=[]
    def line(i,b,**kw):return dict(text='ordinary body text',bbox=list(map(float,b)),angle=0,source_index=i,effective_height=10.,**kw)
    def add(label,geometry,width=300.,height=10.):
        for recalculate in (False,True):
            request=dict(line_geometry=copy.deepcopy(geometry),local_page_width=width,median_height=height,recalculate_intervals=recalculate)
            items=[]
            for l,b in geometry:
                l=copy.deepcopy(l)
                if l.get('font_signature')is not None:l['font_signature']=tuple(l['font_signature'])
                items.append((_LineItem(**l),tuple(b)))
            identities={id(l):i for i,(l,b)in enumerate(items)};events={};active={}
            observed={'_reattach_span_lane_continuations':'span_continuation','_reattach_repeated_indented_span_tails':'repeated_tail','_reattach_cross_lane_short_tails':'cross_tail'}
            def snapshot(lanes):return [(l.left,l.right,l.is_span,[identities[id(q)]for q,b in l.lines])for l in lanes]
            def profile(frame,event,arg):
                name=frame.f_code.co_name
                if name=='_infer_nested_column_band'and event=='return'and arg is not None:events['nested']=True
                elif name in observed:
                    if name=='_reattach_repeated_indented_span_tails':lanes=[frame.f_locals['span_lane'],*frame.f_locals['regular_lanes']]
                    else:lanes=frame.f_locals.get('lanes')
                    if lanes is None:return
                    if event=='call':active[id(frame)]=snapshot(lanes)
                    elif event=='return':
                        if active.pop(id(frame),None)!=snapshot(lanes):events[observed[name]]=True
            sys.setprofile(profile)
            try:lanes=old._infer_text_lanes(items,width,height,recalculate_intervals=recalculate)
            finally:sys.setprofile(None)
            expected=[dict(left=l.left,right=l.right,is_span=l.is_span,line_indices=[identities[id(q)]for q,b in l.lines])for l in lanes]
            cases.append(dict(label=f'{label}:{recalculate}',request=encode(request),expected=encode(expected),events=events))
    # Stable two-column anchors with wide rows, repeated indents, and supported short tails.
    for variant in range(60):
        ls=[]
        for y in (70.,90.,110.,130.):
            for x in (10.,170.):ls.append(line(len(ls),(x,y,x+120.,y+10.)))
        if variant%5==0:
            for y in (0.,10.):ls.append(line(len(ls),(10.,y,290.,y+10.)))
            ls.append(line(len(ls),(10.,20.,70.,30.)))
        elif variant%5==1:
            for y in (0.,20.,40.):
                ls.append(line(len(ls),(10.,y,290.,y+10.)))
                ls.append(line(len(ls),(25.,y+10.,100.,y+20.)))
        elif variant%5==2:
            ls.append(line(len(ls),(10.,0.,290.,10.)))
            ls.append(line(len(ls),(10.,18.,100.,28.)))
        elif variant%5==3:
            ls.append(line(len(ls),(10.,0.,130.,10.)))
            ls.append(line(len(ls),(10.,18.,70.,28.)))
            ls.append(line(len(ls),(10.,36.,50.,46.)))
        else:
            for y in (150.,162.,174.,186.,198.,210.,222.):ls.append(line(len(ls),(10.,y,290.,y+10.)))
        if variant//5%2:
            for l in ls:l.update(font_signature=['Body',0],font_coverage=1.,dominant_font_weight=400.)
        if variant//10%2:
            ls[0]['dominant_font_weight']=700.;ls[-1]['semantic_type']='page_footnote'
        if variant//20%2:ls.reverse()
        add(f'columns:{variant}',[(l,l['bbox'])for l in ls])
    # Narrow anchors and wide single-sided body rows, with outer body below.
    for variant in range(32):
        ls=[]
        for y in (0.,10.,20.,30.,40.):
            for x,w in ((10.,45.),(115.,75.)):ls.append(line(len(ls),(x,y,x+w,y+8.)))
        ls.extend([line(20,(10.,32.,105.,40.)),line(21,(10.,42.,105.,50.)),line(22,(10.,74.,130.,82.))])
        for k,y in enumerate((90.,100.,110.,120.,130.,140.,150.,160.,170.,180.)):ls.append(line(30+k,(10.,y,190.,y+8.)))
        if variant%2:ls[-1]['semantic_type']='header'
        if variant%4:ls[1]['style_scale_repaired']=True;ls[1]['em_height']=16.
        if variant%8:ls[2]['restored_inline_cluster']=True
        if variant>=16:ls.reverse()
        add(f'nested:{variant}',[(l,l['bbox'])for l in ls],200.,8.)
    rng=random.Random(581923)
    for n in range(240):
        geometry=[]
        for i in range(rng.randrange(1,94)):
            x=rng.choice([0.,10.,110.,120.,200.]);y=float(rng.randrange(32)*6);w=rng.choice([20.,45.,90.,140.,220.]);h=rng.choice([5.,8.,10.])
            l=line(i,(x,y,x+w,y+h));l.update(effective_height=rng.choice([0.,5.,10.]),em_height=8.,style_scale_repaired=bool(rng.randrange(2)),semantic_type=rng.choice([None,None,None,'header','page_footnote','aside_text']),font_signature=rng.choice([None,['A',0],['B',64],['A',False]]),font_coverage=rng.choice([0.,.5,.75,1.]),dominant_font_weight=rng.choice([None,400.,700.]),baseline=y+rng.choice([4.,10.]),restored_inline_cluster=bool(n%3==0),visual_row_id=rng.choice([None,i]))
            geometry.append((l,l['bbox']))
        add(f'random:{n}',geometry,320.,rng.choice([5.,10.]))
    f=json.loads((ROOT/'build/migration/table-script-preparation.json').read_text(encoding='utf-8'))
    for n,c in enumerate(f['preparation']):
        if n%3:continue
        p=c['request']['table_input']['page_size'];a=c['request']['table_input']['angle'];width=p[1]if a in (90,270)else p[0]
        for j,cell in enumerate(c['expected']['cells'][:6]):
            if not cell['visual_lines']:continue
            geometry=[(l,list(_rotate_bbox_to_upright(l['bbox'],p,a)))for l in cell['visual_lines']]
            add(f'cell:{n}:{j}',geometry,width,c['request']['result']['text']['median_glyph_height'])
    refs=['line_layout.py','geometry.py','models.py']
    sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
    write_json(target,dict(sources=sources,cases=cases))
    print('Frozen',len(cases),'complete old lane cases;', {k:sum(c['events'].get(k,False)for c in cases)for k in ('nested','span_continuation','repeated_tail','cross_tail')})
if __name__=='__main__':main()
