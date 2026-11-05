"""冻结旧三线表候选及共享列假设/空白走廊；不覆写参考。"""
import copy,json,random
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
from export_native_table_vector_grid import plain
def main():
    path=ROOT/'build/migration/native-table-rule-band.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import rule_band as old,sparse_multiline as sparse
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.sparse_common import _local_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.geometry import table_local_size,normalize_angle
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    rng=random.Random(99112);cases=[];columns=[]
    def make(cols=3,header=2,body=4,prefix=0,physical='none'):
        width=cols*60.;height=body*30.+30.;gs=[];rs=[]
        defs=[(4.+14*r,list(range(cols)))for r in range(header)]+[(36.+30*r,list(range(cols))if not prefix or r%2==0 else list(range(prefix,cols)))for r in range(body)]
        for row,(y,cs)in enumerate(defs):
            ts=[];ids=[]
            for c in cs:
                i=len(gs);b=(c*60.+5.,y,c*60.+20.,y+8.);s=f'R{row}C{c}';gs.append(NativeTableGlyph(i,i,s,b,row));ids.append(i);ts.append(NativeTableToken(s,b,(i,),(i,)))
            rs.append(NativeTableTextRow(row,(0.,y,width,y+8.),tuple(ts),tuple(ids)))
        rules=tuple(NativeTableRule((0.,y-.1,width,y+.1),.2,'horizontal')for y in [0.,30.,height]);rects=()
        if physical!='none':rects=tuple(NativeTableRectangle((c*60.,r*30.+31.,(c+1)*60.,r*30.+43.),5,physical=='fill',True)for r in range(body)for c in range(cols))
        return NativeTableInput((0.,0.,width,height),(1000.,1000.),0,(),rules,rects),NativeTableText(tuple(gs),tuple(rs),8.,10.)
    def add(t,text,inspect=True):
        d=[];result=old.build_rule_band_candidates(t,text,d)
        cases.append(dict(request=dict(table_input=asdict(t),text=asdict(text)),expected=dict(candidates=[asdict(c)for c in result],diagnostics=d,html=[serialize_candidate_html(c)for c in result])))
        if inspect:
            w,h=table_local_size(t.table_bbox,normalize_angle(t.angle));rules=_local_rules(t,w,h);rects=sparse._local_rectangles(t,w,h);d={};hyp=sparse._build_column_hypothesis(text,w,rules,rects,d);gd={};stable=None;occupancy=None
            if hyp is not None:
                stable=sparse._stable_gutters(text,hyp.x_tracks,hyp.physical_boundaries,30.,gd);by_id={g.glyph_id:g for g in text.glyphs};occupancy=[sparse._row_occupancy(r,by_id,hyp.x_tracks)for r in text.rows]
            columns.append(dict(request=dict(table_input=asdict(t),text=asdict(text),header_bottom=30.),expected=plain(dict(rules=[asdict(r)for r in rules],rectangles=[asdict(r)for r in rects],hypothesis=None if hyp is None else asdict(hyp)|dict(physical_boundaries=sorted(hyp.physical_boundaries)),hypothesis_diagnostics=d,stable_gutters=stable,gutter_diagnostics=gd,occupancy=occupancy,overlapping_formula_rows=sparse._has_overlapping_formula_rows(text,30.)))))
    for cols in [2,3,4,5]:
        for head in [1,2,3,4]:
            for body in [2,3,4,6]:
                for physical in ['none','stroke','fill']:add(*make(cols,head,body,physical=physical))
    for prefix in [1,2,3]:
        for body in [4,5,6,7,8]:add(*make(cols=4,header=1,body=body,prefix=prefix))
    t,text=make();
    for count in [0,1,2,4]:add(replace(t,drawing_lines=t.drawing_lines[:count]if count<4 else t.drawing_lines+(NativeTableRule((0.,60.,180.,60.2),.2,'horizontal'),)),text)
    for x in [0.,1.5,1.500001,60.,179.,180.]:add(replace(t,drawing_lines=t.drawing_lines+(NativeTableRule((x-.1,0.,x+.1,100.),.2,'vertical'),)),text)
    # 墨迹穿界而token未变；header覆盖及最低稳定走廊支持。
    for x0,x1 in [(5.,42.5),(5.,42.5001),(5.,60.),(41.,50.)]:
        gs=list(text.glyphs);b=gs[6].bbox;gs[6]=replace(gs[6],bbox=(x0,b[1],x1,b[3]));add(t,replace(text,glyphs=tuple(gs)))
    for height in [30.,30.00001]:
        rs=list(text.rows);b=rs[2].bbox;rs[2]=replace(rs[2],bbox=(b[0],b[1],b[2],b[1]+height));add(t,replace(text,rows=tuple(rs)))
    for angle in [90,180,270,45,-90]:add(replace(t,angle=angle),text)
    for b in [(180.,150.,0.,0.),(0.,0.,0.,150.)]:add(replace(t,table_bbox=b),text)
    for _ in range(140):
        t,tx=make(cols=rng.randrange(2,7),header=rng.choice([1,2,3]),body=rng.randrange(3,8),physical=rng.choice(['none','stroke','fill']));g=[]
        for glyph in tx.glyphs:
            b=glyph.bbox;dx=rng.uniform(-5,5);dy=rng.uniform(-.3,.3);g.append(replace(glyph,bbox=(b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy)))
        add(t,replace(tx,glyphs=tuple(g)))
    # 实际金额PDF/角度的原语和字形，独立于给定合成token案例。
    real=[]
    for name in ['native-table-vector-pdf','native-table-aligned-numeric-pdf']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for ref in f['real']:
            r=f['cases'][ref['case_index']]['request'];v=r['table_input'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],(),tuple(NativeTableRule(**q)for q in v['drawing_lines']),tuple(NativeTableRectangle(**q)for q in v['rectangles']))
            q=r['text'];txt=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**tok)for tok in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height']);i=len(cases);add(ti,txt);real.append(ref|dict(case_index=i))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in ['rule_band.py','sparse_multiline.py','sparse_common.py','candidate.py']},cases=cases,columns=columns,real=real))
    print(f'Frozen {len(cases)} rule-band candidates +{len(columns)} shared column/gutter facts; {sum(bool(c["expected"]["candidates"])for c in cases)} nonempty cases')
if __name__=='__main__':main()
