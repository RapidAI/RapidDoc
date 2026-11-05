"""冻结旧PDF原语→完整矢量候选（line/rect/raw/supported），既有参考不覆写。"""
import copy,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_table_vector_grid import plain
def main():
    target=ROOT/'build/migration/native-table-vector-pipeline.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    rng=random.Random(91251);cases=[];track_cases=[]
    def table(rows=3,cols=3,mode='lines',angle=0):
        lines=[];rects=[]
        if mode in ['lines','both']:
            lines=[NativeTableRule((0.,r*30.-.1,cols*50.,r*30.+.1),.2,'horizontal')for r in range(rows+1)]+[NativeTableRule((c*50.-.1,0.,c*50.+.1,rows*30.),.2,'vertical')for c in range(cols+1)]
        if mode in ['rects','both']:
            rects=[NativeTableRectangle((c*50.,r*30.,(c+1)*50.,(r+1)*30.),5,False,True)for r in range(rows)for c in range(cols)]
        return NativeTableInput((0.,0.,cols*50.,rows*30.),(600.,800.),angle,(),tuple(lines),tuple(rects))
    def text(rows=3,cols=3,missing=()):
        gs=[];rs=[]
        for r in range(rows):
            ts=[];ids=[]
            for c in range(cols):
                if(r,c)in missing:continue
                i=len(gs);b=(c*50.+8.,r*30.+6.,c*50.+18.,r*30.+16.)
                gs.append(NativeTableGlyph(i,i,'X',b,r));ids.append(i);ts.append(NativeTableToken('X',b,(i,),(i,)))
            if ids:rs.append(NativeTableTextRow(r,(0.,r*30.+6.,cols*50.,r*30.+16.),tuple(ts),tuple(ids)))
        return NativeTableText(tuple(gs),tuple(rs),10.,10.)
    def add(t,txt=None,tracks=False):
        txt=text()if txt is None else txt;diagnostics=[]
        result=old.build_vector_candidates(t,txt,diagnostics)
        cases.append(dict(request=dict(table_input=asdict(t),text=asdict(txt)),expected=plain(dict(candidates=[asdict(c)for c in result],diagnostics=diagnostics,html=[serialize_candidate_html(c)for c in result]))))
        if tracks:
            for drawing,rect,supported in [(True,False,False),(True,False,True),(False,True,False)]:
                d={};value=old._build_vector_tracks(t,txt,include_drawing=drawing,include_rectangles=rect,prune_unsupported_horizontal=supported,diagnostics=d)
                track_cases.append(dict(request=dict(table_input=asdict(t),text=asdict(txt),include_drawing=drawing,include_rectangles=rect,prune_unsupported_horizontal=supported),expected=plain(dict(tracks=None if value is None else asdict(value),diagnostics=d))))
    from dataclasses import replace
    for rows,cols in [(1,2),(1,3),(2,1),(3,1),(2,2),(3,3),(4,3)]:
        for mode in ['none','lines','rects','both']:
            add(table(rows,cols,mode),text(rows,cols),True)
    for angle in [90,180,270,-90,360,45]:add(table(angle=angle),text(),True)
    # 装饰短横触发supported假设；窄边/短竖线按空白或穿正文决定剔除。
    for box in [(15.,40.,25.,40.2),(0.,40.,50.,40.2),(3.,0.,3.2,8.),(3.,0.,3.2,27.),(3.,0.,3.2,26.999999),(145.,0.,145.2,8.)]:
        t=table();t=replace(t,drawing_lines=t.drawing_lines+(NativeTableRule(box,.2,'horizontal'),));add(t,text(),True)
    # halo：外框在表框外；多行与单行/单列条件分开。
    for rows,cols in [(1,3),(3,1),(3,3)]:
        for offset in [.5,1.,2.,3.]:
            t=table(rows,cols);t=replace(t,drawing_lines=tuple(replace(r,bbox=(r.bbox[0],r.bbox[1]-offset,r.bbox[2],r.bbox[3]-offset))if r.orientation=='horizontal'else r for r in t.drawing_lines));add(t,text(rows,cols),True)
    for width in [0.,.2,1.,3.,10.]:
        t=table();add(replace(t,drawing_lines=tuple(replace(r,width=width)for r in t.drawing_lines)),text(),True)
    # 原子行内稠密双基线：答题列表头例外、isdigit（含非decimal字符）及部分续行。
    for header in ['A','2','²','①','½','四','𝟚']:
        t=table(4,3);txt=text(4,3,missing=[(r,c)for r in range(1,4)for c in[1,2]]+[(0,0)])
        g=list(txt.glyphs);rs=list(txt.rows);new=[]
        for token in rs[0].tokens:
            i=len(g);b=(token.bbox[0],18.,token.bbox[2],27.);g.append(NativeTableGlyph(i,i,header,b,10));new.append(NativeTableToken(header,b,(i,),(i,)))
        rs.insert(1,NativeTableTextRow(10,(50.,18.,150.,27.),tuple(new),tuple(x.glyph_ids[0]for x in new)))
        add(t,NativeTableText(tuple(g),tuple(rs),10.,10.),True)
    # 全列的第二文本基线不是单元格换行；移去内部纵线时部分续行仍应拒绝。
    for full_vertical in [False,True]:
        t=table();
        if not full_vertical:t=replace(t,drawing_lines=tuple(r for r in t.drawing_lines if r.orientation!='vertical'or r.bbox[0]<1 or r.bbox[0]>149))
        txt=text();rs=list(txt.rows);gs=list(txt.glyphs);ts=[]
        for c in range(3):
            i=len(gs);b=(c*50.+8.,18.,c*50.+18.,27.);gs.append(NativeTableGlyph(i,i,'X',b,9));ts.append(NativeTableToken('X',b,(i,),(i,)))
        rs.insert(1,NativeTableTextRow(9,(0.,18.,150.,27.),tuple(ts),tuple(x.glyph_ids[0]for x in ts)))
        add(t,NativeTableText(tuple(gs),tuple(rs),10.,10.),True)
    # 退化/反向table bbox，旧decoration使用raw框，后续证据规范化。
    for b in [(150.,90.,0.,0.),(0.,0.,0.,90.)]:add(replace(table(),table_bbox=b),text(),True)
    for _ in range(180):
        rows,cols=rng.randrange(2,6),rng.randrange(2,6);t=table(rows,cols,rng.choice(['lines','rects','both']));ls=[]
        for r in t.drawing_lines:
            if rng.random()<.12:continue
            if rng.random()<.2:
                b=r.bbox
                r=replace(r,bbox=(b[0],b[1],b[2]*rng.choice([.4,.8,1.]),b[3]))if r.orientation=='horizontal'else replace(r,bbox=(b[0],b[1],b[2],b[3]*rng.choice([.4,.8,1.])))
            ls.append(r)
        t=replace(t,drawing_lines=tuple(ls));missing=[(r,c)for r in range(rows)for c in range(cols)if rng.random()<.15]
        add(t,text(rows,cols,missing))
    sources=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'+p for p in ['vector.py','geometry.py','candidate.py','text.py','contracts.py']]
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in sources},cases=cases,track_cases=track_cases))
    print(f'Frozen {len(cases)} complete vector generation +{len(track_cases)} track-stage cases')
if __name__=='__main__':main()
