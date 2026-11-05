"""冻结旧数值重分行/数值分组/对齐候选与完整正则事实，不覆写参考。"""
import copy,json,random
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-aligned-numeric.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import aligned_numeric as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableText,NativeTableTextRow,NativeTableToken,NativeTableGlyph
    cases=[];regex=[];rng=random.Random(61225)
    def make(defs=None,amounts=3,extra=False):
        if defs is None:
            boxes=[(5.,70.)]+[(155.+100*i,175.+100*i)for i in range(amounts)]
            if extra:boxes+=[(amounts*100.+100.,amounts*100.+135.)]
            head=['Item']+[f'Year{chr(65+i)}'for i in range(amounts)]+(['Status']if extra else[])
            defs=[(8.,[(s,*b)for s,b in zip(head,boxes)]),(38.,[(s,*b)for s,b in zip(['Alpha']+[f'{i+1},234'for i in range(amounts)]+(['OK']if extra else[]),boxes)]),(68.,[(s,*b)for s,b in zip(['Beta']+[f'-{i+2}.5'for i in range(amounts)]+(['OK']if extra else[]),boxes)])]
        gs=[];rs=[]
        for row,(y,tokens)in enumerate(defs):
            ts=[];ids=[]
            for s,x0,x1 in tokens:
                i=len(gs);b=(x0,y,x1,y+8.);gs.append(NativeTableGlyph(i,i,s,b,row));ts.append(NativeTableToken(s,b,(i,),(i,)));ids.append(i)
            rs.append(NativeTableTextRow(row,(min([t[1]for t in tokens],default=0.),y,max([t[2]for t in tokens],default=0.),y+8.),tuple(ts),tuple(ids)))
        width=100.+100*amounts+(70. if extra else 0.)
        lines=tuple(NativeTableRule((0.,y-.1,width,y+.1),.2,'horizontal')for y in [0.,30.,100.])
        return NativeTableInput((0.,0.,width,100.),(1000.,1000.),0,(),lines,()),NativeTableText(tuple(gs),tuple(rs),8.,10.)
    def add(inp,text,regroup=True):
        t=old.regroup_numeric_text(text)if regroup else text;groups=old.numeric_row_groups(t);c=old.build_aligned_numeric_candidate(inp,t)
        cases.append(dict(request=dict(table_input=asdict(inp),text=asdict(text),regroup=regroup),expected=dict(text=asdict(t),numeric_groups=groups,candidate=None if c is None else asdict(c),html=None if c is None else serialize_candidate_html(c))))
    for n in [2,3,4,5]:
        for extra in [False,True]:
            for regroup in [False,True]:add(*make(amounts=n,extra=extra),regroup=regroup)
    inp,text=make()
    for count in [0,1,2,3,4,5]:
        lines=tuple(NativeTableRule((0.,y-.1,400.,y+.1),.2,'horizontal')for y in [0.,30.,100.,50.,60.][:count]);add(replace(inp,drawing_lines=lines),text)
    for count in [0,2,3]:
        lines=inp.drawing_lines+tuple(NativeTableRule((x-.1,0.,x+.1,100.),.2,'vertical')for x in [0.,110.,210.][:count]);add(replace(inp,drawing_lines=lines),text)
    for label in ['Alpha','1、Section','一、Section','一2、mixed','Time 12:30','９：３０','Unit(kg)','Unit（）']:
        gs=list(text.glyphs);gs[4]=replace(gs[4],text=label);rs=list(text.rows);ts=list(rs[1].tokens);ts[0]=replace(ts[0],text=label);rs[1]=replace(rs[1],tokens=tuple(ts));add(inp,replace(text,glyphs=tuple(gs),rows=tuple(rs)))
    for numeric in ['0','123','−(1,234.50)%','(123','123)','-','–','—','１２３','١٢٣','²','1e3','2024年','12:30']:
        gs=list(text.glyphs);rs=list(text.rows)
        for index in [1,2]:
            ts=list(rs[index].tokens)
            for col in range(1,4):
                t=ts[col];ts[col]=replace(t,text=numeric);i=t.glyph_ids[0];gs[i]=replace(gs[i],text=numeric)
            rs[index]=replace(rs[index],tokens=tuple(ts))
        add(inp,replace(text,glyphs=tuple(gs),rows=tuple(rs)))
    # header spanning/rowspan, empty subsections, pending units/continuation and an extra status column.
    base=[('Item',5.,70.),('YearA',155.,175.),('YearB',255.,275.),('YearC',355.,375.)]
    data=[('Alpha',5.,70.),('123',155.,175.),('456',255.,275.),('789',355.,375.)]
    for y in [18.,20.5,24.,28.,35.]:
        for label in ['Section','1、Section','一、Section','Unit(kg)','Unit(\na)','Unit(a)\n']:
            defs=[(4.,base),(y,[(label,5.,70.)]),(45.,data),(75.,[('Beta',15.,70.),*data[1:]])]
            add(*make(defs=defs))
    for group in [[('Group',120.,400.)],[('Item',5.,70.),('Group',120.,390.)],[('Item',5.,70.)]]:
        ti,tx=make(defs=[(2.,group),(18.,base),(45.,data),(75.,[('Beta',5.,70.),*data[1:]])]);ti=replace(ti,drawing_lines=ti.drawing_lines+(NativeTableRule((120.,13.,390.,13.2),.2,'horizontal'),));add(ti,tx)
    for height in [3.,6.499999,6.5,8.]:
        ti,tx=make();g=list(tx.glyphs);g[1]=replace(g[1],text='²',bbox=(155.,8.,175.,8.+height));add(ti,replace(tx,glyphs=tuple(g)))
    for offset in [0.,6.5,6.500001,20.]:
        ti,tx=make();g=list(tx.glyphs);rs=list(tx.rows);ts=list(rs[2].tokens);b=ts[2].bbox;ts[2]=replace(ts[2],bbox=(b[0]+offset,b[1],b[2]+offset,b[3]));g[ts[2].glyph_ids[0]]=replace(g[ts[2].glyph_ids[0]],bbox=ts[2].bbox);rs[2]=replace(rs[2],tokens=tuple(ts));add(ti,replace(tx,glyphs=tuple(g),rows=tuple(rs)))
    # Tall numeric boxes interleave label lines; explicit spacing and duplicate IDs survive regroup.
    for tie in [False,True]:
        g=list(text.glyphs);g[4]=replace(g[4],bbox=(5.,25.,70.,33.));g[5]=replace(g[5],bbox=(155.,20.,175.,55.));g.append(replace(g[4],glyph_id=99 if not tie else 4,source_index=99,text='cont',bbox=(15.,38.,65.,46.),explicit_space_before=True));add(inp,replace(text,glyphs=tuple(g)))
    # Reuse actual old PDF primitive/text inputs; no Rust PDF extraction is claimed.
    pdf=json.loads((ROOT/'build/migration/native-table-vector-pdf.json').read_text(encoding='utf-8'));real=[]
    for ref in pdf['real']:
        r=pdf['cases'][ref['case_index']]['request'];v=r['table_input'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],(),tuple(NativeTableRule(**q)for q in v['drawing_lines']),())
        t=r['text'];tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in t['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**tok)for tok in row['tokens']))))for row in t['rows']),t['median_glyph_width'],t['median_glyph_height']);i=len(cases);add(ti,tx);real.append(ref|dict(case_index=i))
    for _ in range(120):
        n=rng.choice([2,3,4]);ti,tx=make(amounts=n,extra=rng.choice([False,True]));gs=[]
        for g in tx.glyphs:
            dx=rng.choice([0.,0.,0.,rng.uniform(-8,8)]);dy=rng.uniform(-1,1);b=g.bbox;gs.append(replace(g,bbox=(b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy)))
        add(ti,replace(tx,glyphs=tuple(gs)))
    strings=['','-','–','—','−','+','1.','1,','1,23.4','(1','1)','(1)','1 %','1\n%','+１２３‰','١٢٣','²','½','2024年','1\x1c%']
    for sign in ['','+','−','-']:
        for number in ['0','123','١٢٣','１２３','1.23','1,234.5','²','1e3']:
            for open_,close in [('', ''),('(',')'),('(',''),('',')')]:
                for suffix in ['','%',' ‰','\n%']:strings.append(sign+open_+number+close+suffix)
    for s in strings:regex.append(dict(text=s,expected=bool(old._VALUE.fullmatch(s.strip()))))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(target,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in ['aligned_numeric.py','text.py','sparse_common.py','candidate.py']},cases=cases,regex=regex,real=real))
    print(f'Frozen {len(cases)} numeric regroup/candidate cases +{len(regex)} old numeric regex cases; {sum(c["expected"]["candidate"]is not None for c in cases)} candidates')
if __name__=='__main__':main()
