"""冻结实际旧text_grid.py三路候选/诊断，拒绝覆写已有参考。"""
import copy,json,random
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-text-grid.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import text_grid as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableText,NativeTableGlyph,NativeTableTextRow,NativeTableToken
    cases=[];rng=random.Random(37521)
    def add(inp,text):
        d=[];c=old.build_text_candidates(inp,text,d)
        cases.append(dict(request=dict(table_input=asdict(inp),text=asdict(text)),expected=dict(candidates=[asdict(x)for x in c],diagnostics=d,html=[serialize_candidate_html(x)for x in c])))
    def text(row_defs):
        gs=[];rows=[]
        for index,(top,boxes)in enumerate(row_defs):
            tokens=[];ids=[]
            for col,(x0,x1)in enumerate(boxes):
                i=len(gs);b=(x0,top,x1,top+8.);s=f'{index}:{col}'
                gs.append(NativeTableGlyph(i,i,s,b,index));tokens.append(NativeTableToken(s,b,(i,),(i,)));ids.append(i)
            rows.append(NativeTableTextRow(index,(min([b[0]for b in boxes],default=0.),top,max([b[1]for b in boxes],default=0.),top+8.),tuple(tokens),tuple(ids)))
        return NativeTableText(tuple(gs),tuple(rows),8.,10.)
    def inp(rows=4,cols=3,physical='none'):
        w,h=50.*cols,30.*rows;lines=();rects=()
        if physical in ['lines','both']:lines=tuple(NativeTableRule((0.,r*30.-.1,w,r*30.+.1),.2,'horizontal')for r in range(rows+1))
        if physical in ['stripes','both']:rects=tuple(NativeTableRectangle((0.,r*30.,w,r*30.+12.),5,True,False)for r in range(rows))
        return NativeTableInput((0.,0.,w,h),(500.,500.),0,(),lines,rects)
    for rows in [0,1,2,3,4,6]:
        for cols in [1,2,3,4]:
            for physical in ['none','lines','stripes','both']:
                add(inp(max(1,rows),cols,physical),text([(r*30.+6.,[(c*50.+5.,c*50.+20.)for c in range(cols)])for r in range(rows)]))
    # 全宽/分组表头、需rowspan的前导表头与不连续子列。
    body=[(r*30.+6.,[(5.,20.),(55.,70.),(105.,120.),(155.,170.)])for r in range(1,5)]
    for header in [[(5.,195.)],[(5.,90.),(105.,195.)],[(5.,20.),(105.,170.)],[(5.,20.)],[(180.,195.),(5.,20.)]]:
        add(inp(5,4),text([(6.,header)]+body))
        add(inp(5,4,'lines'),text([(6.,header)]+body))
    # 紧邻稠密行拒绝，真横线保留；subset可续行、相等/非subset不可合并。
    full=[(5.,20.),(55.,70.),(105.,120.)]
    for partial in [full,[(5.,20.)],[(55.,70.),(105.,120.)],[],[(130.,145.)]]:
        for gap in [0.,4.,4.000001,5.]:
            for line in [False,True]:
                ti=inp();
                if line:ti=replace(ti,drawing_lines=(NativeTableRule((0.,14.+gap/2-.01,150.,14.+gap/2+.01),.02,'horizontal'),))
                add(ti,text([(6.,full),(14.+gap,partial),(45.,full),(75.,full)]))
    # 60%锚点/间隔、过界/非单调token、反向/退化表框及重复row_index。
    for gap in [0.,1.,1.499999,1.5,1.500001,10.]:
        for bad in range(6):
            defs=[]
            for r in range(5):defs.append((r*30.+6.,[(5.,20.),(20.+(gap if r<bad else 10.),45.),(75.,90.)]))
            add(inp(5,3),text(defs))
    t=text([(r*30.+6.,full)for r in range(4)])
    for b in [(150.,120.,0.,0.),(0.,0.,0.,120.),(-1.,-1.,149.,119.)]:add(replace(inp(),table_bbox=b),t)
    add(inp(),replace(t,rows=tuple(replace(r,row_index=0)for r in t.rows)))
    add(inp(),text([(6.,full),(36.,list(reversed(full))),(66.,full)]))
    # 既有实际旧PDF原语/字形输入也执行三路旧候选，不在此进程加载Rust。
    pdf=json.loads((ROOT/'build/migration/native-table-vector-pdf.json').read_text(encoding='utf-8'));real=[]
    def from_request(r):
        v=r['table_input'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],(),tuple(NativeTableRule(**q)for q in v['drawing_lines']),tuple(NativeTableRectangle(**q)for q in v['rectangles']))
        t=r['text'];txt=NativeTableText(tuple(NativeTableGlyph(**g)for g in t['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**tok)for tok in row['tokens']))))for row in t['rows']),t['median_glyph_width'],t['median_glyph_height'])
        return ti,txt
    for ref in pdf['real']:
        i=len(cases);add(*from_request(pdf['cases'][ref['case_index']]['request']));real.append(ref|dict(case_index=i))
    for _ in range(180):
        rows,cols=rng.randrange(2,8),rng.randrange(2,6);defs=[]
        for r in range(rows):
            boxes=[(c*50.+5.+rng.uniform(-3,3),c*50.+20.+rng.uniform(-3,3))for c in range(cols)if rng.random()>.1]
            if rng.random()<.05:boxes.reverse()
            defs.append((r*30.+6.+rng.uniform(-2,2),boxes))
        add(inp(rows,cols,rng.choice(['none','lines','stripes','both'])),text(defs))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(path,dict(sources={source+p:digest((ROOT/(source+p)).read_bytes())for p in ['text_grid.py','candidate.py','geometry.py','contracts.py']},cases=cases,real=real))
    print(f'Frozen {len(cases)} old text-grid generation cases including {len(real)} actual PDF/angle inputs')
if __name__=='__main__':main()
