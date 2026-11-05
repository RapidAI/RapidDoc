"""冻结旧规范轨道→拓扑→独立物理证据→文字/质量/普通HTML，拒绝覆写。"""
import copy,math,random,statistics
from dataclasses import asdict
from common import ROOT,digest,write_json
def plain(v):
    if isinstance(v,dict):return {k:plain(x)for k,x in v.items()}
    if isinstance(v,set):return sorted(v)
    if isinstance(v,(list,tuple)):return[plain(x)for x in v]
    return v
def main():
    target=ROOT/'build/migration/native-table-vector-grid.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    cases=[];rng=random.Random(4427)
    def text(rows,cols,missing=(),extra=()):
        gs=[];rs=[]
        for r in range(rows):
            glyphs=[];tokens=[]
            for c in range(cols):
                if (r,c)in missing:continue
                i=len(gs);b=(c*50.+5.,r*30.+5.,c*50.+12.,r*30.+15.);s=f'{r}{c}'
                g=NativeTableGlyph(i,i,s,b,r);gs.append(g);glyphs.append(i);tokens.append(NativeTableToken(s,b,(i,),(i,)))
            if glyphs:rs.append(NativeTableTextRow(r,(0.,r*30.+5.,cols*50.,r*30.+15.),tuple(tokens),tuple(glyphs)))
        gs.extend(extra)
        return NativeTableText(tuple(gs),tuple(rs),7.,10.)
    def tracks(rows=2,cols=2,line=True):
        x=[c*50. for c in range(cols+1)];y=[r*30. for r in range(rows+1)]
        rules=[old._MergedRule('horizontal',v,0.,x[-1])for v in y]+[old._MergedRule('vertical',v,0.,y[-1])for v in x]
        return old._VectorTracks(.5,x[-1],rules,[old._CanonicalTrack(v,(v,))for v in x],[old._CanonicalTrack(v,(v,))for v in y],x,y,7.5,line,rows==1 and cols>=2,cols==1 and rows>=2,rows,cols)
    def add(t,txt=None,label=None,seed=None):
        txt=text(t.rows,t.cols)if txt is None else txt
        d={}if seed is None else copy.deepcopy(seed)
        request=dict(tracks=asdict(t),text=asdict(txt),evidence_label=label or('line_grid'if t.is_line_grid else'rect_grid'),diagnostics=copy.deepcopy(d))
        topo=old._build_vector_topology(t,txt,d)
        candidate=None if topo is None else old._materialize_vector_candidate(t,topo,txt,request['evidence_label'],d)
        cases.append(dict(request=request,expected=plain(dict(topology=None if topo is None else asdict(topo),candidate=None if candidate is None else asdict(candidate),diagnostics=d,html=None if candidate is None else serialize_candidate_html(candidate)))))
    # 完整/单行/单列、逻辑退化与正文覆盖。
    for rows,cols in [(1,2),(1,3),(2,1),(3,1),(2,2),(3,3),(4,3)]:
        for line in [False,True]:
            t=tracks(rows,cols,line);add(t);add(t,text(rows,cols,missing=[(r,c)for r in range(rows)for c in range(cols)]));add(t,seed={'keep':{'a':1},'first_rejection_gate':'old'})
            for row in range(rows):add(t,text(rows,cols,missing=[(row,c)for c in range(cols)]))
    # 单行/列硬物理门：0.8边界、字号高度、字形跨隔断。
    for rows,cols in [(1,3),(3,1)]:
        for cov in [0.,.19999999999999996,.2,math.nextafter(.8,0),.8,math.nextafter(.8,1),1.]:
            for orientation in ['horizontal','vertical']:
                t=tracks(rows,cols);t.rules[:]=[old._MergedRule(r.orientation,r.coordinate,r.start,r.start+(r.end-r.start)*cov)if r.orientation==orientation else r for r in t.rules];add(t)
        for minimum in [0.,.1,30.,37.5,math.nextafter(37.5,math.inf),50.]:
            t=tracks(rows,cols);object.__setattr__(t,'narrow_empty_threshold',minimum);add(t)
        b=(45.,5.,55.,15.)if rows==1 else(5.,25.,15.,35.)
        add(tracks(rows,cols),text(rows,cols,extra=[NativeTableGlyph(99,99,'cross',b,0)]))
    # 各隔断的覆盖、合并、歧义、L形拒绝，顺序与原实现一致。
    for cov in [0.,.1,.19999999999999996,.2,.5,.7999999999999999,.8,.9,1.]:
        for orientation in ['vertical','horizontal']:
            t=tracks(3,3)
            t.rules[:]=[old._MergedRule(r.orientation,r.coordinate,r.start,r.start+(r.end-r.start)*cov)if r.orientation==orientation and r.coordinate in [30.,50.]else r for r in t.rules];add(t)
    t=tracks();t.rules[:]=[r for r in t.rules if r.coordinate not in [30.,50.]];add(t)
    # alias严格位置无覆盖但原坐标有证据；空白行不能借别名恢复越门。
    for axis in ['x','y']:
        for missing_row in [None,0,1]:
            t=tracks(3,3);cs=t.canonical_x_tracks if axis=='x'else t.canonical_y_tracks
            oldc=cs[1].coordinate;cs[1]=old._CanonicalTrack(oldc,(oldc-2.,oldc,oldc+2.))
            orientation='vertical'if axis=='x'else'horizontal'
            t.rules[:]=[old._MergedRule(r.orientation,oldc+2.,r.start,r.end)if r.orientation==orientation and r.coordinate==oldc else r for r in t.rules]
            add(t,text(3,3,missing=[]if missing_row is None else[(missing_row,c)for c in range(3)]))
    # 首项inclusive文字占用，空行的物理补证与字形穿行。
    for y in [0.,30.,60.,90.,-1.,91.]:
        t=tracks(3,3);g=NativeTableGlyph(100,100,'edge',(5.,y-5.,12.,y+5.),0)
        add(t,text(3,3,extra=[g]))
    # 100字时98%捕获能通过builder，但单行/单列要求100%。
    for rows,cols in [(1,2),(2,1)]:
        t=tracks(rows,cols);gs=[]
        for i in range(100):
            b=(5.,5.,12.,15.)if i<99 else(1000.,1000.,1007.,1010.)
            gs.append(NativeTableGlyph(i,i,'X',b,0))
        token=NativeTableToken('X'*100,(5.,5.,12.,15.),tuple(range(100)),tuple(range(100)))
        txt=NativeTableText(tuple(gs),(NativeTableTextRow(0,(5.,5.,12.,15.),(token,),tuple(range(100))),),7.,10.)
        add(t,txt)
    for _ in range(260):
        rows,cols=rng.randrange(2,7),rng.randrange(2,7);t=tracks(rows,cols,rng.choice([False,True]))
        rules=[]
        for r in t.rules:
            if rng.random()<.12:continue
            if rng.random()<.3:r=old._MergedRule(r.orientation,r.coordinate,r.start,r.end*rng.choice([.1,.5,.8,.9,1.]))
            rules.append(r)
        t.rules[:]=rules
        missing=[(r,c)for r in range(rows)for c in range(cols)if rng.random()<.2]
        add(t,text(rows,cols,missing))
    means=[]
    for values in [[1.],[.5,1.],[.5,math.nextafter(.5,1),1.],[.8]*100]+[[rng.uniform(.5,1.)for _ in range(rng.randrange(2,500))]for _ in range(100)]:
        means.append(dict(values=values,expected=statistics.mean(values)))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(target,dict(sources={source+p:digest((ROOT/(source+p)).read_bytes())for p in ['vector.py','candidate.py','contracts.py','geometry.py']},cases=cases,means=means))
    print(f'Frozen {len(cases)} canonical-grid topology/materialization cases +{len(means)} exact mean cases')
if __name__=='__main__':main()
