"""额外冻结5%歧义、L形拓扑、开口边/空行和别名门，不覆写主参考。"""
import copy
from dataclasses import asdict,replace
from common import ROOT,digest,write_json
from export_native_table_vector_grid import plain
def main():
    path=ROOT/'build/migration/native-table-vector-grid-edges.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableText,NativeTableTextRow,NativeTableToken,NativeTableGlyph
    cases=[]
    def add(rows,cols,*,lshape=False,ambiguous=False,unverified=False,alias_blank=False,rect=False,blank=False,cross=False):
        x=[50.*c for c in range(cols+1)];y=[30.*r for r in range(rows+1)]
        cx=[old._CanonicalTrack(v,(v,))for v in x];cy=[old._CanonicalTrack(v,(v,))for v in y]
        rules=[old._MergedRule('horizontal',v,0.,x[-1])for v in y]+[old._MergedRule('vertical',v,0.,y[-1])for v in x]
        if ambiguous:
            orientation='vertical'if rows==1 else'horizontal';coordinate=x[1]if rows==1 else y[1]
            rules=[replace(r,end=r.end*.5)if r.orientation==orientation and r.coordinate==coordinate else r for r in rules]
        if lshape:
            rules=[replace(r,start=30.)if r.orientation=='vertical'and r.coordinate==50. else replace(r,start=50.)if r.orientation=='horizontal'and r.coordinate==30. else r for r in rules]
        if unverified:rules=[r for r in rules if r.orientation!='horizontal' or r.coordinate!=30.]
        if alias_blank:
            cy[1]=old._CanonicalTrack(y[1],(y[1],y[1]+2.))
            rules=[replace(r,coordinate=r.coordinate+2.)if r.orientation=='horizontal'and r.coordinate==y[1]else r for r in rules]
        glyphs=[];rs=[]
        for row in range(rows):
            if blank and row==1:continue
            ts=[];ids=[]
            for col in range(cols):
                i=len(glyphs);b=(col*50.+5.,row*30.+5.,col*50.+12.,row*30.+15.);s='X'
                glyphs.append(NativeTableGlyph(i,i,s,b,row));ids.append(i);ts.append(NativeTableToken(s,b,(i,),(i,)))
            rs.append(NativeTableTextRow(row,(0.,row*30.+5.,x[-1],row*30.+15.),tuple(ts),tuple(ids)))
        if cross:glyphs.append(NativeTableGlyph(999,999,'cross',(5.,25.,12.,50.),0))
        t=old._VectorTracks(.5,x[-1],rules,cx,cy,x,y,7.5,not rect,rows==1 and cols>=2,cols==1 and rows>=2,rows,cols)
        text=NativeTableText(tuple(glyphs),tuple(rs),7.,10.);d={};label='rect_grid'if rect else'line_grid'
        topo=old._build_vector_topology(t,text,d);candidate=None if topo is None else old._materialize_vector_candidate(t,topo,text,label,d)
        cases.append(dict(request=dict(tracks=asdict(t),text=asdict(text),evidence_label=label),expected=plain(dict(topology=None if topo is None else asdict(topo),candidate=None if candidate is None else asdict(candidate),diagnostics=d,html=None if candidate is None else serialize_candidate_html(candidate)))))
    for n in [20,21,22]:
        add(1,n,ambiguous=True);add(n,1,ambiguous=True)
    add(2,2,lshape=True)
    for kwargs in [{},{'rect':True},{'unverified':True},{'alias_blank':True},{'cross':True}]:add(3,3,blank=True,**kwargs)
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/vector.py'
    write_json(path,dict(sources={source:digest((ROOT/source).read_bytes())},cases=cases))
    print(f'Frozen {len(cases)} extra grid/physical-gate cases')
if __name__=='__main__':main()
