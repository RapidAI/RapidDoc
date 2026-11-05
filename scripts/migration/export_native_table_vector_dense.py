"""追加完整矢量路线的部分续行/真纵隔断及alias/残留窄带硬门。"""
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_table_vector_grid import plain
def main():
    path=ROOT/'build/migration/native-table-vector-dense.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableText,NativeTableTextRow,NativeTableToken,NativeTableGlyph
    cases=[]
    def make(x=(0.,50.,100.,150.),y=(0.,30.,60.,90.),vertical=1.,partial=None):
        lines=[NativeTableRule((0.,v-.1,x[-1],v+.1),.2,'horizontal')for v in y]
        lines+=[NativeTableRule((v-.1,0. if i in [0,len(x)-1]else 30.*(1.-vertical),v+.1,y[-1]),.2,'vertical')for i,v in enumerate(x)]
        gs=[];rs=[]
        rows=[(0,6.,(0,1,2)),(1,36.,(0,1,2)),(2,66.,(0,1,2))]
        if partial is not None:rows=[(0,4.,(0,1,2)),(8,12.,partial),(9,21.,(0,1,2))]+rows[1:]
        for r,yy,cols in rows:
            tokens=[];ids=[]
            for c in cols:
                i=len(gs);b=(8.+50*c,yy,18.+50*c,yy+6.);gs.append(NativeTableGlyph(i,i,'X',b,r));tokens.append(NativeTableToken('X',b,(i,),(i,)));ids.append(i)
            rs.append(NativeTableTextRow(r,(0.,yy,150.,yy+6.),tuple(tokens),tuple(ids)))
        return NativeTableInput((0.,0.,x[-1],y[-1]),(500.,500.),0,(),tuple(lines),()),NativeTableText(tuple(gs),tuple(rs),10.,10.)
    def add(t,text):
        d=[];result=old.build_vector_candidates(t,text,d)
        cases.append(dict(request=dict(table_input=asdict(t),text=asdict(text)),expected=plain(dict(candidates=[asdict(c)for c in result],diagnostics=d,html=[serialize_candidate_html(c)for c in result]))))
    for vertical in [0.,.79,1.]:
        for partial in [(),(0,)]:add(*make(vertical=vertical,partial=partial))
    add(*make(x=(0.,5.,10.,50.,100.)))
    add(*make(x=(0.,5.,100.)))
    add(*make(y=(0.,5.,30.,60.,90.)))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/vector.py'
    write_json(path,dict(sources={src:digest((ROOT/src).read_bytes())},cases=cases))
    print(f'Frozen {len(cases)} extra dense/track-gate pipeline cases')
if __name__=='__main__':main()
