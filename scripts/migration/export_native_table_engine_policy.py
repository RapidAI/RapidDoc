"""冻结实际旧引擎早期几何/相交原语计数与矢量诊断守卫；不替代完整候选执行。"""
import random,copy,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/('build/migration/native-table-engine-policy-stable.json'if '--stable'in sys.argv else'build/migration/native-table-engine-policy.json')
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import engine as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableText,NativeTableTextRow,NativeTableCandidate,NativeTableCell
    cases=[];rng=random.Random(11478)
    def candidate(rows=3,cols=3,source='vector_grid',score=1.,capture=1.,order=1.,issues=('evidence=rect_grid','ambiguous_separator_ratio=0.0000')):
        return NativeTableCandidate(source,rows,cols,tuple(NativeTableCell(r,c,1,1,(c*10.,r*10.,c*10.+10.,r*10.+10.),'X')for r in range(rows)for c in range(cols)),score,capture,1.,1.,1.,order,issues)
    def add(attempts=(),candidates=(),rows=3,bbox=(0.,0.,100.,100.),size=(200.,200.),boxes=((1.,1.,2.,2.),)):
        ti=NativeTableInput(bbox,size,0,(),tuple(NativeTableRule(b,.2,'horizontal')for b in boxes),())
        text=NativeTableText((),tuple(NativeTableTextRow(r,(0.,r*10.,100.,r*10.+5.),(),())for r in range(rows)),1.,1.)
        # 仅隔离早期输入守卫；旧_evaluate在此阶段之后固定返回native_text，不声称完整执行。
        saved=old.build_native_table_text;old.build_native_table_text=lambda _:None
        try:gate=old._evaluate_native_pdf_table(ti).first_rejection_gate
        finally:old.build_native_table_text=saved
        resolved=old._resolved_rect_undercount_candidate(candidates,attempts,text)
        expected=dict(primitive_count=old._table_primitive_count(ti),input_rejection_gate=None if gate=='native_text'else gate,
                      alias_affected_physical_blank_row=old._has_alias_affected_physical_blank_row(attempts),physical_row_undercount=old._has_physical_row_undercount(attempts),
                      minimum_columns=old._minimum_columns_after_row_undercount(attempts),column_consistent=[i for i,c in enumerate(candidates)if c.cols>=old._minimum_columns_after_row_undercount(attempts)],
                      attempted_line_rect_grid_conflict=old._has_attempted_line_rect_grid_conflict(attempts),resolved_rect_undercount=None if resolved is None else next(i for i,c in enumerate(candidates)if c is resolved))
        cases.append(dict(request=dict(table_input=asdict(ti),text=asdict(text),vector_attempts=copy.deepcopy(list(attempts)),candidates=[asdict(c)for c in candidates]),expected=expected))
    for bbox in [(0.,0.,100.,100.),(100.,100.,0.,0.),(0.,0.,0.,100.),(-.01,0.,100.,100.),(0.,0.,200.00001,100.)]:
        for size in [(200.,200.),(0.,200.),(-1.,200.)]:add(bbox=bbox,size=size)
    for b in [(100.,0.,101.,1.),(99.999999,0.,101.,1.),(5.,5.,5.,6.),(2.,2.,1.,1.)]:add(boxes=(b,))
    add(boxes=((1.,1.,2.,2.),)*5000);add(boxes=((1.,1.,2.,2.),)*5001);add(boxes=((300.,300.,301.,301.),)*5001)
    line=dict(evidence='line_grid',first_rejection_gate='physical_row_undercount',grid=dict(rows=2,cols=3),canonical_tracks=[{}]*4)
    rect=dict(evidence='rect_grid',grid=dict(rows=3,cols=3))
    for score in [.9499999999999999,.95,1.]:
        for capture,order in [(1.,1.),(.99,1.),(1.,.99)]:add([line,rect],[candidate(score=score,capture=capture,order=order)])
    add([line,rect],[candidate(),candidate()]);add([line,rect],[candidate(rows=4)])
    for revoked in [False,True]:
        d=copy.deepcopy(line)
        if revoked:d['first_rejection_gate']=None;d['track_hypotheses']=[line]
        for tracks in [[],[{}]*3,[{}]*4]:
            d['canonical_tracks']=tracks;add([d,rect],[candidate(cols=2),candidate(cols=3)])
    for evidence in ['line_grid','rect_grid']:
        for nested in [False,True]:
            for gate in ['empty_row','physical_row_undercount',None]:
                d=dict(evidence=evidence,first_rejection_gate=gate,empty_rows=[1],alias_affected_rows=[1])
                if nested:d=dict(evidence=evidence,track_hypotheses=[d])
                add([d])
    for _ in range(300):
        attempts=[]
        for __ in range(rng.randrange(5)):
            cols=rng.randrange(2,6);a=dict(evidence=rng.choice(['line_grid','rect_grid','other']),first_rejection_gate=rng.choice(['physical_row_undercount','empty_row',None]),grid=dict(rows=rng.randrange(2,6),cols=cols),canonical_tracks=[{}]*rng.choice([cols,cols+1]),empty_rows=rng.sample(range(5),2),alias_affected_rows=rng.sample(range(5),2))
            if rng.random()<.4:a['track_hypotheses']=[dict(first_rejection_gate=rng.choice(['physical_row_undercount','empty_row',None]),empty_rows=[0,1],alias_affected_rows=[1,2])]
            attempts.append(a)
        cs=[candidate(rows=rng.randrange(2,6),cols=rng.randrange(2,6),source=rng.choice(['vector_grid','text_grid']),score=rng.choice([.9,.95,1.]))for __ in range(rng.randrange(4))]
        add(attempts,cs,rows=rng.randrange(2,6))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/engine.py'
    write_json(path,dict(sources={src:digest((ROOT/src).read_bytes())},cases=cases))
    print(f'Frozen {len(cases)} old engine input/diagnostic-policy cases')
if __name__=='__main__':main()
