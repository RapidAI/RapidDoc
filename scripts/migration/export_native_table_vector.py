"""冻结实际旧vector.py裁剪/吸附/连接/轨道推断，不覆写既有参考。"""
from dataclasses import asdict
import random
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/native-table-vector.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle
    cases=[];rng=random.Random(89354)
    def add(lines=(),rectangles=(),box=(10.,20.,110.,120.),angle=0,snap=.5,join=1.,halo=0.,drawing=True,rect=True):
        ti=NativeTableInput(box,(500.,500.),angle,(),tuple(NativeTableRule(tuple(b),w,o)for b,w,o in lines),tuple(NativeTableRectangle(tuple(b),n,f,s)for b,n,f,s in rectangles))
        request=dict(table_input=asdict(ti),snap_tolerance=snap,join_gap=join,evidence_halo=halo,include_drawing=drawing,include_rectangles=rect)
        w,h=old.table_local_size(old.normalize_bbox(box),old.normalize_angle(angle)) if old.normalize_bbox(box) else (0.,0.)
        # prepare对无效表框只用于片段跳过分支；其后推断空集。
        prep=old._prepare_rule_evidence(ti,snap,join,w,h,include_drawing=drawing,include_rectangles=rect,evidence_halo=halo)
        supported=old._infer_grid_tracks(prep.rules,snap,w,h,prune_unsupported_horizontal=True)
        cases.append(dict(request=request,expected=dict(fragments=[asdict(r)for r in prep.fragments],rules=[asdict(r)for r in prep.rules],inferred=prep.inferred,supported=supported)))
    base=[([10.,20.,110.,20.2],1.,'vertical'),([10.,70.,110.,70.2],0.,'horizontal'),([10.,119.8,110.,120.],99.,'horizontal'),([10.,20.,10.2,120.],1.,'horizontal'),([60.,20.,60.2,120.],1.,'vertical'),([109.8,20.,110.,120.],1.,'vertical')]
    for angle in [0,90,180,270,-90,450,45,359,-360]:
        for halo in [0.,1.,4.]:
            for drawing,rect in [(True,False),(False,True),(True,True),(False,False)]:
                add(base,[([12.,22.,58.,68.],5,True,False),([62.,72.,108.,118.],5,False,True)],angle=angle,halo=halo,drawing=drawing,rect=rect)
    for box in [(110.,120.,10.,20.),(10.,20.,10.,120.),(0.,0.,100.,100.)]:add(base,box=box)
    for snap in [0.,.1,.5,1.,2.]:
        for join in [0.,.5,1.,2.]:
            add(base+[([10.,45.,35.,45.1],1.,'horizontal'),([35.5,45.,59.8,45.1],1.,'horizontal'),([10.,45.4,60.,45.6],1.,'horizontal'),([30.,80.,40.,80.2],1.,'horizontal')],snap=snap,join=join)
    # 零面积线会被旧normalize_bbox跳过；halo只在完全不相交时生效。
    for halo in [0.,.5,1.,2.]:
        add(base+[([5.,19.,115.,19.2],1.,'horizontal'),([5.,18.,15.,22.],1.,'vertical'),([0.,70.,120.,70.],1.,'horizontal')],halo=halo)
    for area in [84.9999,85.,85.0001,100.]:
        for n,f,s in [(5,True,False),(5,False,False),(4,True,True),(5,False,True)]:
            add(rectangles=[([10.,20.,110.,20.+area],n,f,s)])
    # 连续聚类是与当前均值比较，不是相邻点链；同坐标重复/嵌套/反向框。
    for offsets in [[0.,.4,.8],[0.,.5,1.],[0.,0.,0.],[0.,1e-10,1e-9]]:
        add([([10.,50.+x,110.,50.1+x],1.,'horizontal')for x in offsets])
    for _ in range(180):
        lines=[];rectangles=[]
        for __ in range(rng.randrange(1,30)):
            x,y=rng.uniform(-5,125),rng.uniform(5,135)
            if rng.random()<.5:b=[x,y,x+rng.uniform(.01,100),y+rng.uniform(.01,2)]
            else:b=[x,y,x+rng.uniform(.01,2),y+rng.uniform(.01,100)]
            if rng.random()<.2:b=[b[2],b[3],b[0],b[1]]
            lines.append((b,rng.random()*4,rng.choice(['horizontal','vertical'])))
        for __ in range(rng.randrange(8)):
            x,y=rng.uniform(0,100),rng.uniform(0,100)
            rectangles.append(([x,y,x+rng.uniform(.1,80),y+rng.uniform(.1,80)],rng.choice([4,5,6]),rng.choice([True,False]),rng.choice([True,False])))
        add(lines,rectangles,angle=rng.choice([0,90,180,270]),snap=rng.choice([0.,.2,.5,1.]),join=rng.choice([0.,1.,2.]),halo=rng.choice([0.,2.]))
    # 旧5000片段门与halo分支：输入原语有界，矩形展开可以超过5000片段。
    add(rectangles=[([20.,30.,30.,40.],5,True,False)]*1251)
    sources=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'+p for p in ['vector.py','geometry.py','contracts.py']]+['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in sources},cases=cases))
    print(f'Frozen {len(cases)} old physical evidence/track cases: {target}')

if __name__=='__main__':main()
