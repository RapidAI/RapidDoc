"""冻结旧vector.py轨道规范化/alias/双边界行为；不覆写参考。"""
from dataclasses import asdict
import random
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-tracks.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    rng=random.Random(6123);cases=[]
    def add(positions,centers=(),threshold=2.,extent=100.,halo=1.,minimum=3,rules=None,orientation='horizontal',separator=100.,snap=.5,preserve=False,narrow=True,leading=True):
        request=dict(positions=list(map(float,positions)),glyph_centers=list(map(float,centers)),threshold=threshold,extent=extent,evidence_halo=halo,minimum_track_count=minimum,rules=None if rules is None else[asdict(r)for r in rules],orientation=orientation,separator_extent=separator,snap_tolerance=snap,preserve_double_boundary=preserve,collapse_narrow_bands=narrow,collapse_leading_edge=leading)
        kwargs=request.copy();kwargs.update(rules=rules)
        tracks,conflict,count=old._canonicalize_axis_tracks(**kwargs)
        unique=True if rules is None else old._canonical_tracks_are_unique(tracks,rules,orientation,snap)
        cases.append(dict(request=request,expected=dict(tracks=[asdict(t)for t in tracks],alias_conflict=conflict,outer_collapse_count=count,unique=unique)))
    for positions in [[],[0],[0,100],[0,1,2,50,100],[0,.4,.8,50,99.5,100],[0,1.5,3.,4.5,50,100],[-.5,0,.5,50,99.5,100,100.5],[0,0,50,100],[100,50,0]]:
        for minimum in [0,2,3]:
            for centers in [[],[.25,1.,50.],[99.75]]:add(positions,centers,minimum=minimum)
    for starts in [[0.,1.,50.,100.],[0.,.5,50.,99.5,100.],[0.,1.8,50.,100.]]:
        for coords in [[0.,50.,100.],[.4,50.,100.],[0.,1.,50.,100.],[0.,99.5],[1.,99.]]:
            for preserve in [False,True]:
                for leading in [False,True]:
                    rs=[old._MergedRule('horizontal',c,0.,100.)for c in coords]
                    add(starts,rules=rs,preserve=preserve,leading=leading,snap=1.)
    for extent in [0.,1.,100.]:
        for halo in [0.,.5,2.]:
            add([0.,.4,.8,1.,50.,100.],extent=extent,halo=halo)
    for _ in range(220):
        positions=sorted([0.,100.]+[rng.uniform(-1,101)for _ in range(rng.randrange(1,25))])
        rs=[old._MergedRule(rng.choice(['horizontal','vertical']),c+rng.choice([0.,.2,.8]),rng.uniform(0,20),rng.uniform(30,100))for c in positions if rng.random()<.7]
        add(positions,[rng.uniform(0,100)for _ in range(rng.randrange(8))],threshold=rng.choice([.5,1.,2.,4.]),halo=rng.choice([0.,1.,2.]),minimum=rng.choice([2,3,4]),rules=rs,orientation=rng.choice(['horizontal','vertical','']),snap=rng.choice([0.,.5,1.]),preserve=rng.choice([False,True]),narrow=rng.choice([False,True]),leading=rng.choice([False,True]))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/vector.py'
    write_json(path,dict(sources={source:digest((ROOT/source).read_bytes())},cases=cases))
    print(f'Frozen {len(cases)} old canonical track cases')
if __name__=='__main__':main()
