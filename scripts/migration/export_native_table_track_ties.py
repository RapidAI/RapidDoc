"""追加旧整数索引set遍历与等距最近物理线参考，不覆写主参考。"""
from dataclasses import asdict
import random,sys
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-track-ties.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import vector as old
    cases=[];sets=[];rng=random.Random(871)
    for i,j in [(1,32),(2,33),(1,64),(31,32),(0,8),(5,100),(1,256)]:
        for reverse in [False,True]:
            rules=[old._MergedRule('vertical',200.,0.,100.)for _ in range(max(i,j)+1)]
            rules[i]=old._MergedRule('horizontal',2. if reverse else 0.,0.,100.)
            rules[j]=old._MergedRule('horizontal',0. if reverse else 2.,0.,100.)
            r=dict(positions=[1.,2.,50.,100.],glyph_centers=[],threshold=3.,extent=100.,evidence_halo=0.,minimum_track_count=2,rules=rules,orientation='horizontal',separator_extent=100.,snap_tolerance=2.,preserve_double_boundary=False,collapse_narrow_bands=False,collapse_leading_edge=True)
            t,c,n=old._canonicalize_axis_tracks(**r)
            unique=old._canonical_tracks_are_unique(t,rules,'horizontal',2.)
            r['rules']=[asdict(x)for x in rules]
            cases.append(dict(request=r,expected=dict(tracks=[asdict(x)for x in t],alias_conflict=c,outer_collapse_count=n,unique=unique)))
    inputs=[[1,32],[0,8],list(range(5000)),list(range(0,5000,32))]+[sorted(rng.sample(range(5000),rng.randrange(1,900)))for _ in range(200)]
    for values in inputs:
        # 原函数使用set comprehension按枚举顺序插入，不是set(list)的可能批预分配。
        sets.append(dict(input=values,expected=list({i for i in values})))
    source='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/vector.py'
    write_json(path,dict(python=sys.version,sources={source:digest((ROOT/source).read_bytes())},cases=cases,sets=sets))
    print(f'Frozen {len(cases)} physical tie cases +{len(sets)} CPython3.14 integer sets')
if __name__=='__main__':main()
