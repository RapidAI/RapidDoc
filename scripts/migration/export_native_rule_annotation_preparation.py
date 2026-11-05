"""Freeze original caption-row clipping/collection and annotation records with real core assembly."""
import copy,random
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def objects(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _VisualRow,_Fragment
    return [_VisualRow([_Fragment(**dict(f,bbox=tuple(f['bbox']),local_bbox=tuple(f['local_bbox'])))for f in r['fragments']],r['center_y'],tuple(r['bbox']),r.get('visual_row_id'))for r in raw]
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_annotations import _collect_caption_rows,_build_table_annotation
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_union_many,_bbox_union
    rows=objects(q['rows']);rule=_bbox_union_many(q['rule_bboxes']);core=[rows[i]for i in q['core_row_indices']];ids={f.line_index for r in core for f in r.fragments};bounds=_bbox_union(rule,_bbox_union_many([r.bbox for r in core]));caption=SimpleNamespace(source_index=q['caption_source_index'])if q.get('caption_source_index')is not None else None;captions=_collect_caption_rows(rows,caption,rule,q['median_height']);annotations=[]
    for kind,selected,excluded in [('caption',captions,bounds),('footnote',[rows[i]for i in q.get('footnote_row_indices',[])],None)]:
        a=_build_table_annotation(kind,selected,excluded_line_indices=ids,excluded_local_bbox=excluded)
        if a is not None:a=asdict(a);a['line_bboxes']=[[k,b]for k,b in a['line_bboxes'].items()];annotations.append(a)
    return clean(dict(rule_bbox=rule,core_local_bbox=bounds,core_line_indices=ids,caption_rows=[asdict(r)for r in captions],annotations=annotations))
def fragment(id,b,local=None,text='words'):return dict(text=text,bbox=list(b),local_bbox=list(local if local is not None else b),line_index=id,visual_row_id=None)
def row(fs,visual=None):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_union_many,_bbox_center_y
    return dict(fragments=fs,center_y=sum(_bbox_center_y(f['local_bbox'])for f in fs)/len(fs),bbox=_bbox_union_many([f['local_bbox']for f in fs]),visual_row_id=visual)
def base():
    rows=[row([fragment(0,[50.,70.,130.,80.]),fragment(1,[150.,70.,220.,80.])]),row([fragment(2,[50.,85.,180.,95.])]),row([fragment(3,[50.,110.,100.,120.]),fragment(4,[150.,110.,200.,120.])]),row([fragment(5,[50.,200.,160.,210.])])]
    return dict(rows=rows,rule_bboxes=[[40.,100.,240.,100.],[40.,190.,240.,190.]],core_row_indices=[2],median_height=10.,caption_source_index=0,footnote_row_indices=[3])
def main():
    target=ROOT/'build/migration/native-rule-annotation-preparation.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q=None):q=clean(copy.deepcopy(q or base()));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('base')
    for seed in [None,-1,0,1,2,3,5,999]:q=base();q['caption_source_index']=seed;add('seed-'+str(seed),q)
    for x in [19.99999,20.,20.00001,259.99999,260.,260.00001]:
        q=base();q['rows'][0]=row([fragment(0,[x-5.,70.,x+5.,80.])]);add('corridor-centre-'+str(x),q)
    for y in [44.99999,45.,45.00001]:q=base();q['rows'][0]=row([fragment(0,[50.,y,130.,y+10.])]);add('gap-'+str(y),q)
    for y in [79.99999,80.,80.00001]:q=base();q['rows']=q['rows'][:1]+q['rows'][2:];q['rows'][0]=row([fragment(0,[50.,y-10.,130.,y])]);q['core_row_indices']=[1];q['footnote_row_indices']=[2];add('caption-last-gap-'+str(y),q)
    q=base();q['rows'][0]=row([fragment(0,[300.,70.,330.,80.]),fragment(1,[50.,70.,100.,80.])]);add('seed-fragment-outside-other-inside',q)
    for bottom in [99.99999,100.,100.00001]:
        q=base();q['rows'][0]=row([fragment(0,[50.,70.,100.,bottom])]);add('exclude-top-'+str(bottom),q)
    for x in [37.49999,37.5,37.50001]:
        q=base();q['rows'][0]=row([fragment(0,[x-50.,90.,x,110.])]);add('local-overlap-five-percent-'+str(x),q)
    q=base();q['rows'][0]=row([fragment(0,[500.,700.,600.,710.],[50.,70.,100.,80.])]);add('local-original-coordinates-differ',q)
    q=base();q['rows'][0]=row([fragment(0,[150.,70.,200.,80.]),fragment(0,[50.,70.,100.,80.])]);add('duplicate-source-union',q)
    q=base();q['rows'][0]=row([fragment(-2,[50.,70.,100.,80.]),fragment(-1,[150.,70.,200.,80.])]);q['caption_source_index']=-2;add('signed-source',q)
    q=base();q['rows'][3]=row([fragment(3,[50.,200.,100.,210.]),fragment(5,[120.,200.,150.,210.])]);add('footnote-core-member-exclusion',q)
    q=base();q['core_row_indices']=[0,2];add('caption-core-member-exclusion',q)
    q=base();q['footnote_row_indices']=[3,3];add('duplicate-footnote-row',q)
    for reverse in [False,True]:
        q=base();fs=[fragment(8,[0,0,10,10],[50,70,100,80]),fragment(9,[0.,0.,10.,10.],[50.,70.,100.,80.])]
        if reverse:fs.reverse()
        q['rows'][0]=row(fs);q['caption_source_index']=fs[0]['line_index'];add('ties-original-type-'+str(reverse),q)
    rng=random.Random(261117)
    for i in range(100):
        q=base();r=q['rows'][rng.randrange(4)];rng.shuffle(r['fragments']);f=rng.choice(r['fragments']);f['local_bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-20.,20.]);add('geometry-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Complete old caption collection/row clipping/annotation building; footnote row selection explicit, no automatic footnote collection.',cases=cases));print('Frozen',len(cases),'rule annotation cases;',sum(bool(c['expected']['caption_rows'])for c in cases),'caption-row positives;',sum(bool(c['expected']['annotations'])for c in cases),'annotation positives')
if __name__=='__main__':main()
