"""Freeze full original non-prepared rule candidate expansion; no component verdict injection."""
import copy,json
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_rule_annotation_preparation import objects as rows_objects,base
from export_native_bar_graphics import objects as lines_objects
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _expand_rule_table_candidate
    rows=rows_objects(q['rows']);lines=lines_objects(q['lines']);caption=SimpleNamespace(source_index=q['caption_source_index'])if q.get('caption_source_index')is not None else None
    c=_expand_rule_table_candidate([SimpleNamespace(bbox=tuple(b))for b in q['rule_bboxes']],[rows[i]for i in q['core_row_indices']],rows,lines,tuple(q['page_size']),q.get('angle',0),q['median_height'],caption,q.get('has_possible_table_notes',True));c=asdict(c)
    for a in c['annotations']:a['line_bboxes']=[[k,b]for k,b in a['line_bboxes'].items()]
    return clean(dict(candidate=c))
def basic():
    q=base();q.pop('footnote_row_indices');q['rows'][3]['fragments'][0]['text']='Note: measured values';q['lines']=[dict(text=f['text'],bbox=f['bbox'],source_index=f['line_index'],angle=0,chars=[],effective_height=10.)for r in q['rows']for f in r['fragments']];q['page_size']=[400.,600.];q['angle']=0;q['has_possible_table_notes']=True;return clean(q)
def main():
    target=ROOT/'build/migration/native-rule-candidate-expansion.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('basic-full',basic())
    for angle in [0,90,180,270,15]:
        for notes in [False,True]:
            q=basic();q['angle']=angle;q['has_possible_table_notes']=notes
            for r in q['rows']:
                for f in r['fragments']:f['bbox']=_rotate_bbox_from_upright(tuple(f['local_bbox']),tuple(q['page_size']),angle)
            for l in q['lines']:l['angle']=angle;l['bbox']=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),angle)
            add(f'angle-{angle}-notes-{notes}',q)
    for name in ['native-rule-annotation-preparation']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            q=copy.deepcopy(c['request']);q.pop('footnote_row_indices',None);q['lines']=[dict(text=f['text'],bbox=f['bbox'],source_index=f['line_index'],angle=0,chars=[],effective_height=max(.1,f['bbox'][3]-f['bbox'][1]))for r in q['rows']for f in r['fragments']];q['page_size']=[400.,600.];q['angle']=0;add('prior-'+c['label'],q)
    for notes in [False,True]:
        q=basic();q['has_possible_table_notes']=notes;q['caption_source_index']=None;add('no-caption-'+str(notes),q)
    q=basic();q['core_row_indices']=[0,2];add('core-owns-caption',q)
    q=basic();q['core_row_indices']=[2,3];add('core-owns-note',q)
    for angle in [0,90,180,270]:
        q=basic();q['angle']=angle;q['page_size']=[400,600];q['has_possible_table_notes']=False
        for r in q['rows']:
            r['bbox']=[int(x)for x in r['bbox']]
            for f in r['fragments']:f['local_bbox']=[int(x)for x in f['local_bbox']];f['bbox']=_rotate_bbox_from_upright(tuple(f['local_bbox']),tuple(q['page_size']),angle)
        q['rule_bboxes']=[[int(x)for x in b]for b in q['rule_bboxes']];add('integer-coordinate-'+str(angle),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Full original _expand_rule_table_candidate ordinary non-prepared/non-owned path; upstream rule/core/caption facts explicit.',cases=cases));print('Frozen',len(cases),'full rule expansions;',sum(bool(c['expected']['candidate']['annotations'])for c in cases),'with annotations')
if __name__=='__main__':main()
