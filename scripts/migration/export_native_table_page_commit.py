"""Freeze continuous old table-preparation/commit stages; never replace recovery or inject candidates."""
import copy,json
from dataclasses import replace
from common import ROOT,digest,write_json
from export_native_table_page_policies import source as projected_source
from export_native_inline_clusters import clean
from export_native_bar_graphics import MUTABLE,snapshot

def typed(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PageSource
    r=copy.deepcopy(raw)
    for p in r.get('path_infos',[]):
        p.setdefault('rectangle_bboxes',[])
        if p.get('fill_rgba')is not None:p['fill_rgba']=tuple(p['fill_rgba'])
    ns=projected_source(r);ns.page_size=tuple(ns.page_size)
    for k in ['image_bboxes','form_bboxes','signature_bboxes']:setattr(ns,k,[tuple(b)for b in getattr(ns,k)])
    return _PageSource(**vars(ns))
def reference(q,include_code=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _detect_strong_graphic_bboxes,_graphic_members_are_numeric_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_aligned_numeric_tables,_detect_table_candidates
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_materialization import _materialize_table_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_smaller,_bbox_area,_bbox_overlap_in_first
    from types import SimpleNamespace as NS
    raw=copy.deepcopy(q['source'])
    for k in ['lines','chars','drawing_lines','path_infos','image_bboxes','signature_bboxes','form_bboxes']:raw.setdefault(k,[])
    s=typed(raw);before=[tuple(getattr(l,k)for k in MUTABLE)for l in s.lines];prior=set(q.get('prior_claimed_line_indices',[]));protected=prior|{l.source_index for l in s.lines if l.semantic_type is not None};analysis=replace(s,lines=[l for l in s.lines if l.source_index not in protected])
    strong=_detect_strong_graphic_bboxes(analysis);numeric=_detect_aligned_numeric_tables(analysis,list(s.image_bboxes));strong=[b for b in strong if not any(_bbox_overlap_in_smaller(b,c.bbox)>=.5 and _bbox_area(b)>=.25*_bbox_area(c.bbox)for c in numeric)]
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.code_blocks import _build_rule_delimited_code_blocks,_build_code_blocks
    excluded=strong+[tuple(b)for b in q.get('excluded_bboxes',[])]
    rule_code,rule_claimed=_build_rule_delimited_code_blocks(analysis,excluded+s.image_bboxes+s.signature_bboxes,prior)if include_code else([],set())
    separators={tuple(b)for b in q.get('table_header_separator_bboxes',[])};table_source=replace(analysis,drawing_lines=[r for r in s.drawing_lines if tuple(r.bbox)not in separators]);cs=_detect_table_candidates(table_source,excluded+[b['bbox']for b in rule_code])
    maps=lambda k:{int(i):b for i,b in(q.get(k)or{}).items()}
    tables,annotations,claimed=_materialize_table_blocks(s,cs,tight_bboxes=maps('tight_bboxes'),origins=maps('origins'),calibrated_char_bboxes=maps('calibrated_char_bboxes')or None)
    for b in tables:b['_native_numeric_grid']=_graphic_members_are_numeric_grid([l for l in s.lines if _bbox_overlap_in_first(l.bbox,b['bbox'])>=.95])
    claimed|=prior|rule_claimed
    filled_code,filled_claimed=_build_code_blocks(analysis,excluded+s.image_bboxes+s.signature_bboxes+[b['bbox']for b in tables],claimed)if include_code else([],set())
    claimed|=filled_claimed;remaining=[l for l in s.lines if l.source_index not in claimed];fixed=copy.deepcopy(q.get('prior_fixed_blocks',[]))+rule_code+annotations+tables+filled_code
    next=_trailing_table_continuation_grid(s,NS(fixed_blocks=fixed,remaining_lines=remaining))
    raw['lines']=snapshot(raw['lines'],s.lines,before)
    for r,l in zip(raw['lines'],s.lines):
        if r.get('explicit_section_title',False)!=l.explicit_section_title:r['explicit_section_title']=l.explicit_section_title
    remaining=[r for r,l in zip(raw['lines'],s.lines)if l.source_index not in claimed]
    result=dict(source=raw,strong_graphic_bboxes=strong,table_blocks=tables,annotation_blocks=annotations,claimed_line_indices=sorted(claimed),remaining_lines=remaining,fixed_blocks=fixed,table_bboxes=[b['bbox']for b in tables],next_table_grid=None if next is None else vars(next)if hasattr(next,'__dict__')else dict(page_index=next.page_index,angle=next.angle,positions=next.positions,tolerance=next.tolerance))
    if include_code:result.update(rule_code_blocks=rule_code,code_blocks=filled_code)
    return clean(result)
def main():
    target=ROOT/'build/migration/native-table-page-commit.json';pdf_target=ROOT/'build/migration/pdf-table-page-commit.json'
    if target.exists()or pdf_target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[];errors=[]
    def add(label,q):
        q=clean(copy.deepcopy(q))
        try:out=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=out))
    original=json.loads((ROOT/'build/migration/native-complete-table-candidates.json').read_text(encoding='utf-8'))
    for c in original['cases'][::13]:add(c['label'],dict(source=c['request']['source'],excluded_bboxes=c['request'].get('excluded_bboxes',[])))
    positives=[c for c in original['cases']if c['expected']['candidates']and c['request']['source'].get('chars')]
    base=copy.deepcopy(positives[0]['request']);base['source'].setdefault('page_index',0)
    for mode in ['all-excluded','protected-header','prior-claim','duplicate-id','prior-code-after','prior-footer-after','header-separators','all-protected','malformed-origins','front-matter','page-two','no-lines']:
        q=copy.deepcopy(base);s=q['source']
        if mode=='all-excluded':q['excluded_bboxes']=[[0.,0.,*s['page_size']]]
        if mode=='protected-header':s['lines'][0]['semantic_type']='header'
        if mode=='prior-claim':q['prior_claimed_line_indices']=[s['lines'][0]['source_index']]
        if mode=='duplicate-id':
            for l in s['lines'][:2]:l['source_index']=0
        if mode in ['prior-code-after','prior-footer-after']:q['prior_fixed_blocks']=[dict(type='footer'if mode=='prior-footer-after'else'code',bbox=[0.,s['page_size'][1]-20.,100.,s['page_size'][1]-10.],content='prior',angle=0)]
        if mode=='header-separators':q['table_header_separator_bboxes']=[r['bbox']for r in s['drawing_lines']]
        if mode=='all-protected':
            for l in s['lines']:l['semantic_type']='header'
        if mode=='malformed-origins':q['origins']={str(c.get('char_idx',i)):['bad']for i,c in enumerate(s['chars'])}
        if mode=='front-matter':s['page_index']=0
        if mode=='page-two':s['page_index']=1
        if mode=='no-lines':s['lines']=[]
        add(mode,q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['pipeline.py','graphics.py','table_detection.py','table_materialization.py','table_rules.py','table_text_styles.py','geometry.py']];meta=dict(sources={p:digest((ROOT/p).read_bytes())for p in refs})
    write_json(target,dict(meta,scope='Continuous table portion of page preparation after external semantic/container evidence; not the whole _prepare_page_source. Actual old detector/recovery/commit/claims/numeric-grid/last-band functions, no injected candidates or restored HTML.',cases=cases,errors=errors))
    pdf=[]
    for name in ['pdf-complete-table-candidates','pdf-strong-graphics-positive']:
        for c in json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']:
            s=c['request']['source'];q=dict(source=s,tight_bboxes={str(ch['char_idx']):ch.get('tight_bbox')for ch in s['chars']},origins={str(ch['char_idx']):ch.get('origin')for ch in s['chars']});pdf.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],request=q,expected=reference(q)))
    write_json(pdf_target,dict(meta,scope='59 actual PDF source queries through continuous detection and commit; image/Form boxes retain explicit original source evidence.',cases=pdf));print('Frozen table page',len(cases),'errors',len(errors),'source tables',sum(bool(c['expected']['table_blocks'])for c in cases),'PDF',len(pdf),'with tables',sum(bool(c['expected']['table_blocks'])for c in pdf),'next grids',sum(c['expected']['next_table_grid']is not None for c in pdf))
if __name__=='__main__':main()
