"""Freeze physical-table branch orchestration only; other page detector producers remain pending."""
import copy,json
from types import SimpleNamespace as NS
from common import ROOT,digest,write_json
from export_closed_rule_pipeline import setup,dumped
from export_native_tiled_tables import old_source
def reference(q,ordinary=False,supplemental=False,numeric=False,final_producers=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_rules as rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_filled_grid import _detect_filled_grid_table_candidates,_detect_tiled_fill_table_candidates
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_smaller,_expand_bbox,_rotate_bbox_to_upright
    raw=q['source'];source=old_source(dict(raw,chars=raw.get('chars',[])));ex=q.get('excluded_bboxes',[]);filled=_detect_filled_grid_table_candidates(source.path_infos,source.page_size,ex);tiled=_detect_tiled_fill_table_candidates(source,ex);filled=[c for c in filled if not any(_bbox_overlap_in_smaller(c.bbox,t.bbox)>=.9 for t in tiled)]+tiled;bounds=[c.bbox for c in filled];closed=[]
    for angle in sorted({l.get('angle',0)for l in raw.get('lines',[])}):
        _,_,size,_,lines,rows,axis,em,_=setup(dict(q,angle=angle,median_height=None));excluded=[_expand_bbox(_rotate_bbox_to_upright(tuple(b),size,angle),min(2.5*em,.01*size[0])if any(tuple(b)==t.bbox for t in tiled)else 2.5*em)for b in [*ex,*bounds]];obstacles=[_rotate_bbox_to_upright(tuple(b),size,angle)for b in ex+raw.get('image_bboxes',[])+raw.get('signature_bboxes',[])+raw.get('form_bboxes',[])];forms=[_rotate_bbox_to_upright(tuple(b),size,angle)for b in raw.get('form_bboxes',[])];previous=NS(**raw['previous_table_grid'])if raw.get('previous_table_grid')is not None else None
        if ordinary:closed+=rules._build_rule_table_candidates(rows,lines,size,angle,em,axis,path_infos=source.path_infos,excluded_bboxes=excluded,defer_materialization=True)
        closed+=rules._build_closed_rule_grid_candidates(rows,lines,size,angle,em,axis,excluded,previous_table_grid=previous,page_index=raw.get('page_index'),leading_obstacles=obstacles,form_bboxes=forms)
    if ordinary:
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _exclude_prose_spanning_rule_drafts
        from export_native_bar_graphics import objects
        source.lines=objects(raw.get('lines',[]));closed=_exclude_prose_spanning_rule_drafts(source,closed);merged=rules._merge_owned_table_candidates(closed)
    else:merged=rules._merge_table_candidates(closed)
    merged=[c for c in merged if not any(_bbox_overlap_in_smaller(c.bbox,b)>=.2 for b in bounds)];base=[*filled,*merged]
    if supplemental:
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_open_vertical_tables,_detect_short_caption_ruled_tables,_detect_shaded_header_tables,_detect_captioned_blank_answer_tables,_detect_page_clipped_blank_form_tables
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_first
        from export_native_bar_graphics import objects
        source.lines=objects(raw.get('lines',[]));source.image_bboxes=raw.get('image_bboxes',[]);source.form_bboxes=raw.get('form_bboxes',[]);base+=_detect_shaded_header_tables(source,ex)
        if numeric:
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_open_grid import detect_partial_rule_grids
            partial=detect_partial_rule_grids(source,ex)
        else:partial=[]
        recovered=[c for c in partial+_detect_open_vertical_tables(source,ex)if not any(_bbox_overlap_in_first(c.core_bbox,b.core_bbox or b.bbox)>=.98 for b in base)];base=[c for c in base if not any(_bbox_overlap_in_first(c.core_bbox or c.bbox,r.core_bbox)>=.9 for r in recovered)]
        if numeric:
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_year_header_numeric_tables,_detect_ruled_metric_pair_tables,_prefer_complete_table_replacements
            years=_detect_year_header_numeric_tables(source,ex)
            if final_producers:
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_aligned_numeric_tables
                aligned=[c for c in _detect_aligned_numeric_tables(source,ex)if not any(_bbox_overlap_in_smaller(old.core_bbox or old.bbox,c.bbox)>=.9 and old.bbox[2]-old.bbox[0]>=.95*(c.bbox[2]-c.bbox[0])and old.bbox[3]-old.bbox[1]>=.85*(c.bbox[3]-c.bbox[1])for old in base+recovered)]
            else:aligned=[]
            replacements=years+_detect_ruled_metric_pair_tables(source,ex)+aligned;base,recovered,replacements=_prefer_complete_table_replacements(base,recovered,replacements);base+=replacements
        base+=recovered;existing=[c.core_bbox or c.bbox for c in base];base+=_detect_short_caption_ruled_tables(source,ex)+_detect_captioned_blank_answer_tables(source,ex)
        if numeric:
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_unruled_numeric_column_tables,_detect_captioned_pair_text_tables
            base+=_detect_unruled_numeric_column_tables(source,ex+existing)
            if final_producers:
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_captioned_mixed_numeric_tables
                base+=_detect_captioned_mixed_numeric_tables(source,ex+existing)
            base+=_detect_captioned_pair_text_tables(source,ex+existing)
        base+=_detect_page_clipped_blank_form_tables(source,ex+existing)
    return dict(candidates=[dumped(c)for c in sorted(base,key=lambda c:(c.bbox[1],c.bbox[0]))])
def main():
    target=ROOT/'build/migration/native-physical-table-page.json'
    if target.exists():raise FileExistsError(target)
    old=json.loads((ROOT/'build/migration/native-rule-page.json').read_text(encoding='utf-8'));cases=[]
    def add(label,q):cases.append(dict(label=label,request=q,expected=reference(q)))
    for c in old['cases']:add(c['label'],c['request'])
    from export_native_tiled_tables import base as tiled
    from export_native_filled_grid import base as filled
    for header in [False,True]:
        s=tiled(header=header);s.update(form_bboxes=[],image_bboxes=[],signature_bboxes=[])
        for l in s['lines']:l['angle']=0;l['effective_height']=10.
        add('tiled-'+str(header),dict(source=s,excluded_bboxes=[]))
    s=copy.deepcopy(old['cases'][0]['request']['source']);s['path_infos']=filled();s['chars']=[];add('filled-priority',dict(source=s,excluded_bboxes=[]))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_filled_grid.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Exact old filled/tiled priority + per-angle closed-rule candidate branch, local exclusions and merged physical candidates. General rule intervals/shaded/open/numeric/captioned producers and final page caption/tail policies are not included.',cases=cases));print('Frozen physical page',len(cases),'cases;',sum(bool(c['expected']['candidates'])for c in cases),'positive')
if __name__=='__main__':main()
