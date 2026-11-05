"""Original continuous non-text chain, with full builders and real source side effects."""
import copy,json
from dataclasses import fields,replace
from types import SimpleNamespace as NS
from common import ROOT,write_json
from export_form_runtime import normalized_reference,typed
from export_native_page_containers import project_maps
from export_native_inline_clusters import clean
def reference(raw,vector=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import graphics as g
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _build_caption_supported_graphics
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.index_blocks import _extract_index_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.code_blocks import _build_code_blocks,_build_rule_delimited_code_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_aligned_numeric_tables,_detect_table_candidates
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_materialization import _materialize_table_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _bbox_overlap_in_smaller as small,_bbox_overlap_in_first as first,_bbox_area as area
    out=[]
    for raw in normalized_reference(raw):
        raw=copy.deepcopy(raw)
        for key in ['lines','chars','drawing_lines','path_infos','image_bboxes','form_bboxes','signature_bboxes']:raw.setdefault(key,[])
        s=typed(raw);keys=[f.name for f in fields(type(s.lines[0]))if not f.name.startswith('_')]if s.lines else[];before=[[copy.deepcopy(getattr(l,k))for k in keys]for l in s.lines]
        indexes,_=_extract_index_blocks(s.lines,s.page_size,s.image_bboxes+s.signature_bboxes,require_heading=True,require_explicit_heading=True);index_ids={l.source_index for l in s.lines if l.semantic_type=='index'};caps,cap_ids=_build_caption_supported_graphics(s);cap_boxes=[b['bbox']for b in caps];protected=cap_ids|{l.source_index for l in s.lines if l.semantic_type is not None};analysis=replace(s,lines=[l for l in s.lines if l.source_index not in protected]);forms=[b for b in g._select_form_image_bboxes(s)if not any(small(b,c)>=.8 for c in cap_boxes)]
        strong=g._detect_strong_graphic_bboxes(analysis)+cap_boxes;numeric=_detect_aligned_numeric_tables(analysis,list(s.image_bboxes));strong=[b for b in strong if not any(small(b,c.bbox)>=.5 and area(b)>=.25*area(c.bbox)for c in numeric)];rule,rc=_build_rule_delimited_code_blocks(analysis,forms+strong+s.image_bboxes+s.signature_bboxes)
        cs=[c for c in _detect_table_candidates(analysis,strong+[b['bbox']for b in rule])if not any(g._form_supersedes_nested_bbox(f,c.bbox)for f in forms)];tables,annotations,claimed=_materialize_table_blocks(s,cs)
        for b in tables:b['_native_numeric_grid']=g._graphic_members_are_numeric_grid([l for l in s.lines if first(l.bbox,b['bbox'])>=.95])
        claimed|=cap_ids|index_ids|rc;tb=[b['bbox']for b in tables];active=[f for f in forms if not any(small(f,b)>=.5 for b in tb)];code,cc=_build_code_blocks(analysis,tb+active+strong+s.image_bboxes+s.signature_bboxes,claimed);claimed|=cc;form_blocks,fc=g._build_form_image_blocks(analysis,active,claimed);claimed|=fc;graphic,gc=g._build_graphic_like_blocks(analysis,tb+active+[b['bbox']for b in rule+code]+cap_boxes,claimed,strong);claimed|=gc;raster,ras=g._build_raster_image_blocks(analysis,tables+rule+code+form_blocks+graphic+caps,claimed);claimed|=ras
        vector_blocks=[]
        if vector:
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.formulas import _build_vector_formula_blocks
            vector_blocks,vc=_build_vector_formula_blocks(analysis,tables+rule+code+form_blocks+graphic+raster+caps,claimed);claimed|=vc
        for r,l,prev in zip(raw['lines'],s.lines,before):
            for k,previous in zip(keys,prev):
                if getattr(l,k)!=previous:r[k]=clean(getattr(l,k))
        raw=project_maps(raw,s);remaining=[r for r in raw['lines']if r['source_index']not in claimed];fixed=rule+annotations+tables+code+form_blocks+graphic+raster+vector_blocks+caps+indexes;grid=_trailing_table_continuation_grid(s,NS(fixed_blocks=fixed,remaining_lines=[l for l in s.lines if l.source_index not in claimed]));out.append(clean(dict(source=raw,strong_graphic_bboxes=strong,table_blocks=tables,annotation_blocks=annotations,claimed_line_indices=sorted(claimed),remaining_lines=remaining,fixed_blocks=fixed,table_bboxes=tb,next_table_grid=None if grid is None else dict(page_index=grid.page_index,angle=grid.angle,positions=grid.positions,tolerance=grid.tolerance),rule_code_blocks=rule,code_blocks=code,index_blocks=indexes,selected_form_bboxes=forms,active_form_bboxes=active,form_image_blocks=form_blocks,graphic_blocks=graphic,raster_image_blocks=raster,caption_graphics=caps)))
        if vector:out[-1]['vector_formula_blocks']=clean(vector_blocks)
    return dict(pages=out)
def main():
    target=ROOT/'build/migration/native-document-images.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    for name in ['native-caption-images','native-image-chain']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text())
        for c in f['cases']:
            s=c['request']if name=='native-caption-images'else c['request']['source'];q=dict(sources=[s]);item=dict(label=name+'/'+c['label'],request=copy.deepcopy(q))
            try:item['expected']=reference(q['sources'])
            except ValueError as error:item.update(error='invalid_input',old_exception=type(error).__name__)
            cases.append(item)
    f=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text());cases.append(dict(label='actual-form-document',request=f['request'],expected=reference(f['request']['sources'])));write_json(target,dict(sources=f['sources'],cases=cases));print(len(cases),'continuous document container scenes')
if __name__=='__main__':main()
