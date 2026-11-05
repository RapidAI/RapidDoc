"""Complete original _build_text_blocks, including every block refinement."""
import copy,json
from common import ROOT,write_json
from export_native_final_text_lines import typed_lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
STEPS=dict(prefix='_merge_short_same_baseline_prefix_blocks',spatial='_merge_spatial_text_components',list='_merge_list_intro_text_components',unterminated='_merge_unterminated_text_components',overlap='_merge_overlapping_same_line_text_blocks',math='_merge_inline_math_fragment_text_blocks',context='_merge_paragraph_formula_context_blocks')
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.assembly import _build_text_blocks
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    return clean(_build_text_blocks(typed_lines(q['lines']),[tuple(b)for b in q['table_bboxes']],tuple(q['page_size']),[_AxisLine(bbox=tuple(b['bbox']),orientation=b['orientation'],width=b['width'])for b in q['drawing_lines']],page_footnote_groups=[set(g)for g in q['page_footnote_groups']],page_index=q['page_index'],visual_bboxes=[tuple(b)for b in q['visual_bboxes']]))
def main():
    target=ROOT/'build/migration/native-text-blocks.json';steps_target=ROOT/'build/migration/native-text-block-merges.json'
    if target.exists()or steps_target.exists():raise FileExistsError('immutable full assembly references exist')
    from loguru import logger
    logger.remove();cases=[];steps=[]
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import assembly as old
    originals={kind:getattr(old,name)for kind,name in STEPS.items()};context=['']
    def wrap(kind):
        def call(bs,*args,**kw):
            request=clean(dict(blocks=bs,page_size=list(args[0])if args else[400.,600.],kind=kind));out=originals[kind](bs,*args,**kw);steps.append(dict(label=context[0]+'/'+kind,request=request,expected=clean(out)));return out
        return call
    for kind,name in STEPS.items():setattr(old,name,wrap(kind))
    def add(label,q):
        q=copy.deepcopy(q)
        for k,v in dict(page_size=[400.,600.],table_bboxes=[],drawing_lines=[],visual_bboxes=[],page_footnote_groups=[],page_index=0).items():q.setdefault(k,v)
        q['lines']=[project(l)for l in typed_lines(q['lines'])];c=dict(label=label,request=q);context[0]=label
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    try:
        for name in ['native-text-rows','native-text-footnotes']:
            for c in json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']:
                if name=='native-text-rows'and c['label'].startswith('direct/')is False:continue
                add(name+'/'+c['label'],c['request'])
        for c in json.loads((ROOT/'build/migration/native-final-text-lines.json').read_text(encoding='utf-8'))['cases']:
            if 'expected'not in c:continue
            p=c['request']['page'];state=c['expected'];add('final/'+c['label'],dict(lines=state['remaining_lines'],page_footnote_groups=state['page_footnote_groups'],page_size=p.get('page_size',p.get('source',{}).get('page_size')),table_bboxes=p.get('table_bboxes',[]),drawing_lines=p.get('drawing_lines',p.get('source',{}).get('drawing_lines',[])),visual_bboxes=[b['bbox']for b in p.get('fixed_blocks',[])if b.get('type')=='image'],page_index=c['request'].get('page_index',0)))
    finally:
        for kind,name in STEPS.items():setattr(old,name,originals[kind])
    write_json(target,dict(cases=cases));write_json(steps_target,dict(cases=steps));print(len(cases),'full assembly cases;',sum('error'in c for c in cases),'exceptions;',len(steps),'complete block-refinement replays')
if __name__=='__main__':main()
