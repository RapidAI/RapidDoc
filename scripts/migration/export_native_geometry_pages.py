"""Execute the untouched old pipeline statements from geometry repair through fractions."""
import ast,copy,json
from common import ROOT,write_json,digest
from export_native_geometry import load_plan,evidence,lines,plan_raw
from export_native_late_image_text import project
from export_native_table_page_policies import empty

def reference(q,post_text=False,full=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from export_form_runtime import typed
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and any(isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='image_containers'for t in x.targets)for x in n.body))
    start=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='unclaimed_lines'for t in n.targets))
    end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='remaining_lines'for t in n.targets)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='restore_leading_inline_fractions')
    if post_text:end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='restore_native_cjk_bands')
    if full:end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_classify_page_auxiliary_text')
    block=ast.Module(body=copy.deepcopy(fn.body[start:end+1]),type_ignores=[]);ast.fix_missing_locations(block)
    p=copy.deepcopy(q['prepared']);source=typed(p['source']);scope=dict(vars(old));scope.update(source=source,page_index=q['page_index'],geometry_plan=load_plan(q['plan'])if q['plan']is not None else None,repaired_char_bboxes=None,style_lines=evidence(q.get('style_lines',[]),'style'),link_lines=evidence(q.get('link_lines',[]),'link'),claimed_line_indices=set(p.get('claimed_line_indices',[])),table_bboxes=[tuple(b)for b in p['table_bboxes']],strong_graphic_bboxes=[tuple(b)for b in p['strong_graphic_bboxes']])
    for key in ['caption_graphics','form_image_blocks','graphic_blocks','raster_image_blocks']:scope[key]=p.get(key,[])
    if post_text or full:scope['vector_formula_blocks']=p.get('vector_formula_blocks',[])
    if full:
        scope.update(tight_bboxes={int(k):tuple(v)for k,v in q.get('tight_bboxes',{}).items()},origins={int(k):tuple(v)for k,v in q.get('origins',{}).items()},_owned_script_inputs=None)
        for key in ['rule_code_blocks','table_blocks','code_blocks','early_index_blocks']:scope[key]=p.get('index_blocks'if key=='early_index_blocks'else key,[])
        scope['table_annotation_blocks']=p.get('annotation_blocks',[])
    exec(compile(block,str(path),'exec'),scope)
    p['source']['lines']=[project(l)for l in source.lines];p['remaining_lines']=[project(l)for l in scope['remaining_lines']];p['reclaimed_line_indices']=sorted(scope['reclaimed']);p['source_index_start']=scope['next_source_index'];p['style_lines']=json.loads(json.dumps([__import__('dataclasses').asdict(l)for l in scope['style_lines']]));p['link_lines']=json.loads(json.dumps([__import__('dataclasses').asdict(l)for l in scope['link_lines']]))
    for key in ['caption_graphics','form_image_blocks','graphic_blocks','raster_image_blocks']:p[key]=scope[key]
    p['fixed_blocks']=list(p.get('prior_fixed_blocks',[]))
    for key in ['rule_code_blocks','annotation_blocks','table_blocks','code_blocks','form_image_blocks','graphic_blocks','raster_image_blocks','vector_formula_blocks','caption_graphics','index_blocks']:p['fixed_blocks']+=p.get(key,[])
    if post_text or full:p['canonical_formula_source_lines']=[project(l)for l in scope['canonical_formula_source_lines']]
    if full:
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
        prepared=scope['prepared']
        for k in ['script_lines','formula_candidate_lines','formula_ink_bboxes','table_body_profile_lines','bullet_paths']:p[k]=__import__('export_native_inline_clusters').clean(getattr(prepared,k))
        p['page_footnote_groups']=[sorted(g)for g in prepared.page_footnote_groups]
        grid=_trailing_table_continuation_grid(source,prepared);p['next_table_grid']=None if grid is None else dict(page_index=grid.page_index,angle=grid.angle,positions=grid.positions,tolerance=grid.tolerance)
    return p
def main():
    target=ROOT/'build/migration/native-geometry-pages.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);p=q['prepared'];p['source']['lines']=[project(l)for l in lines(p['source']['lines'])];p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])];cases.append(dict(label=label,request=q,expected=reference(q)))
    f=json.loads((ROOT/'build/migration/native-geometry.json').read_text(encoding='utf-8'))
    for c in f['resplit']:
        r=c['request'];p=dict(source=dict(empty(),page_size=r['page_size'],lines=r['lines']),remaining_lines=r['lines'],strong_graphic_bboxes=[],table_bboxes=[],claimed_line_indices=[])
        plan=dict(char_repairs=[dict(page_index=0,char_idx=x['char_idx'],source_bbox=[0.,0.,1.,1.],tight_bbox=x['bbox'],layout_bbox=x['bbox'],ink_bbox=x['bbox'],origin=[0.,0.],baseline=210.,advance=5.,em_height=10.,x_state='abnormal',y_state='healthy',confidence=.9)for x in r['visual_bboxes']])
        add('resplit/'+c['label'],dict(prepared=p,plan=plan,page_index=0,style_lines=r['style_lines'],link_lines=r['link_lines']))
        if c['label']=='split':
            p['graphic_blocks']=[dict(type='image',bbox=[20.,180.,235.,230.],angle=0,content='left'),dict(type='image',bbox=[280.,180.,480.,230.],angle=0,content='right')];add('resplit/labels-into-images',dict(prepared=p,plan=plan,page_index=0,style_lines=r['style_lines'],link_lines=r['link_lines']))
    for c in f['apply']:
        if c['label'].startswith('actual-form/')or c['label'].startswith('font/'):continue
        r=c['request'];p=dict(source=dict(empty(),page_size=[500.,600.],lines=r['lines']),remaining_lines=r['lines'],strong_graphic_bboxes=[],table_bboxes=[],claimed_line_indices=[]);add('repair/'+c['label'],dict(prepared=p,plan=r['plan'],page_index=r['page_index'],style_lines=[],link_lines=[]))
    pages=json.loads((ROOT/'build/migration/native-vector-document.json').read_text(encoding='utf-8'))
    for c in pages['cases']:
        if not c['label'].startswith('native-document-images/')or'expected'not in c:continue
        for i,p in enumerate(c['expected']['pages']):
            add(c['label']+f'/{i}',dict(prepared=p,plan=None,page_index=i,style_lines=[],link_lines=[]))
            if c['label'].endswith('actual-form-document'):add('actual-plan/'+str(i),dict(prepared=p,plan=f['actual_document_plan'],page_index=i,style_lines=[],link_lines=[]))
    write_json(target,dict(pipeline_sha256=digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py').read_bytes()),cases=cases));print(len(cases),'complete old post-container geometry sequences')
if __name__=='__main__':main()
