"""Execute untouched old page statements, with the real document geometry prefix."""
import ast,copy,json
from dataclasses import asdict
from types import SimpleNamespace as NS
from common import ROOT,write_json,digest
from export_native_document_source_prefix import reference as prefix
from export_native_geometry import load_plan,evidence
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_form_runtime import typed
from export_native_page_containers import project_maps

def reference(q,post_text=False,footnotes=False,full=False,chain=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    pfx=prefix(q);plan=load_plan(pfx['plan']);result=[]
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_prepare_page_source')
    split=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='unclaimed_lines'for t in n.targets))
    end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='remaining_lines'for t in n.targets)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='restore_leading_inline_fractions')
    if post_text:end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='restore_native_cjk_bands')
    if footnotes:end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='early_footnote_groups'for t in n.targets))
    if full:end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='_classify_page_auxiliary_text')
    def block(nodes):
        b=ast.Module(body=copy.deepcopy(nodes),type_ignores=[]);ast.fix_missing_locations(b);return compile(b,str(path),'exec')
    before=block(fn.body[:split]);after=block(fn.body[split:end+1])
    previous_grid=None
    for i,p in enumerate(pfx['pages']):
        raw=copy.deepcopy(p['source'])
        if chain:
            raw['page_index']=i
            if previous_grid is None:raw.pop('previous_table_grid',None)
            else:raw['previous_table_grid']=clean(asdict(previous_grid))
        s=typed(raw);g=p['geometry'];maps=lambda k,v:{e['char_idx']:tuple(e[v])for e in g.get(k,[])}
        scope=dict(vars(old));scope.update(source=s,page_index=i,geometry_plan=plan,tight_bboxes=maps('tight_bboxes','bbox'),origins=maps('origins','origin'),repaired_char_bboxes=None,style_lines=evidence(p.get('style_lines',[]),'style'),link_lines=evidence(p.get('link_lines',[]),'link'),table_header_separator_bboxes={tuple(b)for b in pfx['header_separator_bboxes'][i]},_owned_script_inputs=None)
        exec(before,scope)
        keys=['rule_code_blocks','table_annotation_blocks','table_blocks','code_blocks','form_image_blocks','graphic_blocks','raster_image_blocks','vector_formula_blocks','caption_graphics','early_index_blocks']
        fixed=sum((scope[k]for k in keys),[]);claims=scope['claimed_line_indices'];remaining=[l for l in s.lines if l.source_index not in claims]
        grid=_trailing_table_continuation_grid(s,NS(fixed_blocks=fixed,remaining_lines=remaining))
        exec(after,scope)
        if full:grid=_trailing_table_continuation_grid(s,scope['prepared'])
        previous_grid=grid
        raw['lines']=[project(l)for l in s.lines];raw=project_maps(raw,s)
        out=dict(source=raw,strong_graphic_bboxes=scope['strong_graphic_bboxes'],claimed_line_indices=sorted(claims),remaining_lines=[project(l)for l in scope['remaining_lines']],table_bboxes=scope['table_bboxes'],next_table_grid=None if grid is None else dict(page_index=grid.page_index,angle=grid.angle,positions=grid.positions,tolerance=grid.tolerance),selected_form_bboxes=scope['form_bboxes'],active_form_bboxes=scope['active_form_bboxes'],reclaimed_line_indices=sorted(scope['reclaimed']),source_index_start=scope['next_source_index'],style_lines=[asdict(l)for l in scope['style_lines']],link_lines=[asdict(l)for l in scope['link_lines']])
        for k in keys:out[{'table_annotation_blocks':'annotation_blocks','early_index_blocks':'index_blocks'}.get(k,k)]=scope[k]
        out['fixed_blocks']=sum((scope[k]for k in keys),[]);result.append(clean(out))
        if post_text:result[-1]['canonical_formula_source_lines']=[project(l)for l in scope['canonical_formula_source_lines']]
        if footnotes:result[-1]['page_footnote_groups']=[sorted(g)for g in scope['early_footnote_groups']]
        if full:
            prepared=scope['prepared']
            for k in ['script_lines','formula_candidate_lines','formula_ink_bboxes','table_body_profile_lines','bullet_paths']:result[-1][k]=clean(getattr(prepared,k))
    return clean(dict(pages=result,plan=pfx['plan'],header_separator_bboxes=pfx['header_separator_bboxes']))

def main():
    target=ROOT/'build/migration/native-document-pages-canonical.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q)
        for p in q['pages']:
            for k in ['lines','chars','drawing_lines','path_infos','image_bboxes','form_bboxes','signature_bboxes']:p['source'].setdefault(k,[])
            p['source']['lines']=[project(l)for l in typed(p['source']).lines]
        c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except ValueError as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    def wrap(s):return dict(source=s,geometry=dict(chars=s.get('chars',[]),tight_bboxes=[],origins=[],loose_bboxes=[]))
    f=json.loads((ROOT/'build/migration/native-precontainers.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        add('classifier/'+c['label'],dict(pages=[wrap(c['request']['source'])]))
        if c['label']=='contact/positive':add('contact/second-page',dict(pages=[wrap(dict(c['request']['source'],lines=[])),wrap(c['request']['source'])]))
    for name in ['native-caption-images','native-image-chain','native-vector-formulas']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if c['label'].startswith('continuous/'):continue
            s=c['request']if name=='native-caption-images'else c['request']['source'];add(name+'/'+c['label'],dict(pages=[wrap(s)]))
    f=json.loads((ROOT/'build/migration/native-document-source-prefix.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if c['label'].startswith('native-geometry-plan-proofs/')or c['label']in ['live-form-geometry-document','marginals/repeat','marginals/connected-table']:add(c['label'],c['request'])
    write_json(target,dict(cases=cases,pipeline_sha256=digest(path.read_bytes())if (path:=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py')else None));print(len(cases),'original document-to-calibrated-page chains;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
