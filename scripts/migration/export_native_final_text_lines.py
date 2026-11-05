"""Contiguous existing Python final-page prefix, stopping before block assembly."""
import ast,copy,json,sys
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_page_structure import BODY
def typed_lines(raw):
    ls=lines(raw)
    for l in ls:
        if l.wrapped_title_parts is not None:l.wrapped_title_parts=tuple(typed_lines(list(l.wrapped_title_parts)))
    return ls
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PreparedPage,_AxisLine,_DocumentBodyProfile,_DocumentTitleProfile,_TitleStylePrototype
    p=q['page'];sz=p.get('page_size',p.get('source',{}).get('page_size'));rules=p.get('drawing_lines',p.get('source',{}).get('drawing_lines',[]))
    page=_PreparedPage(page_size=tuple(sz),remaining_lines=typed_lines(p['remaining_lines']),table_bboxes=[tuple(b)for b in p.get('table_bboxes',[])],drawing_lines=[_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in rules],fixed_blocks=copy.deepcopy(p.get('fixed_blocks',[])),canonical_formula_geometry=p.get('canonical_formula_geometry',False),canonical_formula_source_lines=typed_lines(p.get('canonical_formula_source_lines',[])),formula_candidate_lines=typed_lines(p.get('formula_candidate_lines',[])),formula_ink_bboxes=[tuple(b)for b in p.get('formula_ink_bboxes',[])],page_footnote_groups=[set(g)for g in p.get('page_footnote_groups',[])],reference_regions=[tuple(b)for b in p.get('reference_regions',[])],numbered_references=p.get('numbered_references',False),bracketed_reference_numbers=p.get('bracketed_reference_numbers',False))
    def body(key):
        value=q.get(key)
        return None if value is None else _DocumentBodyProfile(**dict(value,regular_fonts=frozenset(tuple(f)for f in value['regular_fonts'])))
    title=q.get('document_title_profile');title=None if title is None else _DocumentTitleProfile(prototypes=tuple(_TitleStylePrototype(**p)for p in title['prototypes']))
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py';fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)and n.name=='_finalize_prepared_page');stop=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='text_blocks'for t in n.targets));module=ast.Module(body=copy.deepcopy(fn.body[:stop]),type_ignores=[]);ast.fix_missing_locations(module);scope=dict(vars(old),prepared=page,page_index=q.get('page_index',0),canonical_body_profile=body('canonical_body_profile'),document_body_profile=body('document_body_profile'),document_title_profile=title);exec(compile(module,str(path),'exec'),scope)
    return clean(dict(formula_blocks=scope['formula_blocks'],index_blocks=scope['index_blocks'],semantic_lines=[project(l)for l in scope['semantic_lines']],remaining_lines=[project(l)for l in scope['remaining_lines']],canonical_formula_source_lines=[project(l)for l in page.canonical_formula_source_lines],page_footnote_groups=page.page_footnote_groups))
def chain(q):
    from export_native_reference_chain import reference as references
    out=references(q);out['text_pages']=[reference(dict(page=p,page_index=i,document_body_profile=out.get('body_profile'),canonical_body_profile=out.get('canonical_body_profile'),document_title_profile=out.get('title_profile')))for i,p in enumerate(out['pages'])];return out
def main():
    if '--canonical-chain'in sys.argv:
        target=ROOT/'build/migration/native-final-text-lines-chain-canonical.json'
        if target.exists():raise FileExistsError(target)
        from loguru import logger
        logger.remove();f=json.loads((ROOT/'build/migration/native-final-text-lines-chain.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            for k in ['expected','error','old_exception']:c.pop(k,None)
            try:c['expected']=chain(c['request'])
            except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        write_json(target,f);print(len(f['cases']),'canonical typed title-profile final text chains;',sum('error'in c for c in f['cases']),'exceptions');return
    target=ROOT/'build/migration/native-final-text-lines.json';chains=ROOT/'build/migration/native-final-text-lines-chain.json'
    if target.exists()or chains.exists():raise FileExistsError('immutable final text-line references exist')
    from loguru import logger
    logger.remove();cases=[];document=[]
    def add(label,q):
        q=copy.deepcopy(q);p=q['page']
        for key in ['remaining_lines','canonical_formula_source_lines','formula_candidate_lines']:
            if key in p:p[key]=[project(l)for l in typed_lines(p[key])]
        c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-final-formula.json').read_text(encoding='utf-8'))
    for i,c in enumerate(f['cases']):
        if i%13==0 or c['label'].startswith(('final/','live-')):add('formula/'+c['label'],dict(c['request'],document_body_profile=BODY))
    for name in ['native-page-structure','native-final-title-rules-typed','native-semantic-merges']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for i,c in enumerate(f['cases']):
            if i%11 or 'expected'not in c:continue
            q=c['request'];ls=q['lines'];p=dict(page_size=q['page_size'],remaining_lines=ls,table_bboxes=q.get('table_bboxes',[]),fixed_blocks=[dict(type='image',bbox=b)for b in q.get('image_bboxes',[])],drawing_lines=[])
            add(name+'/'+c['label'],dict(page=p,page_index=q.get('page_index',0),document_body_profile=q.get('document_body_profile',BODY)))
    f=json.loads((ROOT/'build/migration/native-reference-chain.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        q=copy.deepcopy(c['request']);entry=dict(label=c['label'],request=q)
        try:entry['expected']=chain(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:entry.update(error='invalid_input',old_exception=type(e).__name__)
        document.append(entry)
    write_json(target,dict(cases=cases));write_json(chains,dict(cases=document));print(len(cases),'continuous existing Python final text-line cases;',len(document),'raw-source document chains;',sum('error'in c for c in cases+document),'exceptions')
if __name__=='__main__':main()
