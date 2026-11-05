"""Execute the contiguous original final-page prefix through formula/index separation."""
import ast,copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PreparedPage,_AxisLine
    p=q['page'];size=p.get('page_size',p.get('source',{}).get('page_size'));rs=p.get('drawing_lines',p.get('source',{}).get('drawing_lines',[]))
    page=_PreparedPage(page_size=tuple(size),remaining_lines=lines(p['remaining_lines']),table_bboxes=[tuple(b)for b in p.get('table_bboxes',[])],drawing_lines=[_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in rs],fixed_blocks=copy.deepcopy(p.get('fixed_blocks',[])),canonical_formula_geometry=p.get('canonical_formula_geometry',False),canonical_formula_source_lines=lines(p.get('canonical_formula_source_lines',[])),formula_candidate_lines=lines(p.get('formula_candidate_lines',[])),formula_ink_bboxes=[tuple(b)for b in p.get('formula_ink_bboxes',[])])
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py';fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)and n.name=='_finalize_prepared_page');stop=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='title_container_bboxes'for t in n.targets));body=ast.Module(body=copy.deepcopy(fn.body[:stop]),type_ignores=[]);ast.fix_missing_locations(body);scope=dict(vars(old),prepared=page,page_index=q.get('page_index',0));exec(compile(body,str(path),'exec'),scope)
    return clean(dict(formula_blocks=scope['formula_blocks'],index_blocks=scope['index_blocks'],semantic_lines=[project(l)for l in scope['semantic_lines']],remaining_lines=[project(l)for l in scope['remaining_lines']],canonical_formula_source_lines=[project(l)for l in page.canonical_formula_source_lines]))

def main():
    target=ROOT/'build/migration/native-final-formula.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);p=q['page']
        for key in ['remaining_lines','formula_candidate_lines','canonical_formula_source_lines']:
            if key in p:p[key]=[project(l)for l in lines(p[key])]
        c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-formula-input.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if c['request'].get('dense_only'):continue
        q=c['request'];add('input/'+c['label'],dict(page=dict(q,remaining_lines=q['lines'])))
    refs=json.loads((ROOT/'build/migration/native-reference-document-canonical.json').read_text(encoding='utf-8'))
    for i,c in enumerate(refs['cases']):
        if i%35:continue
        for j,p in enumerate(c.get('expected',{}).get('pages',[])):add('references/'+c['label']+'/'+str(j),dict(page=p,page_index=j))
    seed=dict(page_size=[400.,600.],remaining_lines=[line('Original complete body paragraph',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])]+[dict(line('x = y + z (1)',[130.,190.,370.,200.],20,font='Math'),style_scale_repaired=True)],fixed_blocks=[],table_bboxes=[],drawing_lines=[])
    for mode in ['canonical-same','canonical-different','canonical-empty','canonical-exclude','styles','candidate','semantic','index-heading','fallback-index','index-container','ink','ink-remote','ink-rotated']:
        p=copy.deepcopy(seed)
        if mode.startswith('canonical')or mode=='styles':
            p['canonical_formula_geometry']=True;p['canonical_formula_source_lines']=copy.deepcopy(p['remaining_lines']);p['remaining_lines'][-1]['style_scale_repaired']=False
            if mode=='canonical-different':p['canonical_formula_source_lines'][-1]['bbox']=[140.,190.,370.,200.]
            if mode=='canonical-empty':p['canonical_formula_source_lines']=[]
            if mode=='canonical-exclude':p['remaining_lines'][0]['semantic_type']='paragraph_title'
        if mode=='candidate':p['formula_candidate_lines']=[p['remaining_lines'].pop()];p['formula_candidate_lines'][0]['formula_candidate_only']=True
        if mode=='semantic':p['remaining_lines'][-1]['semantic_type']='paragraph_title'
        if mode.startswith('index')or mode=='fallback-index':
            p['remaining_lines'] += [line('Contents',[30.,330.,130.,340.],30)]+[line(f'{i}. Original section ........ {i+2}',[30.,355.+i*18,370.,365.+i*18],31+i)for i in range(4)]
            if mode=='fallback-index':p['remaining_lines']=p['remaining_lines'][:-5]+p['remaining_lines'][-4:]
            if mode=='index-container':p['fixed_blocks']=[dict(type='image',bbox=[20.,320.,380.,450.])]
        if mode.startswith('ink'):p['formula_ink_bboxes']=[[128.,190.,130.,200.]]if mode!='ink-remote'else[[20.,190.,30.,200.]]
        if mode=='ink-rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            for l in p['remaining_lines']:l.update(angle=180,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(p['page_size']),180))
        add('final/'+mode,dict(page=p))
    live=json.loads((ROOT/'build/migration/native-formula-runtime.json').read_text(encoding='utf-8'))
    for c in live['cases']:add(c['label'],dict(page=dict(c['request'],remaining_lines=c['request']['lines'])))
    write_json(target,dict(cases=cases));print(len(cases),'original final-page formula prefixes;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
