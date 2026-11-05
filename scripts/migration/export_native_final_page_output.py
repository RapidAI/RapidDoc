"""Execute every original _finalize_prepared_page statement, without parser substitutions."""
import ast,copy,inspect,json
from common import ROOT,write_json
import export_native_final_text_lines as original
from export_native_refined_block_chain import _bullet_paths
_runner=None
def reference(q):
    global _runner
    if _runner is None:
        tree=ast.parse(inspect.getsource(original.reference));fn=tree.body[0];fn.name='complete_reference'
        for node in ast.walk(fn):
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='_PreparedPage':node.keywords.append(ast.keyword(arg='bullet_paths',value=ast.parse("_bullet_paths(p.get('bullet_paths', []))",mode='eval').body))
        stop=next(n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='stop'for t in n.targets))
        stop.value=ast.parse('len(fn.body)',mode='eval').body
        module=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='module'for t in n.targets))
        fn.body.insert(module,ast.parse("fn.body[-1] = ast.Assign(targets=[ast.Name(id='model_list',ctx=ast.Store())],value=fn.body[-1].value)").body[0])
        result=next(n for n in fn.body if isinstance(n,ast.Return));result.value=ast.parse("clean(scope['model_list'])",mode='eval').body
        ast.fix_missing_locations(tree);scope=dict(original.__dict__,_bullet_paths=_bullet_paths);exec(compile(tree,__file__,'exec'),scope);_runner=scope['complete_reference']
    return _runner(q)
def main():
    targets=[ROOT/f'build/migration/native-final-page-output{s}.json'for s in ['', '-chain']]
    if any(t.exists()for t in targets):raise FileExistsError('immutable final page references exist')
    from loguru import logger
    logger.remove()
    for target,source in zip(targets,['native-final-refined-blocks-paths','native-final-refined-blocks-chain-paths']):
        f=json.loads((ROOT/f'build/migration/{source}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if 'expected'not in c:continue
            try:
                if source.endswith('chain-paths'):
                    e=c['expected']
                    for i,(p,b)in enumerate(zip(e['pages'],e['block_pages'])):b['model_list']=reference(dict(page=p,page_index=i,document_body_profile=e.get('body_profile'),canonical_body_profile=e.get('canonical_body_profile'),document_title_profile=e.get('title_profile')))
                else:c['expected']['model_list']=reference(c['request'])
            except(ValueError,TypeError,KeyError,IndexError,AttributeError,ZeroDivisionError)as err:c.pop('expected',None);c.update(error='invalid_input',old_exception=type(err).__name__)
        write_json(target,f);print(target.name,len(f['cases']),sum('error'in c for c in f['cases']),'exceptions')
if __name__=='__main__':main()
