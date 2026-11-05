"""Execute contiguous original final-page statements through split_local_list_blocks."""
import ast,copy,inspect,json,sys
from types import SimpleNamespace
from common import ROOT,write_json
import export_native_final_text_lines as original
_runner=None
def _bullet_paths(raw):return tuple(SimpleNamespace(**dict(p,bbox=tuple(p['bbox']),rectangle_bboxes=tuple(tuple(b)for b in p.get('rectangle_bboxes',[]))))for p in raw)
def reference(q):
    global _runner
    if _runner is None:
        tree=ast.parse(inspect.getsource(original.reference));fn=tree.body[0];fn.name='refined_reference'
        for node in ast.walk(fn):
            if isinstance(node,ast.Call)and isinstance(node.func,ast.Name)and node.func.id=='_PreparedPage':node.keywords.append(ast.keyword(arg='bullet_paths',value=ast.parse("_bullet_paths(p.get('bullet_paths', []))",mode='eval').body))
            if isinstance(node,ast.Compare)and isinstance(node.left,ast.Attribute)and node.left.attr=='id':
                for c in node.comparators:
                    if isinstance(c,ast.Constant)and c.value=='text_blocks':c.value='absolute_blocks'
        result=next(n for n in fn.body if isinstance(n,ast.Return));payload=result.value.args[0]
        payload.keywords.append(ast.keyword(arg='refined_text_blocks',value=ast.Subscript(value=ast.Name(id='scope',ctx=ast.Load()),slice=ast.Constant(value='text_blocks'),ctx=ast.Load())))
        ast.fix_missing_locations(tree);scope=dict(original.__dict__,_bullet_paths=_bullet_paths);exec(compile(tree,str(ROOT/'scripts/migration/export_native_refined_block_chain.py'),'exec'),scope);_runner=scope['refined_reference']
    return _runner(q)
def main():
    suffix='-paths'if '--paths'in sys.argv else'';target=ROOT/f'build/migration/native-final-refined-blocks{suffix}.json';chain=ROOT/f'build/migration/native-final-refined-blocks-chain{suffix}.json'
    if target.exists()or chain.exists():raise FileExistsError('immutable refined block references exist')
    from loguru import logger
    logger.remove();f=json.loads((ROOT/'build/migration/native-final-text-blocks.json').read_text(encoding='utf-8'));g=json.loads((ROOT/'build/migration/native-final-text-blocks-chain.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        c['request']['refine_blocks']=True
        if 'expected'in c:
            try:c['expected'].update(reference(c['request']))
            except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.pop('expected',None);c.update(error='invalid_input',old_exception=type(e).__name__)
    for c in g['cases']:
        c['request']['refine_blocks']=True
        if 'expected'not in c:continue
        e=c['expected']
        try:
            for i,(p,b)in enumerate(zip(e['pages'],e['block_pages'])):b.update(reference(dict(page=p,page_index=i,document_body_profile=e.get('body_profile'),canonical_body_profile=e.get('canonical_body_profile'),document_title_profile=e.get('title_profile'))))
        except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as err:c.pop('expected',None);c.update(error='invalid_input',old_exception=type(err).__name__)
    if '--paths'in sys.argv:
        from export_native_title_layout import line
        from export_native_final_text_lines import typed_lines
        from export_native_late_image_text import project
        from export_native_text_block_chain import page as base_page
        from export_native_page_structure import BODY
        for mode in ['vector','letters','stroke','rectangle','off-row','second-page','continuation']:
            texts=[('abc'[i]+'. 'if mode=='letters'else'')+'Original item words continue across the ordinary text region'for i in range(3)];ls=[project(l)for l in typed_lines([line(t,[30.,180.+14*i,370.,190.+14*i],i)for i,t in enumerate(texts)])];paths=[dict(bbox=[24.,183.+14*i,27.,186.+14*i],fill_visible=True,stroke_visible=mode=='stroke',rectangle_bboxes=[[24.,183.,27.,186.]]if mode=='rectangle'else[],segment_count=12)for i in range(3)]
            if mode=='off-row':
                for p in paths:p['bbox'][1]+=10.;p['bbox'][3]+=10.
            if mode=='continuation':ls.insert(1,project(typed_lines([line('Original aligned continuation',[40.,194.,370.,204.],8)])[0]));ls[2]['bbox'][1]+=14.;ls[2]['bbox'][3]+=14.;ls[3]['bbox'][1]+=14.;ls[3]['bbox'][3]+=14.
            q=dict(page=dict(page_size=[400.,600.],remaining_lines=ls,table_bboxes=[],drawing_lines=[],fixed_blocks=[],bullet_paths=paths),page_index=1 if mode=='second-page'else 0,document_body_profile=BODY,refine_blocks=True);c=dict(label='bullet-path/'+mode,request=q);state=original.reference(q);c['expected']=base_page(q,state);c['expected'].update(reference(q));f['cases'].append(c)
    write_json(target,f);write_json(chain,g);print(len(f['cases']),'contiguous refined pages;',len(g['cases']),'raw-source chains;',sum('error'in c for c in f['cases']+g['cases']),'exceptions')
if __name__=='__main__':main()
