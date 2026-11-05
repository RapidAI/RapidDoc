"""Original post-assembly member/title/marginal/front-matter rules."""
import copy,json
from common import ROOT,write_json
from export_native_final_text_lines import typed_lines
from export_native_inline_clusters import clean
from export_native_text_block_edges import block
def typed(raw):
    out=copy.deepcopy(raw)
    for b in out:
        if '_text_lines'in b:b['_text_lines']=typed_lines(b['_text_lines'])
        for k in ['_font_signatures','_visual_row_ids']:
            if k in b:b[k]=set(tuple(v)if isinstance(v,list)else v for v in b[k])
    return out
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import annotations as a,continuity as c
    bs=typed(q['blocks']);p=tuple(q['page_size']);kind=q['kind']
    fn=dict(members=c.merge_overlapping_member_blocks,titles=a._merge_multiline_title_blocks,front=a._merge_front_matter_column_blocks,headers=a._merge_fragmented_header_blocks,compact=a._merge_repeated_compact_title_continuations)[kind]
    return clean(fn(bs,p,page_index=q['page_index'])if kind=='front'else fn(bs)if kind in ['titles','headers']else fn(bs,p))
def main():
    target=ROOT/'build/migration/native-page-block-rules.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,bs,p,kind,page=0):
        q=dict(blocks=clean(bs),page_size=p,kind=kind,page_index=page);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except(ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for i,c in enumerate(json.loads((ROOT/'build/migration/native-text-blocks.json').read_text(encoding='utf-8'))['cases']):
        if 'expected'not in c or i%7:continue
        for kind in ['members','titles','front','headers','compact']:add(c['label']+'/'+kind,c['expected'],c['request']['page_size'],kind,c['request'].get('page_index',0))
    from export_native_title_layout import line
    from export_native_late_image_text import project
    def with_lines(t,b,i,**kw):
        ls=typed_lines([line(t,b,i)]);return block(t,b,_text_lines=[project(l)for l in ls],**kw)
    groups=[('members',[with_lines('Original con-', [30.,180.,370.,190.],0),with_lines('tinuation words',[30.,194.,370.,204.],1)]),('members',[with_lines('Original body words',[30.,180.,370.,190.],0,_reference_group=7),with_lines('Original next member',[30.,194.,370.,204.],1,_reference_group=7)]),('members',[with_lines('Original ordinary sentence',[30.,180.,370.,190.],0),with_lines('sidecar',[160.,184.,175.,194.],1)]),('titles',[with_lines('Original first title',[30.,180.,370.,190.],0,type='paragraph_title'),with_lines('Original second title',[30.,194.,370.,204.],1,type='paragraph_title')]),('headers',[block('A',[30.,30.,35.,40.],type='header',_single_run_row_id=8),block('B',[45.,30.,50.,40.],type='header',_single_run_row_id=8)]),('compact',[block('Original weak heading',[30.,160.,220.,184.],type='paragraph_title',rows=[[30.,160.,220.,170.],[30.,174.,220.,184.]],font='Title'),block('Original continuation',[30.,188.,220.,198.]),block('Next weak heading',[30.,240.,220.,264.],type='paragraph_title',rows=[[30.,240.,220.,250.],[30.,254.,220.,264.]],font='Title'),block('Next continuation',[30.,268.,220.,278.])]),('front',[block('Original title',[30.,20.,370.,40.],type='doc_title')]+[block(f'Original author {r} {c}',[20.+95*c,65.+14*r,90.+95*c,75.+14*r])for r in range(3)for c in range(4)])]
    for i,(kind,bs)in enumerate(groups):
        for mode in ['base','reverse','rule','explicit','geometry','protected','font-conflict','duplicate-source','second-page','same-band','different-band']:
            v=copy.deepcopy(bs)
            if mode=='reverse':v.reverse()
            for b in v:
                for l in b.get('_text_lines',[]):
                    if mode=='duplicate-source':l['source_index']=7
                    if mode in ['same-band','different-band']:l.update(structural_title=True,explicit_section_title=True,title_band_id=9 if mode=='same-band'else l['source_index'])
            if mode in ['rule','explicit','geometry','protected']:v[-1][dict(rule='_rule_break_before',explicit='_explicit_break_before',geometry='_geometry_break_before',protected='_protected_hard_break_before')[mode]]=True
            if mode=='font-conflict':v[-1]['_font_signatures']={('Different',64)}
            add(f'direct/{kind}/{i}/{mode}',v,[400.,600.],kind,1 if mode=='second-page' else 0)
    write_json(target,dict(cases=cases));print(len(cases),'post-assembly rules;',sum('error'in c for c in cases),'exceptions;',sum(c.get('expected')!=c['request']['blocks']for c in cases if 'expected'in c),'positive changes')
if __name__=='__main__':main()
