"""冻结未修改的旧公式编号后处理及其与页面/段落的串联，拒绝覆盖。"""
import argparse,copy,importlib,json,sys
from common import ROOT,digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='build/migration/legacy-formula-numbers.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Frozen baseline exists; use a new --output')
    module=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    split=importlib.import_module('rapid_doc.backend.pipeline.para_split').para_split
    cases=[];chains=[];coverage={};filename=module._optimize_formula_number_blocks.__code__.co_filename
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=filename:return None
        if event=='line':coverage.setdefault(frame.f_code.co_name,set()).add(frame.f_lineno)
        return trace
    def add(name,pages):
        expected=copy.deepcopy(pages);sys.settrace(trace)
        try:module._optimize_formula_number_blocks(expected)
        finally:sys.settrace(None)
        cases.append(dict(name=name,pages=pages,expected=expected))
    def block(kind,content='',bbox=None,spans=None):
        box=bbox or [0,10,200,30]
        if spans is None:spans=[dict(type='interline_equation'if kind=='interline_equation'else'text',content=content,custom=[False,None])]
        return dict(type=kind,bbox=box,lines=[dict(bbox=box,spans=spans,custom='line')],custom={'preserved':'block'})
    def eq(content='x+y',bbox=None):return block('interline_equation',content,bbox)
    def number(content='(1)',bbox=None):return block('formula_number',content,bbox or [250,10,280,30])
    def page(blocks):return dict(page_idx=0,page_size=[600,800],preproc_blocks=blocks,para_blocks=['not-recomputed'],custom={'preserved':True})
    add('no-pages',[]);add('missing-blocks',[{}]);add('opaque-block',[page([dict(type='image',opaque=[1,False,None])])])
    tags=['(1)','（１）',' （ Ａ－２ ） ','1','(1','1)','((1))','','()',' \x1c ','\u3000（2）\u3000',r'\tag{1}','{a}']
    for tag in tags:
        for formula in ['x+y',r'x+y\tag{1}',r'x+y\tag{2}',r'x\tag{1}\tag{2}','',None]:
            for layout in ['after','before','both','non-adjacent']:
                blocks=[eq(formula),number(tag)]if layout=='after'else[number(tag),eq(formula)]if layout=='before'else[number(tag),eq(formula),number('(2)')]if layout=='both'else[eq(formula),block('text','body',[0,100,300,120]),number(tag)]
                add(f'{layout}/{repr(tag)}/{repr(formula)}',[page(blocks)])
    add('removed-number-still-in-original-neighbor-list',[page([eq(),number('(1)'),number('(1)'),number('(2)')])])
    add('first-equation-span-only',[page([block('interline_equation',spans=[dict(type='inline_equation',content='inline'),dict(type='interline_equation',content='first'),dict(type='interline_equation',content='second')]),number()])])
    add('no-equation-span',[page([block('interline_equation',spans=[dict(type='text',content='not latex')]),number()])])
    add('number-extracts-only-text',[page([eq(),block('formula_number',spans=[dict(type='inline_equation',content='ignored'),dict(type='text',content='（'),dict(type='text',content='１'),dict(type='text',content='）')])])])
    for overlap in [9.99999999,10.,10.00000001]:
        for gap in [239.99999999,240.,240.00000001]:
            add(f'geometry/{overlap}/{gap}',[page([eq(bbox=[0,0,100,20]),block('text','barrier',[0,100,100,120]),number(bbox=[100+gap,20-overlap,140+gap,40-overlap])])])
    for left in [-.00000001,0.,.00000001]:
        add(f'geometry-left/{left}',[page([eq(bbox=[0,0,100,20]),block('text','barrier',[0,100,100,120]),number(bbox=[left,0,left+20,20])])])
    add('stable-first-geometric-tie',[page([eq('first',[0,10,100,30]),eq('second',[200,10,300,30]),block('text','barrier',[0,100,100,120]),number(bbox=[350,10,400,30])])])
    add('nearest-center-not-first',[page([eq('first',[0,0,100,20]),eq('second',[200,5,300,25]),block('text','barrier',[0,100,100,120]),number(bbox=[350,5,400,25])])])
    add('closest-conflict-does-not-retry-others',[page([eq(r'first\tag{2}',[0,10,100,30]),eq('second',[200,10,300,30]),block('text','barrier',[0,100,100,120]),number(bbox=[350,10,400,30])])])
    add('adjacent-priority-ignores-geometry',[page([eq(bbox=[0,0,100,20]),number(bbox=[500,700,550,720])])])
    add('no-cross-page-association',[page([eq()]),dict(page([number()]),page_idx=1)])
    old=json.loads((ROOT/'build/migration/legacy-page-chain.json').read_text())
    for c in old['cases']:
        add('page-chain/'+c['name'],copy.deepcopy(c['expected']))
        pages=copy.deepcopy(c['expected']);module._optimize_formula_number_blocks(pages);split(pages)
        chains.append(dict(name=c['name'],requests=c['requests'],expected=pages))
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/utils/char_utils.py','rapid_doc/backend/pipeline/para_split.py']
    write_json(target,dict(cases=cases,chains=chains,sources={p:digest((ROOT/p).read_bytes())for p in sources},
        executed_lines={k:sorted(v)for k,v in coverage.items()},profile='完整旧公式编号 JSON，邻块优先/几何回退/既有 tag/孤立编号；单页对象引用与未知字段；串联优化→para_split，不含跨页表格'))
    print(f'Frozen {len(cases)} complete old formula-number references and {len(chains)} model-page→formula→paragraph chains')


if __name__=='__main__':main()
