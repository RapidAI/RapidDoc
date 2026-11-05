"""冻结旧 union_make 输出；非算法、默认 $/$$；不改旧字段或归一化文本。"""
import argparse,copy,importlib,json,sys
from common import ROOT,digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='build/migration/legacy-output.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Frozen baseline exists; use a new --output')
    old=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    assert (old.inline_left_delimiter,old.inline_right_delimiter,old.display_left_delimiter,old.display_right_delimiter)==('$','$','$$','$$')
    cases=[];excluded=[];coverage={};filename=old.union_make.__code__.co_filename
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=filename:return None
        if event=='line':coverage.setdefault(frame.f_code.co_name,set()).add(frame.f_lineno)
        return trace
    def add(name,pages,prefix='images'):
        for mode in ('mm_markdown','nlp_markdown','content_list'):
            sys.settrace(trace)
            try:expected=old.union_make(copy.deepcopy(pages),mode,prefix)
            finally:sys.settrace(None)
            cases.append(dict(name=name+'/'+mode,request=dict(pages=pages,make_mode=mode,image_path_prefix=prefix),expected=expected))
    chain=json.loads((ROOT/'build/migration/legacy-page-chain.json').read_text())
    def algorithm(v):
        if isinstance(v,dict):return v.get('original_label')=='algorithm'or any(algorithm(x)for x in v.values())
        if isinstance(v,list):return any(algorithm(x)for x in v)
        return False
    for c in chain['cases']:
        if algorithm(c['expected']):excluded.append(dict(name=c['name'],reason='Magika 语言分类尚未迁移，不伪造 txt 标签'));continue
        add('model-chain/'+c['name'],c['expected'])
    def line(text,kind='text',**extra):return dict(bbox=[10,10,200,30],spans=[dict(type=kind,content=text)],**extra)
    def block(kind='text',lines=None,**extra):return dict(type=kind,bbox=[10,10,200,80],lines=lines or [],**extra)
    def pages(blocks,discarded=None):return [dict(page_idx=3,page_size=[600,800],para_blocks=blocks,discarded_blocks=discarded or [])]
    for gap in [1.499999999,1.5,1.500000001]:
        add('inline-gap/'+str(gap),pages([block(lines=[dict(bbox=[0,0,210,10],spans=[dict(type='text',content='one',bbox=[0,0,100,10]),dict(type='text',content='two',bbox=[100+gap,0,200,10])])])]))
    for text,next_text in [('文','字'),('English','words'),('inter-','national'),('open-','source'),('https://example.','com/path'),('Ｆｕｌｌ１２３*`~','尾'),('1–','2'),('a\u00ad','b'),(' A ',' B ')]:
        add('text-join/'+text,pages([block(lines=[line(text),line(next_text)])]))
    add('list-hard-break',pages([block('list',[line('first'),line('second',is_list_start_line=True),line('tail')])]))
    add('mixed-spans',pages([block(lines=[dict(bbox=[10,10,200,30],spans=[dict(type='text',content='中文'),dict(type='inline_equation',content='x^2'),dict(type='checkbox',content='☑'),dict(type='text',content='tail')])])]))
    for level in [-2,0,1,4,7]:add('title/'+str(level),pages([block('title',[line('Heading-\nline')],level=level)]))
    for latex in ['',r'\frac{a}{b}']:
        add('equation/'+latex,pages([block('interline_equation',[dict(bbox=[10,10,200,30],spans=[dict(type='interline_equation',content=latex,image_path='equation.png')])])]))
    caption=block('image_caption',[line('Caption')]);footnote=block('image_footnote',[line('Footnote')])
    for feet in [[],[footnote]]:
        for seal in ['', '印章', [' 印 ', '', '章']]:
            body=block('image_body',[dict(bbox=[10,10,200,50],spans=[dict(type='image',image_path='crop/image.png',original_label='seal',content=seal)])])
            add('image/'+str(len(feet))+'/'+str(seal),pages([dict(type='image',bbox=[10,10,200,80],blocks=[caption,body]+feet)]),prefix='')
    for html in ['', '<table><tr><td rowspan="2">A &amp; B</td></tr></table>']:
        body=block('table_body',[dict(bbox=[10,10,200,50],spans=[dict(type='table',html=html,image_path='table.png')])])
        add('table/'+str(bool(html)),pages([dict(type='table',bbox=[],blocks=[block('table_caption',[line('Table 1')]),body,block('table_footnote',[line('note')])])]))
    add('empty-with-discarded',pages([],discarded=[block('discarded',[line('skip like old')])]))
    add('discarded-after-body',pages([block(lines=[line('body')])],discarded=[block('discarded',[line('discarded')])]))
    paths=['rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py','rapid_doc/utils/text_boundary.py','rapid_doc/_vendor/docvortex_core/foundation/_text.py']
    write_json(target,dict(cases=cases,excluded=excluded,sources={p:digest((ROOT/p).read_bytes())for p in paths},
        executed_lines={k:sorted(v)for k,v in coverage.items()},profile='旧 union_make 完整 Markdown/content_list；默认 LaTeX；算法 Magika 分类显式排除；不执行资产写出'))
    print(f'Frozen {len(cases)} complete old output references; {len(excluded)} algorithm chains explicitly excluded')


if __name__=='__main__':main()
