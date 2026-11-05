"""补充真实旧跨页表格边界与三种输出；原始 99 组保持不变。"""
import copy,importlib,json,random
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-table-merge-extended.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    old=importlib.import_module('rapid_doc.utils.table_merge');render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    base=json.loads((ROOT/'build/migration/legacy-table-merge.json').read_text());cases=[]
    template=base['cases'][0]['pages'][0]['para_blocks'][0]
    def table(html):
        block=copy.deepcopy(template);block['blocks']=block['blocks'][:1];block['blocks'][0]['lines'][0]['spans'][0]['html']=html;return block
    def page(*blocks):return dict(para_blocks=list(blocks),preproc_blocks=[],discarded_blocks=[],page_size=[200,200])
    def add(name,pages):
        for i,p in enumerate(pages):p['page_idx']=i
        expected=copy.deepcopy(pages);old.merge_table(expected)
        outputs={mode:render.union_make(copy.deepcopy(expected),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}
        cases.append(dict(name=name,pages=pages,expected=expected,outputs=outputs))
    for c in base['cases']:add('base/'+c['name'],copy.deepcopy(c['pages']))
    previous='<table><tr><td>1</td><td>A</td></tr></table>'
    for content in ['A&amp;B','&lt;b&gt;','&nbsp;','&#65;&#x42;','&alpha;&beta;','&NotEqualTilde;','中Ａ（Ｂ）',"x<b>y</b>z",'<img z="Z" src="a.png" alt="A &amp; B"/>','<!--keep--><br/>','<svg><path d="x"></path></svg>','"double" \'single\'','  ','\n\t','&nbsp; \n']:
        html=f'<table class=" b  a " data-x=\'A " B\'><tbody>\n<tr><td>{content}</td><td>2</td></tr>\n</tbody></table>'
        add('html/'+str(len(cases)),[page(table(previous)),page(table(html))])
    for rows in [1,2,3,5,6,7]:
        a='<table>'+''.join('<tr><td>H</td><td>X</td></tr>'for _ in range(rows))+'<tr><td>1</td><td>A</td></tr></table>'
        b='<table>'+''.join('<tr><td>H</td><td>X</td></tr>'for _ in range(rows))+'<tr><td>2</td><td>B</td></tr></table>'
        add('header-limit/'+str(rows),[page(table(a)),page(table(b))])
    for flags in [[1,1],[1.,1.],[True,True],[0,1],[0,0],[],[1,0,1],['1',1]]:
        b=table('<table><tr><td></td><td>B</td></tr><tr><td>2</td><td>C</td></tr></table>');b['cell_merge']=flags
        add('flags/'+str(flags),[page(table(previous)),page(b)])
    for content in ['text','<img src="x"/>','<math><mi>x</mi></math>','<!--empty--><br/>','&nbsp;']:
        b=table(f'<table><tr><td rowspan="2"></td><td>{content}</td></tr><tr><td>C</td></tr></table>');b['cell_merge']=[1,1]
        add('carry-rowspan/'+str(len(cases)),[page(table(previous)),page(b)])
    for length in [3,4,6,10]:
        add('multi-page/'+str(length),[page(table(previous))for _ in range(length)])
    # 中间页有首尾两个表，首表下方 caption 恢复会移动已缓存尾表的下标。
    middle=page(table(previous),table(previous));cap=dict(type='table_caption',bbox=[0,81,100,90],lines=[dict(spans=[dict(type='text',content='下一节')])]);middle['para_blocks'][0]['blocks'].append(cap)
    add('identity-after-caption-insertion',[page(table(previous)),middle,page(table(previous))])
    rng=random.Random(20261024)
    for i in range(150):
        def html():
            rows=[]
            for _ in range(rng.randint(1,5)):
                cells=[]
                for _ in range(rng.randint(1,5)):
                    cs,rs=rng.randint(1,3),rng.randint(1,3);content=rng.choice(['','A','B','中','１','<img src="x.png"/>'])
                    cells.append(f'<td colspan="{cs}" rowspan="{rs}">{content}</td>')
                rows.append('<tr>'+''.join(cells)+'</tr>')
            return '<table>'+''.join(rows)+'</table>'
        add('generated-span/'+str(i),[page(table(html())),page(table(html()))])
    paths=['rapid_doc/utils/table_merge.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py','rapid_doc/utils/char_utils.py','rust/rapiddoc/src/legacy_table_merge/entities.json']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in paths},profile='实际旧 merge_table 和 union_make 完整 JSON/HTML/三种输出；随机跨度固定种子，合法闭合非嵌套 HTML，不归一化序列化差异'))
    print(f'Frozen {len(cases)} extended table merges and {len(cases)*3} old outputs')

if __name__=='__main__':main()
