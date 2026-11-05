"""冻结旧表格选优规则；候选限 Rust 输出器支持的规范表格 HTML。"""
import html,random,json
from pathlib import Path
from bs4 import BeautifulSoup
from rapid_doc.model.table.utils import select_best_table_model,count_table_cells_physical
from common import ROOT,write_json,digest

def escape(text):return html.escape(text).replace('&#x27;','&#39;')
def table(values,wrapped=False,header=False):
    cells=''.join(f'<td rowspan="1" colspan="1">{escape(v)}</td>'for v in values)
    body=f'<tr>{cells}</tr>'
    if header:body='<thead>'+body+'</thead>'
    result='<table>'+body+'</table>'
    return '<html><body>'+result+'</body></html>'if wrapped else result
def metrics(source,texts):
    count=count_table_cells_physical(source)
    blank=sum(1 for cell in BeautifulSoup(source,'html.parser').find_all(['td','th'])if not cell.text.strip())
    return dict(physical_cells=count,blank_cells=blank,non_blank_cells=count-blank,ocr_hits=sum(escape(t)in source for t in texts))
def contract(name,texts,wired,wireless):
    selected=select_best_table_model([[],[escape(v)for v in texts]],wired,wireless)
    route=('wired'if selected==wired else'wireless')if wired!=wireless else None
    return dict(id=name,ocr_texts=texts,wired_html=wired,wireless_html=wireless,expected_html=selected,expected_route=route,wired=metrics(wired,texts),wireless=metrics(wireless,texts))
def main():
    cases=[]
    fixed=[
        ('wired-priority',['A','B'],['A','B','C'],['A','','']),
        ('wireless-scale',[],['A'],['A','B','C']),
        ('small-gap',[],['A','B','C','',''],['A','B','C','','','','']),
        ('small-equal',[],['A','B','C'],['A','B','C']),
        ('coverage',[f'word{i}'for i in range(20)],['x']*25,[f'word{i}'for i in range(20)]+['x']*5),
        ('wired-default',[],['A']*9,['A']*9),
        ('unicode-whitespace',['\u001c','\u0085',''],['\u001c\u001f\u0085','\u00a0','中文'],['\u001c','\u3000','中文']),
        ('escaped-ocr',['<angle>','&','"',"'"],['<angle> & " \'']*6,['<angle>']*6),
        ('substring-duplicates',['A','A','1',''],['AB']*8,['A']*9),
        ('empty-content',[],[],[]),
    ]
    for name,texts,w,l in fixed:cases.append(contract(name,texts,table(w),table(l,True)))
    cases.append(contract('empty-html',['','x'],'',''))
    cases.append(contract('thead-legacy-prefix',['A'],table(['','A'],header=True),table(['A'],True)))
    cases.append(contract('line-break-only',[], '<table><tr><td><br/></td><th>\u001c<br>\u3000</th></tr></table>',table(['x',''],True)))
    rng=random.Random(7135);pool=['',' ','\t','\u001c','\u0085','中文','alpha','123','<tag>','&amp;',"O'Reilly",'"quote"']
    for i in range(180):
        w=[rng.choice(pool)for _ in range(rng.randrange(0,35))];l=[rng.choice(pool)for _ in range(rng.randrange(0,35))]
        texts=[rng.choice(pool)for _ in range(rng.randrange(0,25))]
        cases.append(contract(f'random-{i}',texts,table(w,header=i%9==0),table(l,True)))
    # 已冻结的实际 UNET 和两种 SLANet 输出，对同一张图应用旧 selector。
    wired=json.loads((ROOT/'build/migration/wired-pipeline.json').read_text())
    wireless=json.loads((ROOT/'build/migration/table-match.json').read_text())
    for variant in ['slanet-plus','slanet-1m']:
        for name in ['wired','wireless']:
            w=next(v for v in wired['cases']if Path(v['input']).name==name+'.png')
            l=next(v for v in wireless['png_cases']if v['variant']==variant and Path(v['input']).name==name+'.png')
            # SLANet 的规范 HTML 由冻结 token/文字在纯 Python 参考中构建。
            texts=[' '.join(l['ocr_lines'][i]['text']for i in indices)for indices in l['expected']['cells']]
            # 先填 token，再规范化为 Rust renderer 的 table/tr/td 格式。
            tokens=l['structure']['tokens'];idx=0;parts=[];pending=False
            for token in tokens:
                if token=='<td></td>':parts.append('<td>'+escape(texts[idx])+'</td>');idx+=1
                elif token.startswith('<td'):
                    parts.append(token);pending=True
                    if token.endswith('>'):parts.append(escape(texts[idx]));idx+=1;pending=False
                elif pending:
                    parts.append(token)
                    if token.endswith('>'):parts.append(escape(texts[idx]));idx+=1;pending=False
                else:parts.append(token)
            assert idx==len(texts)
            soup=BeautifulSoup(''.join(parts),'html.parser');canonical=['<table>\n']
            for row in soup.find_all('tr'):
                canonical.append('<tr>')
                for cell in row.find_all(['td','th'],recursive=False):
                    canonical.append('<td')
                    for attr in ['rowspan','colspan']:
                        value=int(cell.get(attr,1))
                        if value>1:canonical.append(f' {attr}="{value}"')
                    canonical.extend(['>',escape(cell.get_text()).replace('\r\n','\n').replace('\n','<br/>'),'</td>'])
                canonical.append('</tr>\n')
            canonical.append('</table>')
            cases.append(contract(f'actual-{variant}-{name}',[line['text']for line in l['ocr_lines']],w['html'],''.join(canonical)))
    write_json(ROOT/'build/migration/table-quality.json',dict(cases=cases,source_sha256=digest((ROOT/'rapid_doc/model/table/utils.py').read_bytes())))
    print(f'Exported {len(cases)} table-quality decisions and metrics')
if __name__=='__main__':main()
