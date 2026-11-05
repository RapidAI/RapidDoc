"""冻结旧换行连接器参考和独立几何段落合同；不导入 Rust 扩展。"""
import json,random,re
from common import ROOT,write_json
from rapid_doc.utils.text_boundary import merge_text_line_contents

def main():
    joins=[]
    def add(lines,name):joins.append(dict(name=name,lines=lines,expected=merge_text_line_contents(lines)))
    for a,b in [('中文','段落'),('한국어','문장'),('hello','world'),('hello',','),('(hello','world)'),('中</b>','文'),
                ('hello\u0301','word'),('hello\u200d','word'),('','word'),('\x1c hello\x1c',' next '),('http','s://'),
                ('https://exa','mple.test'),('https://one/','https://two/'),('abchttps://exa','mple.test'),('中https://exa','mple.test')]:add([a,b],'basic')
    for prefix in ['inter','open','non','self','GLGE','A','of','state-of','cross']:
        for hyphen in '-\u00ad\u2010\u2011\u2043':
            for following in ['national','Source','école','Ελλάδα','中文','123']:add([prefix+hyphen,following],'hyphen')
    for opening in ['<b>','<sup>','<SUB x=\"a\">','<ſup>','<İ>','<strİKe>','<b中文>','<b_>','<b\u0301>','<b','<b x=\">\">']:
        for lines in [[opening+'中','文'],['word'+opening,','],[opening+'(', 'word']]:add(lines,'styles')
    for chunks in [['https://exa','mple.test/','long','/path'],['www.','example.','test'],['中文','https://a/','片段'],['1–','³'],['۱-','२']]:add(chunks,'context')
    # 仅用生成表选择测试点；期望值全部来自未修改的旧 Python 连接器。
    source=(ROOT/'rust/rapiddoc-native/src/text_unicode_data.rs').read_text()
    points=set()
    for a,b,_ in re.findall(r'\(0x([0-9A-F]+),\s*0x([0-9A-F]+),\s*(\d+)\)',source):points.update([int(a,16),int(b,16)])
    for value in sorted(points):
        if 0xd800<=value<=0xdfff:continue
        c=chr(value)
        for lines in [['previous',c+'next'],['a'+c,'b'],['word-',c+'next']]:add(lines,f'unicode-{value:x}')
    rng=random.Random(846)
    parts=['plain','word-','中文','かな','한국어','مرحبا','Été','a\u0301','(open','close)','1–','²','https://exa','mple.test/','<b>文</b>','\x1c','', '  text  ']
    for _ in range(500):add([rng.choice(parts)for _ in range(rng.randint(2,5))],'random')
    def line(b,text):return dict(points=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],text=text,confidence=None)
    def block(index,lines,label='text',kind='text',text=None):return dict(layout_index=index,kind=kind,
        region=dict(label=label,coordinate=[0,0,600,500]),lines=lines,text=text)
    def page(blocks):return dict(width=600,height=500,coordinate_space='page_pixels',blocks=blocks)
    columns=page([block(7,[line([10,10,240,30],'left one'),line([330,10,560,30],'right one'),line([10,40,240,60],'left two'),line([330,40,560,60],'right two')])])
    cases=[dict(name='columns',page=columns,split_columns=True,groups=[[0,2],[1,3]],kinds=['text','text'],texts=['left one left two','right one right two']),
           dict(name='source-order',page=columns,split_columns=False,groups=[[0],[1],[2],[3]],kinds=['text']*4,texts=['left one','right one','left two','right two'])]
    values=[line([10,10,240,30],'first'),line([10,40,240,60],'continued'),line([40,70,240,90],'indented'),line([10,100,240,120],'continued'),
        line([10,130,240,150],''),line([10,160,240,180],'1. item'),line([10,190,240,210],'3.14 value')]
    cases.append(dict(name='indent-blank-list',page=page([block(0,values)]),split_columns=True,groups=[[0,1],[2,3],[4],[5],[6]],kinds=['text','text','text','list_item','text'],texts=['first continued','indented continued','','1. item','3.14 value']))
    values=[line([0,0,200,20],'中文'),line([0,25,200,45],'段落'),line([0,100,200,120],'new paragraph')]
    cases.append(dict(name='script-and-gap',page=page([block(0,values)]),split_columns=True,groups=[[0,1],[2]],kinds=['text','text'],texts=['中文段落','new paragraph']))
    values=[line([0,0,200,20],'line one'),line([0,25,200,45],'line two')]
    cases.append(dict(name='layout-barriers',page=page([block(0,values,'doc_title'),block(1,[],kind='table',text='A\nB'),block(2,values),block(3,[],kind='unsupported')]),
        split_columns=True,groups=[[0,1],[],[0,1],[]],kinds=['title','table','text','unsupported'],texts=['line one line two','A\nB','line one line two',None]))
    cases.append(dict(name='algorithm',page=page([block(0,[line([0,0,200,20],'  if x:'),line([20,80,200,100],'    return x')],'algorithm')]),
        split_columns=True,groups=[[0,1]],kinds=['algorithm'],texts=['  if x:\n    return x']))
    cases.append(dict(name='short-tail',page=page([block(0,[line([0,0,200,20],'full'),line([0,25,60,45],'end.'),line([0,50,200,70],'next')])]),
        split_columns=True,groups=[[0,1],[2]],kinds=['text','text'],texts=['full end.','next']))
    native=json.loads((ROOT/'build/migration/native-fusion.json').read_text())
    runtime=[dict(page_index=c['page_index'],dpi=c['dpi'],expected=merge_text_line_contents([l['text']for l in c['ocr']]))for c in native['cases']]
    write_json(ROOT/'build/migration/paragraphs.json',dict(join_cases=joins,page_cases=cases,runtime_joins=runtime))
    print(f'Exported {len(joins)} legacy line joins and {len(cases)} authored paragraph contracts')
if __name__=='__main__':main()
