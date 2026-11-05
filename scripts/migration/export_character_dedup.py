"""冻结旧DocVortex完整字符去重；不替换任何内部候选或删除判断。"""
import copy,json,math,random,unicodedata
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/character-dedup.json'
    data=ROOT/'rust/rapiddoc-pdf/src/character_primitives/unicode.json'
    if target.exists()or data.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    from rapid_doc._vendor.docvortex_core.document.pdf.text.dedup import deduplicate_chars,_radical_equivalents
    assert unicodedata.unidata_version=='16.0.0'
    def decode(rows):
        rows=copy.deepcopy(rows)
        for c in rows:c['bbox']=Bbox(c['bbox']);c['source_indices']=tuple(c['source_indices'])
        return rows
    cases=[];real=[]
    def add(label,rows):cases.append(dict(label=label,chars=encode(rows),expected=encode(deduplicate_chars(rows))))
    for name in ('pdf-character-primitives','pdf-character-edges'):
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            real.append(dict(case_index=len(cases),pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],extended=c['extended']))
            add(f'{name}:{c["page_index"]}:{c["extended"]}',decode(c['expected']['restored_chars']))
    def char(text,index,obj,x=10.,y=20.,mode=0,visible=None):
        c=dict(char=text,char_idx=index,source_indices=(index,),raw_code=ord(text[0])if text else 0,bbox=Bbox([x,y,x+8.,y+10.]),rotation=0.,font=dict(name='Helvetica',flags=0,size=10.,weight=400),text_object_id=obj,text_render_mode=mode,writing_angle=0.,origin=(x,y+9.))
        if visible is not None:c['text_is_visible']=visible
        return c
    rng=random.Random(719327)
    for n in range(500):
        text=rng.choice(['ABCDEF','A A A','中文測試資料','１２３４５６','office','abcdefghi'])
        dx=rng.choice([0.,.0009,.0011,.1,1.,2.49,2.51]);dy=rng.choice([0.,.0009,.1,1.,2.49,2.51]);rows=[]
        for obj in range(rng.choice([2,3,4])):
            for i,t in enumerate(text):rows.append(char(t,len(rows),obj,10.+i*9.+dx*obj,20.+dy*obj))
        if n%7==0:rows[-1]['font']['weight']=700
        if n%11==0:rows[-2]['origin']=None
        add(f'paint:{n}',rows)
    for n in range(240):
        visible=rng.choice(['ABCDEF','abcdefghi','中文測試資料內容','ＡＢＣＤＥＦ','１２３４５６','ffi'])
        hidden=unicodedata.normalize('NFKC',visible)
        if n%4==0:hidden=hidden[:-1]+rng.choice('X字')
        if n%9==0:hidden+='X'
        rows=[char(t,i,0,10.+i*9.,20.,visible=n%13!=0)for i,t in enumerate(visible)]
        rows += [char(t,len(rows)+i,1,10.+i*9.,20.+rng.choice([0.,0.,.05,.2]),mode=3,visible=False)for i,t in enumerate(hidden)]
        add(f'hidden:{n}',rows)
    for n,(a,b)in enumerate([('一','⼀'),('人','⼈'),('龍','龍'),('f','i'),('f','f'),('A','Ａ'),('一','二'),(' ','一')]):
        for delta in [0.,.0009,.0011]:
            rows=[char(a,0,0),char(b,1,0,10.+delta)];add(f'mapping:{n}:{delta}',rows)
    for count in (63,64,65,127,128,129):add(f'bucket:{count}',[char('A',i,i)for i in range(count)])
    radicals={str(ord(k)):v for k,v in _radical_equivalents().items()}
    han=[];start=last=None;nfkc=[];spaces=[]
    for code in range(0x110000):
        if 0xd800<=code<=0xdfff:continue
        c=chr(code)
        if c.isspace():spaces.append(code)
        v=unicodedata.normalize('NFKC',c)
        if v!=c:nfkc.append([code,v])
        if unicodedata.name(c,'').startswith('CJK UNIFIED IDEOGRAPH-'):
            if last is not None and code==last+1:last=code
            else:
                if last is not None:han.append([start,last])
                start=last=code
    if last is not None:han.append([start,last])
    write_json(data,dict(version=unicodedata.unidata_version,spaces=spaces,han=han,radicals=radicals))
    angles=sorted({c.get('writing_angle',-c['rotation'])for case in cases for c in case['chars']}|{rng.uniform(-6.3,6.3)for _ in range(1000)}|{.0005,.0015,-.0005,-0.})
    numeric=[dict(angle=x,cos=math.cos(x),sin=math.sin(x),rounded=round(x,3),bucket=round(x*1000))for x in angles]
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/text/dedup.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/geometry.py','rapid_doc/_vendor/docvortex_core/resources/unicode/EquivalentUnifiedIdeograph-17.0.0.txt']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,real=real,numerics=numeric,nfkc=nfkc))
    print('Frozen',len(cases),'complete dedup cases,',len(real),'real PDF/modes;',len(nfkc),'NFKC and',len(numeric),'angle facts')
if __name__=='__main__':main()
