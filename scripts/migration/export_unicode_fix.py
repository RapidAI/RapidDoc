"""冻结ftfy6.3.1默认fix_text的实际表、规则与Unicode16差分；不调用Rust生成期望。"""
import argparse,inspect,json,re,random,unicodedata,html
import ftfy,ftfy.chardata as c,ftfy.badness as b,ftfy.fixes as f
from common import ROOT,digest,write_json

def pattern(regex):
    s=regex.pattern
    if isinstance(s,bytes):s=s.decode('latin-1')
    if regex.flags&re.VERBOSE:
        out=[];inside=False;i=0
        while i<len(s):
            ch=s[i]
            if ch=='\\':out.append(s[i:i+2]);i+=2;continue
            if ch=='[':inside=True
            elif ch==']':inside=False
            if ch=='#'and not inside:
                j=s.find('\n',i);i=len(s)if j<0 else j+1;continue
            if inside or ch not in ' \t\r\n\v\f':out.append(ch)
            i+=1
        s=''.join(out)
    s=re.sub(r'\\U([0-9a-fA-F]{8})',lambda m:'\\u{'+m[1]+'}',s)
    # Python正则空白/十进制数类别锁定到当前Unicode16，而不是Rust正则更新后的类别。
    for escape,predicate in [('s',str.isspace),('d',lambda x:unicodedata.category(x)=='Nd')]:
        if '\\'+escape in s:
            group='['+''.join('\\u{'+format(i,'x')+'}'for i in range(0x110000)if predicate(chr(i)))+']'
            s=s.replace('\\'+escape,group)
    return s

def rules():
    codecs=[]
    for name in c.CHARMAP_ENCODINGS:
        decoded=[bytes([i]).decode(name)for i in range(256)]
        allowed=set(map(chr,list(range(0x1a))+list(range(0x1b,0x80))))|set(bytes(list(range(128,256))+[26]).decode(name))
        encoded={ord(ch):ch.encode(name)[0]for ch in allowed}
        codecs.append(dict(name=name,decode=decoded,encode=encoded))
    refs={m.__name__:digest(inspect.getsource(m).encode())for m in [ftfy,c,b,f]}
    return dict(version=ftfy.__version__,unicode=unicodedata.unidata_version,source_hashes=refs,
                codecs=codecs,ligatures=c.LIGATURES,width=c.WIDTH_MAP,controls=list(c.CONTROL_CHARS),entities=c.HTML_ENTITIES,
                invalid_refs=html._invalid_charrefs,invalid_points=sorted(html._invalid_codepoints),
                patterns={name:pattern(regex)for name,regex in [('badness',b.BADNESS_RE),('inconsistent',c.UTF8_DETECTOR_RE),('altered',c.ALTERED_UTF8_RE),('lossy',c.LOSSY_UTF8_RE),('a_grave',f.A_GRAVE_WORD_RE),('html',c.HTML_ENTITY_RE),('ansi',f.ANSI_RE)]})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-rules',action='store_true');args=parser.parse_args()
    assert ftfy.__version__=='6.3.1'and unicodedata.unidata_version=='16.0.0'
    data=rules();table=ROOT/'rust/rapiddoc-inference/src/ocr/unicode_fix_rules.json'
    if args.check_rules:
        assert json.loads(table.read_text(encoding='utf-8'))==json.loads(json.dumps(data));print('ftfy source tables/rules unchanged');return
    target=ROOT/'build/migration/unicode-fix.json'
    if target.exists():raise FileExistsError(target)
    texts=['','αβ','Ｘ＋Ｙ','e\u0301','\u001b[31mx\u001b[0m','&amp;','&amp','&#59;','&#x3B;','&#1abc;','&#xNOPE;','&amp;amp;amp;','<x &amp; y>','&amp;\n<x &amp;\n&amp;','＆amp;amp;','＜&amp;amp;','âœ” No problems','sÃ³','voilÃ le travail','FranÃ§ais','Ã quel','Ã quele','Ã s ','Ã ','í\xa0\xbdí\xb8\x80','À\x80','í\xa0\xbd', '“ﬃＸ”\r\n中文']
    for key in c.HTML_ENTITIES:texts.extend([key,'x '+key+' y','<x>\n'+key])
    for i in list(range(1024))+[0xD7FF,0xD800,0xDFFF,0xE000,0xFFFE,0xFFFF,0x10FFFF,0x110000,10**24-1]:
        texts.extend(['&#'+str(i)+';','&#x'+format(i,'X')+';'])
    samples=['café','Καλημέρα','Привет','中文','数学∑αβ','“hello”','😀😂🙂','áàäéé','日本語','a\u0301','™ – €','\ufffd']
    for enc in c.CHARMAP_ENCODINGS:
        for text in samples:
            for _ in range(3):
                text=text.encode().decode(enc)
                texts.extend([text,'中文 '+text+' 中文',text.replace('\xa0',' '),text.replace('\ufffd','?')])
    rng=random.Random(10007)
    for _ in range(2000):texts.append(bytes(rng.randrange(256)for _ in range(rng.randrange(1,50))).decode(rng.choice(c.CHARMAP_ENCODINGS)))
    # 全部有效Unicode标量及每个规范分解（含Hangul），不是只测常见字符。
    chunk=[]
    for i in range(0x110000):
        if 0xD800<=i<=0xDFFF:continue
        char=chr(i);chunk.append(char)
        nfd=unicodedata.normalize('NFD',char)
        if nfd!=char:texts.append(nfd)
        if len(chunk)==1024:texts.append(''.join(chunk));chunk=[]
    if chunk:texts.append(''.join(chunk))
    unique=list(dict.fromkeys(texts));cases=[dict(text=s,expected=ftfy.fix_text(s))for s in unique]
    write_json(table,data);write_json(target,dict(version=data['version'],unicode=data['unicode'],source_hashes=data['source_hashes'],cases=cases))
    print(f'Frozen {len(cases)} actual ftfy cases; Unicode scalars/NFD, entities, codec variants and mojibake')

if __name__=='__main__':main()
