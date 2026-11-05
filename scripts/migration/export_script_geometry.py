"""冻结旧基础几何上下标分类和数值引用判定；不预先分配脚本角色。"""
import copy,json,random
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/script-geometry.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._script_geometry import classify_char_script_roles,build_script_features,_numeric_superscript_indices,_native_text_flags
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    cases=[]
    def add(label,chars,tight,origins,protected=()):
        request=dict(chars=encode(chars),tight_bboxes=encode(tight),origins=encode(origins),protected_body_indices=list(protected))
        roles=classify_char_script_roles(chars,tight_bboxes=tight,origins=origins,protected_body_indices=set(protected))
        numeric=sorted(_numeric_superscript_indices(build_script_features(chars,tight,origins,set(protected)),['body']*len(chars),set(protected)))
        cases.append(dict(label=label,request=request,expected=dict(roles=roles,numeric_indices=numeric)))
    for name in ('pdf-native-geometry','pdf-native-dedup-extra'):
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            g=c['expected'];tight={int(k):tuple(v)for k,v in g['tight_bboxes'].items()};origins={int(k):tuple(v)for k,v in g['origins'].items()}
            chars=copy.deepcopy(g['chars'])
            for ch in chars:ch['bbox']=Bbox(ch['bbox'])
            add(f'{name}:{c["page_index"]}:{c["extended"]}',chars,tight,origins)
    rng=random.Random(725739)
    for n in range(1400):
        chars=[];tight={};origins={};count=rng.randrange(2,45);protected=[];x=0.
        for i in range(count):
            text=rng.choice('ABCxyz012345[](),—./⁄†中文β² ');height=rng.choice([10.,10.,10.,6.,8.8,9.,11.,13.5]);baseline=rng.choice([30.,30.,30.,26.,28.8,29.5,30.35,34.]);x+=rng.choice([3.,6.,9.,30.,-12.])
            c=dict(char=text,char_idx=i,bbox=Bbox([x,baseline-height,x+6.,baseline]),font=dict(name=rng.choice(['Arial','Arial','Times','']),flags=rng.choice([0,0,64]),weight=400,size=10.))
            if n%19==0 and i%3==0:c['bbox']=Bbox([x,baseline,x,baseline])
            chars.append(c)
            if n%13 or i%5:tight[i]=(x,baseline-height,x+5.,baseline)
            if n%17 or i%4:origins[i]=(x,baseline)
            if n%7==0 and i%5==0:protected.append(i)
        add(f'random:{n}',chars,tight,origins,protected)
    for prefix in range(72):
        chars=[dict(char=' ',char_idx=i,bbox=Bbox([0.,20.,8.,30.]))for i in range(prefix)]
        tight={};origins={}
        for i,h,y,origin in [(prefix,10.,20.,30.),(prefix+1,8.,16.8,29.3),(prefix+2,8.9,21.1,30.)]:
            chars.append(dict(char='A',char_idx=i,bbox=Bbox([0.,y,8.,y+h]),font=dict(name='Arial',flags=0,weight=400,size=10.)));tight[i]=(0.,y,8.,y+h);origins[i]=(0.,origin)
        add(f'nearest-tie:{prefix}',chars,tight,origins)
    flags=[dict(text=chr(n),expected=_native_text_flags(chr(n)))for n in range(0x110000)if not 0xd800<=n<=0xdfff and _native_text_flags(chr(n))]
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_script_geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,flags=flags));print('Frozen',len(cases),'base script cases;',len(flags),'nonzero Unicode16 flag facts;',sum(any(r!='body'for r in c['expected']['roles'])for c in cases),'marked outputs')
if __name__=='__main__':main()
