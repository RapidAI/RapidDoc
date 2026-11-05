"""Freeze complete old character-to-visual-line results, no stage substitutions."""
import copy,json,math,random,unicodedata,sys
from dataclasses import fields,MISSING
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    supplement='--supplement'in sys.argv
    target=ROOT/('build/migration/native-visual-lines-extra.json'if supplement else'build/migration/native-visual-lines.json')
    schema=ROOT/'rust/rapiddoc/src/processing/text/native_line_defaults.json'
    unicode=ROOT/'rust/rapiddoc-native/src/visual_math.json'
    if any(p.exists()for p in ((target,)if supplement else(target,schema,unicode))):raise FileExistsError('frozen visual reference exists')
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import native_text as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.document.pdf.text._contracts import Bbox
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    names=[f.name for f in fields(_LineItem)if not f.name.startswith('_')]
    defaults={}
    for f in fields(_LineItem):
        if f.name not in names:continue
        if f.default is not MISSING:defaults[f.name]=clean(f.default)
        elif f.default_factory is not MISSING:defaults[f.name]=clean(f.default_factory())
    cases=[]
    def add(label,chars,size=(240.,180.),rotation=0,angles=(0.,90.,270.),pdf=None):
        q=dict(chars=copy.deepcopy(chars),page_size=list(size),page_rotation=rotation,supported_angles=list(angles))
        objects=copy.deepcopy(chars)
        for c in objects:c['bbox']=Bbox(c['bbox'])
        result=old._build_native_line_items_from_chars(objects,tuple(size),page_rotation=rotation,supported_angles=angles)
        record=dict(label=label,request=q,expected=[{k:clean(getattr(line,k))for k in names}for line in result])
        if pdf:record.update(pdf)
        cases.append(record)
    def chars(text,*,x=10.,y=25.,height=10.,width=6.,font=None,rotation=0.):
        font=font or dict(name='ABCDEF+Test-Regular',size=height,flags=0,weight=400)
        return [dict(char=t,bbox=[x+i*width,y,x+(i+1)*width,y+height],rotation=rotation,font=copy.deepcopy(font),char_idx=i)for i,t in enumerate(text)]
    texts=['','abc def','中文English','a\u00a0b\u2009c\u200bd','soft\u00ad\nwrap','line\x02\nnext','x\x00\x7fy','A.12','B。３','[12]','α=∑√±','𝛼²⁴','abc\n\nnext','only \t space','\r\n','\u2028\u2029','\ufeff\u2060','ﬁ ﬂ']
    for i,text in enumerate(texts):
        for angle in [0.,math.pi/2,math.pi,math.pi*1.5,math.radians(15),math.radians(44.9)]:
            add(f'text-{i}-{angle}',chars(text,rotation=angle))
    for ratio in [.34,.35,.5,.8,.81]:
        for gap in [-3.,0.,1.,3.,5.]:
            raw=chars('Body')+chars('2',x=34.+gap,y=21.,height=10.*ratio,width=4.)
            raw+=chars('[1]',x=0.,y=21.,height=10.*ratio,width=3.)
            add(f'scripts-{ratio}-{gap}',raw)
    for rotation in [0,90,180,270]:
        for supported in [(0.,),(90.,),(270.,),(0.,90.,180.,270.)]:
            add(f'page-angle-{rotation}-{supported}',chars('Rotate + 2',rotation=math.pi/2),rotation=rotation,angles=supported)
    for edge in [-20.,0.,230.,240.,250.]:
        add(f'clip-{edge}',chars('clip',x=edge))
    rng=random.Random(261026)
    for i in range(90):
        raw=[]
        for row in range(rng.randint(1,7)):
            h=rng.choice([5.,8.,10.,14.]);x=rng.choice([2.,10.,50.]);y=10.+row*18.
            chunk=chars(rng.choice(['text.', 'Amount 123','表头金额','α+β=2','a\u00a0b','[12]']),x=x,y=y,height=h,width=h*.6)
            for c in chunk:
                c['char_idx']=len(raw);c['font']['name']=rng.choice(['ABCDEF+Test-Regular','Test-Bold','Other']);raw.append(c)
        add(f'random-{i}',raw)
    if supplement:
        cases=[]
        for text in ['12','²³','1. Title','1.2 Heading','1.  two spaces','1.2.3.Test','１２． 标题','[１２]','before.12']:
            add('markers-'+text,chars(text))
        for offset in [0.,1.,3.]:
            raw=chars('Root',x=30.)+chars('2',x=54.+offset,y=21.,height=6.,width=4.)+chars('3',x=58.+offset,y=18.,height=3.,width=2.)
            add(f'chain-{offset}',raw)
        for ratio in [.2,.5,1.,2.]:
            raw=chars('Body')+chars('[1]',x=34.,y=19.,height=3.,width=2.)
            for c in raw:c['font']['size']*=ratio
            add(f'canonical-reference-{ratio}',raw)
        actual=dict(cases=[])
        for name in ['pdf-native-geometry','pdf-native-dedup-extra']:
            actual['cases'].extend(json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases'])
    else:actual=json.loads((ROOT/'build/migration/pdf-character-primitives.json').read_text(encoding='utf-8'))
    for i,c in enumerate(actual['cases']):
        if not c['extended']:continue
        e=c['expected']
        add(f'PDF-{i}',e['chars']if supplement else e['restored_chars'],e['page_size'],e['rotation'],pdf=dict({k:c[k]for k in ['pdf','pdf_sha256','page_index']},reader='deduplicated'if supplement else'restored'))
    # Full public LineItem defaults and mathematical category are generated from this pinned interpreter.
    assert unicodedata.unidata_version=='16.0.0'
    ranges=[]
    for n in range(0x110000):
        if unicodedata.category(chr(n))!='Sm':continue
        if ranges and ranges[-1][1]+1==n:ranges[-1][1]=n
        else:ranges.append([n,n])
    if not supplement:write_json(schema,defaults);write_json(unicode,ranges)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/native_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/groups.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/__init__.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},unicode=unicodedata.unidata_version,cases=cases))
    print('Frozen',len(cases),'complete visual-line cases;',sum('pdf'in c for c in cases),'actual PDF cases;',sum(len(c['expected'])for c in cases),'lines')
if __name__=='__main__':main()
