"""冻结实际旧原生表格文字构建；字符几何作为独立输入，不改旧参考或vendor。"""
import copy,json,random,re,unicodedata
from dataclasses import asdict
from types import SimpleNamespace
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-text.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import text
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.pdf import prepare_table_page
    assert unicodedata.unidata_version=='16.0.0'
    ranges=[];start=None
    for n in range(0x110001):
        ok=n<0x110000 and chr(n).isprintable()
        if ok and start is None:start=n
        if not ok and start is not None:ranges.append((start,n-1));start=None
    decimals=[(n,unicodedata.decimal(chr(n)))for n in range(0x110000)if unicodedata.category(chr(n))=='Nd']
    source='//! Generated from CPython Unicode 16.0.0 by export_native_table_text.py.\n'
    source+='pub const PRINTABLE: &[(u32,u32)] = &[\n'+''.join(f'({a},{b}),\n'for a,b in ranges)+'];\n'
    source+='pub const DECIMALS: &[(u32,u8)] = &[\n'+''.join(f'({a},{b}),\n'for a,b in decimals)+'];\n'
    path=ROOT/'rust/rapiddoc-native/src/table_unicode.rs'
    if path.exists():assert re.sub(r'\s+','',path.read_text(encoding='utf-8'))==re.sub(r'\s+','',source)
    else:path.write_text(source,encoding='utf-8')
    cases=[]
    def add(chars,bbox=(0.,0.,200.,160.),angle=0):
        request=dict(chars=copy.deepcopy(chars),table_bbox=list(bbox),angle=angle)
        inp=NativeTableInput(tuple(bbox),(240.,200.),angle,tuple(copy.deepcopy(chars)))
        pending=[asdict(g)for g in text._select_pending_glyphs_python(inp)]
        value=text.build_native_table_text(inp)
        cases.append(dict(request=request,pending=pending,expected=None if value is None else asdict(value)))
    def char(s,b=(1.,1.,8.,11.),index=None,**extra):
        value=dict(char=s,bbox=list(b),**extra)
        if index is not None:value['char_idx']=index
        return value
    for s in ['',' ','\r','\n','\r\n','\x1c','\u00a0','\ufb00','\ufb03','\ufb05','“—”\x02','中','㐀','\ua000','A','🙂','\u200b','\U0001fae9']:
        for h in [.49,.5,.51]:add([char('A'),char(' ',(10.,1.,12.,11.)),char(s,(15.,1.,20.,1+h)),char('B',(30.,1.,38.,11.))])
    for angle in [0,90,180,270,-90,450,90.9,'１８０','9_0',None,'bad']:
        add([char('B',(30.,40.,38.,50.),'1_0',loose_bbox=[29.,38.,39.,52.]),char(' ',(20.,40.,25.,50.),2),char('A',(10.,40.,18.,50.),1),char('中',(80.,20.,90.,30.),True)],angle=angle)
    for index in [None,False,True,1.9,-2.9,'10','  -２  ','١_٢','bad',[],{},'1.5']:
        c=char('A');c['char_idx']=index;add([char('B',(30.,1.,40.,11.),2),c])
    for box in [[0.,0.,0.,10.],[10.,10.,0.,0.],[-5.,0.,5.,10.],[195.,10.,205.,20.],[200.,10.,202.,20.],None,[],['1','2','3','4']]:
        add([dict(char='x',bbox=box),char('A')])
    add([]);add([char('中',(0.,0.,.1,.1))],bbox=(5.,5.,0.,0.))
    # 重复来源的loose box按原输入最后一次有效赋值覆盖；None局部裁剪也覆盖。
    for loose in [[0.,0.,0.,0.],[1000.,1000.,1010.,1010.],[25.,0.,40.,15.]]:
        add([char('a',(10.,2.,15.,12.),1,loose_bbox=[9.,0.,16.,15.]),char('.',(17.,10.,18.,12.),2,loose_bbox=[16.,0.,19.,15.]),char('b',(30.,2.,35.,12.),1,loose_bbox=loose)])
    rng=random.Random(8531);pool=['A','B','中','文','1','.',',','\r','\n',' ','\x1c','“','—','ﬃ','\u200b','🙂','\u0378']
    for _ in range(256):
        chars=[]
        for i in range(rng.randrange(1,65)):
            x,y=rng.randrange(-10,190),rng.randrange(-10,145);w,h=rng.choice([.3,1.,5.,10.]),rng.choice([.4,.5,2.,10.,14.])
            c=char(rng.choice(pool),(float(x),float(y),x+w,y+h),rng.choice([i,i//2,str(i),None]))
            if rng.random()<.4:c['loose_bbox']=[float(x),float(y-1),x+w+3,y+h+2]
            chars.append(c)
        add(chars,angle=rng.choice([0,90,180,270]))
    # 直接采用旧DocVortex实际PDF字符几何，区别于pdftext与给定合成响应。
    real=[]
    pdf=ROOT/'build/migration/model-document/five-pages.pdf'
    with PDFDocument(pdf.read_bytes())as doc:
        for page_index,bbox in [(0,(0.,0.,240.,200.)),(1,(0.,0.,240.,200.)),(2,(18.,38.,222.,162.)),(3,(130.,30.,200.,70.))]:
            p=prepare_table_page(doc[page_index]);chars=[]
            for old in p.geometry.chars:
                c={}
                for key in ['char','char_idx','bbox','loose_bbox']:
                    if key in old:
                        value=old[key];c[key]=list(value.bbox)if hasattr(value,'bbox')else value
                chars.append(c)
            for angle in [0,90,180,270]:
                before=len(cases);add(chars,bbox,angle);real.append(dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=digest(pdf.read_bytes()),page_index=page_index,case_index=before))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/spatial_text.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_geometry.py']
    write_json(target,dict(cases=cases,real=real,sources={p:digest((ROOT/p).read_bytes())for p in refs},unicode=unicodedata.unidata_version,unicode_sha256=digest(source.encode('utf-8')),unicode_hash_format='canonical_generator_text_lf',scope='Actual old pending glyph selection and build_native_table_text with supplied JSON chars, including real DocVortex PDF geometry at four angles. No Rust DocVortex PDF glyph extraction, cell text reconstruction or table candidate generation is claimed.'))
    print(f'Frozen {len(cases)} old native table text cases including {len(real)} actual PDF/angle inputs; Unicode16 printable and decimal data')
if __name__=='__main__':main()
