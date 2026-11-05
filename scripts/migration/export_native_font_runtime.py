"""Independent old font classification and strict codec sequences; no PDFium in this exporter."""
import random
from common import ROOT,digest,write_json
def main():
    from rapid_doc._vendor.docvortex_core.document.pdf import font_runtime as old
    target=ROOT/'build/migration/native-fixed-font-runtime.json'
    if target.exists():raise FileExistsError(target)
    cases=[];decodes=[]
    def add(face,charset):cases.append(dict(face=list(face),charset=charset,expected=old._cjk_charset(face,charset)))
    for name in old._FONT_ALIASES:
        for text in [name,'@ABCDEF+'+name+'-Bold',name.upper(),name+'Regular',name+'x']:
            for encoding in ['utf-8','gb18030','big5','cp932','cp949']:
                try:b=text.encode(encoding)
                except UnicodeEncodeError:continue
                for charset in [0,1,2]:add(b,charset)
    for face in [b'',b'SimSun',b'Helvetica',b'private_GBK',b'UnknownBIG5',b'XGB2312',b'\xffSimSun',b'\x00SimSun']:
        for charset in [-1,0,1,2,128,129,134,136,178]:add(face,charset)
    for whitespace in ['\t','\r','\n','\x1c','\x1f','\u0085','\xa0','\u2003','\u202f','\u3000']:
        for text in [whitespace+'SimSun', 'Sim'+whitespace+'Sun','@ABCDEF+SimSun'+whitespace+'Bold','ſimſun','Kaiti']:
            add(text.encode(),1)
    rng=random.Random(2710279)
    for i in range(800):
        b=bytes(rng.randrange(256)for _ in range(rng.randrange(0,28)));add(b,rng.choice([0,1,2,128,134,178]))
        for n,codec in enumerate(['gb18030','big5','cp932','cp949']):
            try:s=b.decode(codec)
            except UnicodeDecodeError:s=None
            decodes.append(dict(codec=n,face=list(b),expected=s))
    for cp in [0x80,0x81,0x100,0x1234,0xFFFF,0x10000,0x10FFFF,0x9FB4,0x1F600]:
        b=('X'+chr(cp)+'SimSun').encode('gb18030');decodes.append(dict(codec=0,face=list(b),expected=b.decode('gb18030')));add(b,1)
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/font_runtime.py','rapid_doc/_vendor/docvortex_core/resources/fonts/manifest.json','rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual old _cjk_charset and strict Python codec decode sequences; all aliases, five encodings, subset/style/Unicode and charset gates.',cases=cases,decodes=decodes));print('Frozen font classifications',len(cases),'strict decode sequences',len(decodes))
if __name__=='__main__':main()
