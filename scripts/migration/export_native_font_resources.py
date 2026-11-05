"""Freeze exact Python codec mappings and old CJK policy; immutable native resources."""
import array,hashlib,json,shutil,sys
from pathlib import Path
from common import ROOT,write_json,digest

def main():
    from rapid_doc._vendor.docvortex_core.document.pdf import font_runtime as old
    out=ROOT/'rust/rapiddoc-pdf/resources/fonts'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    origin=ROOT/'rapid_doc/_vendor/docvortex_core/resources/fonts'
    for name in ['DroidSansFallbackFull.ttf','manifest.json','NOTICE']:shutil.copyfile(origin/name,out/name)
    assert digest((out/old._FONT_FILE).read_bytes())==old._FONT_SHA256
    write_json(out/'policy.json',dict(aliases=old._FONT_ALIASES,suffixes=old._STYLE_SUFFIXES,casefold={chr(i):chr(i).casefold()for i in range(0x110000)if chr(i).casefold()!=chr(i)}))
    for codec in ['gb18030','big5','cp932','cp949']:
        values=array.array('I')
        for n in range(256+65536):
            b=bytes([n])if n<256 else bytes([(n-256)>>8,(n-256)&255])
            try:s=b.decode(codec);cp=ord(s)if len(s)==1 else 0xffffffff
            except UnicodeDecodeError:cp=0xffffffff
            values.append(cp)
        if sys.byteorder!='little':values.byteswap()
        (out/(codec+'.bin')).write_bytes(values.tobytes())
    ranges=[];last=None
    for i in range(126*10*126*10):
        a,n=divmod(i,10*126*10);b,n=divmod(n,126*10);c,d=divmod(n,10)
        try:cp=ord(bytes([a+0x81,b+0x30,c+0x81,d+0x30]).decode('gb18030'))
        except UnicodeDecodeError:last=None;continue
        if last is not None and last[1]+1==i and last[2]+i-last[0]==cp:last[1]=i
        else:last=[i,i,cp];ranges.append(last)
    write_json(out/'gb18030-ranges.json',ranges)
    upstream=Path('D:/CodeProjects/doc/DocVortex/rust/docvortex-pdfium/src/fonts.rs')
    dest=ROOT/'rust/rapiddoc-pdf/src/fonts/provider.rs';dest.parent.mkdir(exist_ok=True);shutil.copyfile(upstream,dest)
    write_json(out/'SOURCES.json',dict(python_version=sys.version,policy_source='rapid_doc/_vendor/docvortex_core/document/pdf/font_runtime.py',policy_sha256=digest((ROOT/'rapid_doc/_vendor/docvortex_core/document/pdf/font_runtime.py').read_bytes()),provider_source=str(upstream),provider_sha256=digest(upstream.read_bytes()),files={p.name:digest(p.read_bytes())for p in out.iterdir()if p.is_file()}))
    print('Frozen exact codec tables,',len(ranges),'GB18030 ranges and pinned CJK font')
if __name__=='__main__':main()
