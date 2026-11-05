"""Freeze the original Python codecs, avoiding WHATWG/Windows codec drift."""
from pathlib import Path
import gzip,struct,json
ROOT=Path(__file__).resolve().parents[2]
def main():
    data=bytearray(b'RDCS1')
    for codec in ['shift_jis','cp949','gb18030','big5']:
        for length,count in [(1,256),(2,65536)]:
            for n in range(count):
                raw=n.to_bytes(length,'big')
                try:
                    value=raw.decode(codec)
                    scalar=ord(value) if len(value)==1 else 0xffffffff
                except UnicodeDecodeError:scalar=0xffffffff
                data.extend(struct.pack('<I',scalar))
    ranges=[]
    for n in range(126*10*126*10):
        n0=n;fourth=n0%10+0x30;n0//=10;third=n0%126+0x81;n0//=126;second=n0%10+0x30;first=n0//10+0x81
        try:scalar=ord(bytes([first,second,third,fourth]).decode('gb18030'))
        except UnicodeDecodeError:continue
        if ranges and ranges[-1][1]+1==n and ranges[-1][2]+n-ranges[-1][0]==scalar:ranges[-1][1]=n
        else:ranges.append([n,n,scalar])
    data.extend(struct.pack('<I',len(ranges)))
    for record in ranges:data.extend(struct.pack('<3I',*record))
    target=ROOT/'rust/rapiddoc-office/src/metafile/ansi_dbcs.bin.gz';target.write_bytes(gzip.compress(data,mtime=0))
    codecs={0:'cp1252',1:'cp1252',161:'cp1253',162:'cp1254',163:'cp1258',177:'cp1255',178:'cp1256',186:'cp1257',204:'cp1251',222:'cp874',238:'cp1250',255:'cp437'}
    single={str(charset):[ord(bytes([i]).decode(codec,errors='replace')) for i in range(256)] for charset,codec in codecs.items()}
    (ROOT/'rust/rapiddoc-office/src/metafile/ansi_singlebyte.json').write_text(json.dumps(single,separators=(',',':'))+'\n')
    print('Original codec tables:',len(data),'bytes;',len(ranges),'GB18030 ranges;',target.stat().st_size,'compressed bytes')
if __name__=='__main__':main()
