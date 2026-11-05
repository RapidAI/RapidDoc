"""Freeze original BIFF records/continuations, rich strings and number formats."""
import dataclasses,gzip,hashlib,json,random,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls import number_format as nf,strings as st,records as rec,chart
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.models import XlsFontStyle
    from loguru import logger
    logger.remove();rng=random.Random(73418);numbers=[];texts=[];strings=[];records=[];codepages=[];txo=[];codec_pair_hashes=[];charts=[]
    formats=[None,'[>=١٢]0.00;0','[<１２.５]0;0.00']+list(nf.BUILTIN_NUMBER_FORMATS.values())+['General','0.###','00000','???.???','0,,','0.00%%','[>=100]0.00;[<0](0.0);"zero"','0;[Red](0);"-";"text:"@','"金额："0.00" 元"','[$USD-409]#,##0.00','0.00E-00','000.00E+000','# ?/16','?/8','?/?','0.00;0.00','[h]:mm:ss','[ss]','yyyy-mm-dd hh:mm:ss','mmss.0','[foo]0','0"a;b"','0\\;0','[bad','"bad','0;0;0;0;0','[>1]0;[>2]0;[>3]0','[Blue]@','"pre"@"post"','0.0_);(0.0)','0.00,','0.00000000000000000000000000000']
    values=[0.,-0.,1.,-1.,.5,-.25,1.005,2.675,9.995,99.999,999.9999,1e-5,1e-4,1e14,1e15,1e20,1e100,1e-300,1e308,59.,60.,60.5,61.,45292.99999999999,1/3,1/7,3.5,1234567.89123,float('inf'),float('-inf'),float('nan')]
    def capture(v,fmt,epoch):
        c=dict(bits=struct.pack('<d',v).hex(),code=fmt,date1904=epoch)
        try:c['output']=nf.format_number(v,fmt,date1904=epoch)
        except Exception as e:c['error']=type(e).__name__
        numbers.append(c)
    for fmt in formats:
        for v in values:
            for epoch in [False,True]:capture(v,fmt,epoch)
        for s in ['plain','a\r\nb\x00','😀 汉字','@','']:
            texts.append(dict(text=s,code=fmt,output=nf.format_text(s,fmt)))
    for _ in range(1200):capture(rng.uniform(-10,10)*10.**rng.randint(-300,300),rng.choice(formats),rng.choice([False,True]))
    fonts=[XlsFontStyle(),XlsFontStyle(bold=True),XlsFontStyle(italic=True,underline=True),XlsFontStyle(strike=True,superscript=True),XlsFontStyle(subscript=True)]
    def string_case(parts,short=False,rich=True,biff8=True,cp=1252):
        c=dict(parts=[b.hex()for b in parts],short=short,rich=rich,biff8=biff8,codepage=cp)
        reader=rec.SegmentReader(parts)
        d=st.read_biff8_string(reader,short=short,rich=rich)if biff8 else st.read_byte_string(reader,short=short,encoding=st.codepage_name(cp))
        c.update(decoded=dataclasses.asdict(d)if d else None,rich_text=dataclasses.asdict(st.to_rich_text(d,fonts))if d else None,index=reader.segment_index,offset=reader.offset)
        strings.append(c)
    for text in ['','plain','中文😀é','A\r\nB\rC\x00\x07\tZ','é'*14]:
        raw=text.encode('utf-16le');count=len(raw)//2
        for short in [False,True]:
            header=struct.pack('<B'if short else'<H',count)+b'\x01'
            full=header+raw
            string_case([full],short=short)
            for cut in range(len(header),len(full)+1):
                string_case([full[:cut],b'\x01'+full[cut:]],short=short)
            for length in range(len(full)):string_case([full[:length]],short=short)
    for payload in [b'\x00\xd8',b'\x00\xdc',b'\x00\xd8A\x00',b'\x3d\xd8\x00\xde']:
        string_case([struct.pack('<H',len(payload)//2)+b'\x01'+payload])
    for first,second in [(b'AB',b'C\x00D\x00'),(b'A\x00B\x00',b'CD')]:
        string_case([b'\x04\x00'+bytes([1 if len(first)==4 else 0])+first,bytes([1 if len(second)==4 else 0])+second])
    for runs in [[(0,1),(1,2),(3,5)],[(3,3),(0,1),(1,4),(1,2)],[(0,99),(65535,1)],[(1,3),(2,5)]]:
        raw='A😀B'.encode('utf-16le');body=struct.pack('<H',4)+b'\x09'+struct.pack('<H',len(runs))+raw+b''.join(struct.pack('<HH',*r)for r in runs)
        string_case([body])
        for cut in range(3+2+len(raw),len(body)+1):string_case([body[:cut],body[cut:]])
    for cp in [874,932,936,949,950,1200,*range(1250,1259),0]:
        for raw in [bytes(range(256)),b'\x81\x80\xffA\x00',b'\x00\xd8A',*([bytes(rng.randrange(256)for _ in range(17))for _ in range(40)])]:
            codepages.append(dict(codepage=cp,hex=raw.hex(),output=raw.decode(st.codepage_name(cp),'replace')))
            string_case([struct.pack('<H',len(raw))+raw],biff8=False,cp=cp)
        digest=hashlib.sha256()
        for a in range(256):
            for b in range(256):
                output=bytes([a,b]).decode(st.codepage_name(cp),'replace').encode('utf-8')
                digest.update(struct.pack('<I',len(output)));digest.update(output)
        codec_pair_hashes.append(dict(codepage=cp,sha256=digest.hexdigest()))
    for c in range(0xac00,0xd7a4):
        raw=chr(c).encode('euc_kr')
        codepages.append(dict(codepage=949,hex=raw.hex(),output=chr(c)))
        if len(raw)==8 and c%37==0:
            for i in range(2,8):
                b=raw[:i];codepages.append(dict(codepage=949,hex=b.hex(),output=b.decode('euc_kr','replace')))
            for tail in [b'AA',b'\x00\xff',b'\xa4\xff']:
                b=raw[:6]+tail;codepages.append(dict(codepage=949,hex=b.hex(),output=b.decode('euc_kr','replace')))
    korean=json.loads(gzip.decompress((ROOT/'rust/rapiddoc-office/src/xls/codepages/949.json.gz').read_bytes()))
    for raw,expected in korean['hangul'].items():
        assert bytes.fromhex(raw).decode('euc_kr')==expected
        codepages.append(dict(codepage=949,hex=raw,output=expected))
    for text in ['','abc','中文😀','A\r\n B']:
        payload=text.encode('utf-16le');count=len(payload)//2
        for cut in range(0,len(payload)+1,2):
            runs=struct.pack('<HHI',0,1,0)+struct.pack('<HHI',count,0,0)
            base=b'\0'*10+struct.pack('<HH',count,len(runs))
            parts=[b'\1'+payload[:cut],b'\1'+payload[cut:],runs]
            for n in range(4):
                selected=parts[:n];out=st.read_txo_text(base,selected,fonts)
                txo.append(dict(base=base.hex(),parts=[p.hex()for p in selected],output=dataclasses.asdict(out)if out else None))
    blobs=[b'',b'\0'*8,struct.pack('<HH',1,2)+b'AB',struct.pack('<HH',1,3)+b'AB',struct.pack('<HH',1,0)+struct.pack('<HH',rec.CONTINUE,1)+b'X'+struct.pack('<HH',rec.EOF,0),b'\0'*4+struct.pack('<HH',2,0)]
    for data in blobs:
        all_records=[dict(offset=r.offset,kind=r.record_type,payload=r.payload.hex(),next=r.next_offset)for r in rec.iter_records(data)]
        records.append(dict(hex=data.hex(),records=all_records))
    tokens=[]
    for kind in [4,5,0x1a,0x1b]:
        for klass in [0x20,0x40,0x60]:
            for reverse in [False,True]:
                first,last=(4,1)if reverse else(1,4)
                if kind==4:raw=struct.pack('<BHH',kind|klass,first,0xc003)
                elif kind==5:raw=struct.pack('<BHHHH',kind|klass,first,last,0xc000|first,last)
                elif kind==0x1a:raw=struct.pack('<BHHH',kind|klass,0,first,0xc003)
                else:raw=struct.pack('<BHHHHH',kind|klass,0,first,last,0xc000|first,last)
                tokens.append(raw)
    tokens += [b'',b'\0',b'\x3b',b'\x3a'+struct.pack('<HHH',99,1,2),b'\x24'+struct.pack('<HH',65535,255),b'\x24'+struct.pack('<HH',1,3)+b'ignored-tail']
    for token in tokens:
        for current in [0,1,2]:
            for extern in [[0],[1],[None],[]]:
                for malformed in [False,True]:
                    payload=b'\0'*6+struct.pack('<H',len(token)+(1 if malformed else 0))+token
                    record=rec.BiffRecord(0,0x1051,payload,len(payload)+4)
                    selected=chart.chart_source_selection([record],current_sheet_index=current,extern_sheets=extern)
                    axes=chart.chart_source_axes([record],current_sheet_index=current,extern_sheets=extern)
                    charts.append(dict(payloads=[payload.hex()],current=current,external=extern,selection=dataclasses.asdict(selected)if selected else None,axes=axes))
    fixture=dict(numbers=numbers,texts=texts,strings=strings,fonts=[dataclasses.asdict(f)for f in fonts],records=records,codepages=codepages,codec_pair_hashes=codec_pair_hashes,txo=txo,charts=charts)
    path=ROOT/'tests/fixtures/office_rust_reference_068/xls_primitives.json.gz';path.write_bytes(gzip.compress(json.dumps(fixture,ensure_ascii=True).encode(),mtime=0))
    print({k:len(v)for k,v in fixture.items()})
if __name__=='__main__':main()
