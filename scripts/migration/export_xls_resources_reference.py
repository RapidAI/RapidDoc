"""Freeze current Python XLS resources, full page ordering and shared OfficeArt."""
import dataclasses,gzip,json,struct,sys,zlib
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office'),str(ROOT/'scripts/migration')]
from export_xlsx_resources_reference import normalize
def main():
    import _legacy_xls_test_utils as fx
    import _mtef_test_utils as v3
    import _mtef_v5_test_utils as v5
    import _image_mtef_test_utils as comments
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.xls_converter import XlsConverter,_read_embedded_equations
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.parser import parse_xls_workbook
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy import officeart as art
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.records import RecordBudget
    from PIL import Image
    from loguru import logger
    logger.remove();cases=[];art_cases=[];r=fx.biff_record;ar=fx._officeart_record
    def capture(name,data):
        c=dict(name=name,package_hex=data.hex())
        try:
            with BoundedOleReader(data)as ole:
                stream=ole.read_stream('Workbook'if ole.has_stream('Workbook')else'Book');native=_read_embedded_equations(ole);book=parse_xls_workbook(stream,native_equations=native)
            c['sheets']=normalize([dict(name=s.name,cells=[dataclasses.asdict(v)for _,v in sorted(s.cells.items())],images=[dataclasses.asdict(v)for v in s.images],equations=[dataclasses.asdict(v)for v in s.equations],charts=[dataclasses.asdict(v)for v in s.charts])for s in book.sheets])
            engine=XlsConverter();engine.convert(BytesIO(data));c['pages']=normalize(engine.pages)
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    def shape(row,col,pib=None,hidden=False,ident=1):
        payload=ar(0xf00a,struct.pack('<II',ident,0),version=2)
        props=[]
        if pib is not None:props.append((0x0104,pib))
        if hidden:props.append((0x03bf,0x00020002))
        if props:payload+=ar(0xf00b,b''.join(struct.pack('<HI',*p)for p in props),version=3,instance=len(props))
        if row is not None:payload+=ar(0xf010,struct.pack('<9H',0,col,0,row,0,min(col+2,255),0,min(row+2,65535),0))
        return ar(0xf004,payload,version=15)
    def obj(kind=8,ident=1,state=None):
        data=struct.pack('<HHHH',0x15,4,kind,ident)
        if state is not None:data+=struct.pack('<HHH',0x12,2,state)
        return r(0x005d,data+struct.pack('<HH',0,0))
    def txo(text,font=0):
        data=text.encode('utf-16le');n=len(data)//2;formatting=struct.pack('<HHI',0,font,0)+struct.pack('<HHI',n,0,0)
        return r(0x1b6,b'\0'*10+struct.pack('<HH',n,len(formatting)))+r(0x3c,b'\1'+data)+r(0x3c,formatting)
    def bitmap(fmt='PNG',mode='RGB'):
        out=BytesIO();Image.new(mode,(11,7),'red').save(out,fmt);return out.getvalue()
    def blip(data,kind=0xf01e):return ar(kind,b'\0'*17+data)
    def store(blips):return ar(0xf001,b''.join(ar(0xf007,b'\0'*36+b,version=2)for b in blips),version=15)
    def workbook(records,bstore=b'',extra=b''):
        return fx.build_xls([fx.SheetFixture('Data',records)],globals_records=(r(0xeb,bstore)if bstore else b'')+extra)
    prefix=fx.label_cell(0,0,'header')+fx.label_cell(0,1,'value')+fx.number_cell(1,0,10)+fx.number_cell(1,1,20)
    for fmt,kind,mode in [('PNG',0xf01e,'RGB'),('PNG',0xf01e,'RGBA'),('JPEG',0xf01d,'RGB'),('TIFF',0xf029,'RGB'),('BMP',0xf01f,'RGB')]:
        data=bitmap(fmt,mode);data=data[14:]if fmt=='BMP'else data
        for row,col in [(1,1),(8,4)]:
            capture('picture '+repr((fmt,mode,row,col)),workbook(prefix+r(0xec,shape(row,col,1))+obj(),store([blip(data,kind)])))
    for kind in [6,11,8,99]:
        for state in [None,0,1,2]:
            capture('object '+repr((kind,state)),workbook(prefix+r(0xec,shape(1,1,1))+obj(kind,state=state)+txo('caption 😀',1),store([blip(bitmap())]),fx.font_record(bold=True)))
    for variant,drawing,objects in [
        ('hidden',shape(8,4,1,True),obj()),('no anchor',shape(None,0,1),obj()),
        ('missing PIB',shape(8,4,99),obj()),('no shapes',b'',obj()),('no objects',shape(8,4,1),b''),
        ('more objects',shape(8,4,1),obj()+obj(6,2)+txo('unbound')),
        ('more shapes',shape(8,4,1)+shape(10,4,1,ident=2),obj()),
        ('same image twice',shape(8,4,1)+shape(10,4,1,ident=2),obj()+obj(ident=2)),
    ]:capture(variant,workbook(prefix+r(0xec,drawing)+objects,store([blip(bitmap())])))
    for preview in [False,True]:
        for mtef in [v3.mtef_equation(v3.mtef_text('a+b')),v5.v5_equation(v5.v5_text('x+y')),b'bad MTEF']:
            capture('native equation '+str((preview,mtef[:6])),fx.build_equation_xls([(17,mtef)],preview=preview,cell_records=prefix))
    for flags in [0,2,0x10,0x20,0x80,0x100]:
        native=v3.mtef_equation(v3.mtef_text('native'))
        picture=fx._equation_obj(17,1).replace(struct.pack('<HHH',8,2,0),struct.pack('<HHH',8,2,flags))
        capture('embedding flags '+str(flags),fx.build_xls([fx.SheetFixture('Data',prefix+r(0xec,shape(8,4))+picture)],equations={17:native}))
    for format in ['GIF','WMF']:
        mtef=v3.mtef_equation(v3.mtef_text('x+y'))
        data=comments.build_gif_with_mtef(mtef)if format=='GIF'else comments.build_wmf([comments.pre6_wmf_comment(mtef)],placeable=True)
        stored=store([blip(data)])if format=='GIF'else fx._preview_bstore(data)
        for position in [(0,0),(8,4)]:capture('image equation '+str((format,position)),workbook(prefix+r(0xec,shape(*position,1))+obj(),stored))
    # A chart OBJ shares its shape/preview but consumes its own BIFF substream.
    for preview in [False,True]:
        for token in [b'\x25'+struct.pack('<HHHH',0,1,0,1),b'\x24'+struct.pack('<HH',0,1),b'\xff',b'']:
            chart=fx.biff_bof(0x20)+r(0x1051,b'\0'*6+struct.pack('<H',len(token))+token)+r(0xa)
            records=prefix+r(0xec,shape(8,4,1 if preview else None))+obj(5)+chart
            capture('embedded chart '+str((preview,token.hex())),workbook(records,store([blip(bitmap())])if preview else b''))
    # Compressed WMF, zlib/raw streams, declared image limit and later BSE.
    def metafile(payload,compression=0,declared=None,instance=0,kind=0xf01b):
        header=bytearray(34);struct.pack_into('<I',header,0,len(payload)if declared is None else declared);struct.pack_into('<ii',header,20,914400,457200);struct.pack_into('<I',header,28,len(payload));header[32]=compression
        return ar(kind,b'\0'*(32 if instance in [0x217,0x3d5]else 16)+header+payload,instance=instance)
    raw_wmf=comments.build_wmf([comments.baseline_wmf_comment()],placeable=True)
    for payload in [b'',b'bad',zlib.compress(raw_wmf),zlib.compress(raw_wmf)[2:-4],zlib.compress(raw_wmf)[:-2]]:
        for compression in [0,0xfe]:
            data=store([metafile(payload,compression),blip(bitmap())]);budget=RecordBudget()
            c=dict(name='BStore '+str((payload[:8],compression)),hex=data.hex())
            images=art.decode_bstore(data,charge=budget.charge);c['images']={str(i):dict(data_hex=p.data.hex(),extension=p.extension,content_type=p.content_type,render_size_emu=p.render_size_emu)for i,p in images.items()};c['visits']=budget.count;art_cases.append(c)
    for declared in [128*1024*1024+1,0]:
        data=store([metafile(b'bad',declared=declared),blip(bitmap())]);budget=RecordBudget();images=art.decode_bstore(data,charge=budget.charge);art_cases.append(dict(name='declared '+str(declared),hex=data.hex(),images={str(i):dict(data_hex=p.data.hex(),extension=p.extension,content_type=p.content_type,render_size_emu=p.render_size_emu)for i,p in images.items()},visits=budget.count))
    def art_store_case(name,data,delay=None):
        budget=RecordBudget();c=dict(name=name,hex=data.hex(),delay_hex=delay.hex()if delay is not None else None)
        images=art.decode_bstore(data,delay_stream=delay,charge=budget.charge)
        c.update(images={str(i):dict(data_hex=p.data.hex(),extension=p.extension,content_type=p.content_type,render_size_emu=p.render_size_emu)for i,p in images.items()},visits=budget.count);art_cases.append(c)
    for kind,instance in [(0xf01d,0x46b),(0xf01e,0x6e1),(0xf01e,0x6e3),(0xf01f,0x7a9),(0xf029,0x46b)]:
        data=ar(kind,b'\0'*33+bitmap(),instance=instance);art_store_case('double UID '+str((kind,instance)),store([data]))
    for reference in [0,1]:
        for offset in [0,3,0xffffffff]:
            delayed=blip(bitmap());header=bytearray(36);struct.pack_into('<III',header,20,len(delayed),reference,offset)
            art_store_case('delayed '+str((reference,offset)),ar(0xf001,ar(0xf007,header,version=2),version=15),b'pad'+delayed if offset==3 else delayed)
    for level in [0,1,63,64,65]:
        data=shape(1,1,1)
        for _ in range(level):data=ar(0xf002,data,version=15)
        budget=RecordBudget();c=dict(name='shapes depth '+str(level),hex=data.hex())
        try:c['shapes']=[dataclasses.asdict(s)for s in art.extract_excel_shapes(data,charge=budget.charge)]
        except Exception as e:c['error']=type(e).__name__
        c['visits']=budget.count;art_cases.append(c)
    for path in [ROOT/'demo/ms_office_docs/xlsx_01.xls',ROOT/'tests/migration/inputs/synthetic.xls']:
        if path.exists():capture('real '+path.name,path.read_bytes())
    for source in json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/xls_cells.json.gz').read_bytes()))['cases']:
        capture('cells '+source['name'],bytes.fromhex(source['package_hex']))
    dest=ROOT/'tests/fixtures/office_rust_reference_068/xls_resources.json.gz';dest.write_bytes(gzip.compress(json.dumps(dict(cases=cases,officeart=art_cases),ensure_ascii=True).encode(),mtime=0))
    print('XLS resources:',len(cases),'complete cases;',sum('error'in c for c in cases),'errors;',len(art_cases),'OfficeArt cases')
if __name__=='__main__':main()
