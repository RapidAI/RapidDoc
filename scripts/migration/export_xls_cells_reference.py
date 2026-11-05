"""Original BIFF5/8 workbook cells and full cell-only projection references."""
from io import BytesIO
from pathlib import Path
import dataclasses,gzip,json,struct,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
def main():
    import _legacy_xls_test_utils as fx
    from _legacy_ppt_test_utils import _build_cfb
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.legacy.ole import BoundedOleReader
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.parser import parse_xls_workbook
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.xls_converter import XlsConverter,_XlsPageBuilder
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.records import iter_records
    from loguru import logger
    logger.remove();cases=[];r=fx.biff_record
    def capture(name,data,stream=None):
        c=dict(name=name,package_hex=data.hex())
        try:
            if stream is None:
                with BoundedOleReader(data)as ole:stream=ole.read_stream('Workbook'if ole.has_stream('Workbook')else'Book')
            c['stream_hex']=stream.hex()
            book=parse_xls_workbook(stream)
            final_cells=[[dataclasses.asdict(cell)for _,cell in sorted(s.cells.items())]for s in book.sheets]
            # Compare the BIFF cell checkpoint before OfficeArt text boxes and
            # controls append text. Full resource results remain frozen too.
            with patch('rapid_doc._vendor.docvortex_core.analyzers.native.office.xls.parser._bind_objects',lambda *a,**kw:None):
                book=parse_xls_workbook(stream)
            c['after_resources_cells']=final_cells
            c['active_sheet_index']=book.active_sheet_index
            c['chart_sheets']=[dataclasses.asdict(s)for s in book.chart_sheets]
            c['sheets']=[dict(name=s.name,visible=s.visible,order=s.order,recovered=s.recovered,merges=s.merges,cells=[dataclasses.asdict(cell)for _,cell in sorted(s.cells.items())])for s in book.sheets]
            resource=any(rec.record_type in {0x00eb,0x00ec,0x005d,0x01b6}for rec in iter_records(stream))
            c['has_resources']=resource
            if not resource:
                engine=XlsConverter();engine.convert(BytesIO(data));c['pages']=engine.pages
                builder=_XlsPageBuilder(book);adapter=builder._build_openpyxl_workbook();projected=[]
                for sheet in adapter.worksheets:
                    predicates,_=builder._sheet_semantic_predicates(sheet)
                    projected.append(dict(name=sheet.title,cells=[dict(row=row-1,col=col-1,has_value=cell.value is not None,semantic=predicates(row-1,col-1),cell=builder._build_excel_cell(sheet,0,0,row-1,col-1).model_dump())for(row,col),cell in sorted(list(sheet._cells.items()))]))
                c['projected']=projected
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    def workbook(records,extra=b'',**kw):return fx.build_xls([fx.SheetFixture('Data',records)],globals_records=extra,**kw)
    def from_stream(raw):return _build_cfb([('Workbook',raw)])
    def stream(data):
        with BoundedOleReader(data)as ole:return ole.read_stream('Workbook'if ole.has_stream('Workbook')else'Book')
    base_records=fx.label_cell(0,0,'header')+fx.number_cell(1,0,1.005)+fx.formula_number_cell(1,1,21)+fx.formula_string_cell(2,1,'cached')+fx.merged_cells((3,0,3,2))+fx.label_cell(3,0,'merged')
    capture('ordinary',workbook(base_records))
    capture('recovered offset',workbook(base_records,corrupt_first_offset=True))
    capture('encryption',workbook(base_records,encrypted=True))
    capture('BIFF5 cp1252',fx.build_biff5_xls('café €'))
    capture('visible empty hidden',fx.build_xls([fx.SheetFixture('Visible',fx.label_cell(0,0,'ok')),fx.SheetFixture('Empty'),fx.SheetFixture('Secret',fx.label_cell(0,0,'hidden'),False)]))
    for names in [('Same','Same'),('same','Same'),('',''),('Bad/Name','Data'),('😀','😀')]:capture('names '+repr(names),fx.build_xls([fx.SheetFixture(n,fx.label_cell(0,0,str(i)))for i,n in enumerate(names)]))
    for count in [0,1,2]:
        extra=fx.font_record(bold=True)+fx.font_record(italic=True)
        sst=fx.continued_rich_sst('A😀B end',[(0,1),(3,2)])if count else fx.rich_sst([('A😀B end',[(0,1),(3,2)])])
        capture('rich SST '+str(count),workbook(fx.labelsst_cell(0,0,0),extra+sst))
    for starts in [[(0,1),(3,2)],[(3,1),(0,2)],[(0,99)],[(1,1),(2,2)]]:
        raw='A😀B'.encode('utf-16le')
        body=struct.pack('<HHHHB',0,0,0,len(raw)//2,1)+raw+struct.pack('<H',len(starts))+b''.join(struct.pack('<HH',*s)for s in starts)
        capture('RSTRING '+repr(starts),workbook(r(0xd6,body),fx.font_record(bold=True)+fx.font_record(italic=True)))
    # Compressed text changes width in a CONTINUE; only the first segment
    # carries the cell header, and subsequent rich fields have no width byte.
    capture('continued LABEL',workbook(r(0x0204,struct.pack('<HHHHB',0,0,0,4,0)+b'AB')+r(0x3c,b'\1C\0D\0')))
    for fmt in ['0.00','#,##0.00','0%','0.00E+00','# ?/?','yyyy-mm-dd','[h]:mm:ss','0;0;0;"text:"@']:
        encoded=fmt.encode('utf-16le');extra=r(0x041e,struct.pack('<HHB',164,len(encoded)//2,1)+encoded)+r(0x00e0,struct.pack('<HH',0,164)+b'\0'*16)
        for epoch in [False,True]:
            records=b''.join(fx.number_cell(i,0,v,xf_index=1)for i,v in enumerate([0,1.005,-2.675,60.5,45292.5]))+fx.label_cell(6,0,'text',xf_index=1)
            capture('format '+str((fmt,epoch)),workbook(records,extra+r(0x0022,struct.pack('<H',epoch))))
    for code in [0,7,15,23,29,36,42,43,255]:
        cells=r(0x0205,struct.pack('<HHHBB',0,0,0,code,1))+r(0x0205,struct.pack('<HHHBB',0,1,0,code,0))
        for kind in [0,1,2,3,99]:
            cached=bytes([kind,0,code,0,0,0,255,255]);cells+=r(0x0006,struct.pack('<HHH',kind+1,0,0)+cached)
            if kind==0:cells+=r(0x0207,struct.pack('<HB',2,0)+b'ok')
        capture('bool error '+str(code),workbook(cells))
    for text in ['=DISPIMG("id",1)','=_xlfn.DISPIMG("id",1)','normal','', ' =dispimg("id",1)']:
        capture('formula cache '+repr(text),workbook(fx.formula_string_cell(0,0,text)))
    for v in [0,1,2,3,0xffffffff,0x80000002,0x7ffffffe,0x3ff00000]:
        rk=r(0x027e,struct.pack('<HHHI',0,0,0,v))
        mul=r(0x00bd,struct.pack('<HHHIHIH',1,0,0,v,0,v,1))
        capture('RK '+str(v),workbook(rk+mul))
    for address in ['https://example.com/a?x=1&b=2','mailto:a@b','tel:123','javascript:x','#Data!A1','../relative','file:///c:/a']:
        capture('link '+address,workbook(fx.label_cell(0,0,'label')+fx.url_hyperlink(0,0,'label',address)))
    for records in [r(0x0204,struct.pack('<HHHHB',0,256,0,1,0)+b'X'),r(0x0203,b'\0'*5),fx.label_cell(0,0,'first')+fx.label_cell(0,0,''),fx.label_cell(0,0,'first')+fx.label_cell(0,0,'last')]:capture('cell boundary '+records.hex(),workbook(records))
    for merges in [[(0,0,2,2)],[(3,3,1,1)],[(0,0,0,0)],[(0,250,1,300)],[(0,256,1,300)],[(0,0,1,1),(1,1,2,2)]]:
        capture('merge '+repr(merges),workbook(fx.label_cell(0,0,'origin')+fx.label_cell(1,1,'inside')+fx.merged_cells(*merges)))
    capture('original merge budget',workbook(fx.label_cell(0,0,'value')+fx.merged_cells((0,0,65535,255))))
    capture('cumulative original merge budget',fx.build_xls([fx.SheetFixture('One',fx.merged_cells((0,0,8191,255))),fx.SheetFixture('Two',fx.merged_cells((0,0,8191,255)))]))
    raw=stream(workbook(base_records));positions=list(iter_records(raw))
    for name,mutate in [
        ('no directory',lambda b:b''.join(r(rec.record_type,rec.payload)for rec in iter_records(b)if rec.record_type!=0x85)),
        ('unsupported version',lambda b:b[:4]+b'\x00\x04'+b[6:]),
        ('wrong globals kind',lambda b:b[:6]+b'\x10\x00'+b[8:]),
        ('zero tail',lambda b:b+b'\0'*4096),
        ('nested globals',lambda b:b[:20]+fx.biff_bof(0x10)+r(0x002f,b'\0\0')+r(0x0a)+b[20:]),
    ]:capture(name,from_stream(mutate(raw)))
    bounds=[rec for rec in positions if rec.record_type==0x85]
    capture('unreadable extra descriptor',from_stream(raw[:bounds[0].offset]+r(0x85,struct.pack('<IBBBB',0x7ffffff0,0,0,1,0)+b'X')+raw[bounds[0].offset:]))
    for cut in [0,1,4,6,19,len(raw)-1,len(raw)-5,len(raw)-15,len(raw)//2]:capture('truncate '+str(cut),from_stream(raw[:cut]))
    for length in [0,4096]:capture('long plain '+str(length),workbook(fx.label_cell(0,0,'x'*length)))
    # A visible data sheet and a standalone chart sheet using simple BRAI refs.
    ext=r(0x01ae,struct.pack('<HH',2,0x0401))+r(0x0017,struct.pack('<HHHH',1,0,0,0))
    def chart_sheet(tokens,visible=True):
        formula=r(0x1051,b'\0'*6+struct.pack('<H',len(tokens))+tokens)
        package=fx.build_xls([fx.SheetFixture('Data',fx.label_cell(0,0,'key')+fx.label_cell(0,1,'value')+fx.number_cell(1,0,1)+fx.number_cell(1,1,2)),fx.SheetFixture('Chart',formula,visible)],globals_records=ext)
        data=bytearray(stream(package));bounds=[rec for rec in iter_records(data)if rec.record_type==0x85];chart=bounds[1];offset=struct.unpack_from('<I',chart.payload)[0];data[chart.offset+4+5]=2;data[offset+4+2:offset+4+4]=struct.pack('<H',0x20)
        return from_stream(bytes(data))
    for token in [b'\x3b'+struct.pack('<HHHHH',0,0,1,0,1),b'\x3a'+struct.pack('<HHH',0,0,1),b'\x25'+struct.pack('<HHHH',0,1,0,1),b'\x3b'+struct.pack('<HHHHH',99,0,1,0,1),b'\x3b',b'']:
        capture('chart sheet '+token.hex(),chart_sheet(token))
    capture('hidden chart sheet',chart_sheet(b'\x3b'+struct.pack('<HHHHH',0,0,1,0,1),False))
    for path in [ROOT/'demo/ms_office_docs/xlsx_01.xls',ROOT/'tests/migration/inputs/synthetic.xls']:
        if path.exists():capture('real '+path.name,path.read_bytes())
    dest=ROOT/'tests/fixtures/office_rust_reference_068/xls_cells.json.gz';dest.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('XLS cells:',len(cases),'workbooks;',sum(len(c.get('sheets',[]))for c in cases),'sheets;',sum('error'in c for c in cases),'errors;',sum('pages'in c for c in cases),'full cell pages')
if __name__=='__main__':main()
