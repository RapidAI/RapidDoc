"""Original openpyxl/XLSX cell load through native spreadsheet projection reference."""
import copy,gzip,json,sys,zipfile
from pathlib import Path
from io import BytesIO
from datetime import datetime,date,time,timedelta
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
    from openpyxl import Workbook,load_workbook
    from openpyxl.cell.cell import Cell
    from openpyxl.styles import Font,PatternFill,Alignment,GradientFill
    from openpyxl.cell.rich_text import CellRichText,TextBlock
    from openpyxl.cell.text import InlineFont
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.spreadsheet.projector import SpreadsheetProjector
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.spreadsheet.html import render_spreadsheet_table
    from loguru import logger
    logger.remove();cases=[]
    def capture(name,data):
        c=dict(name=name,package_hex=data.hex())
        try:
            book=load_workbook(BytesIO(data),data_only=True,rich_text=True);sheets=[]
            for sheet in book.worksheets:
                projector=SpreadsheetProjector();projector.workbook=book;semantic,span=projector._sheet_semantic_predicates(sheet)
                default_cell=dict(row=0,col=0,text='',row_span=0,col_span=0,styles=projector._extract_cell_style(Cell(sheet)),media=[],equations=[],text_is_html=False,source_row=None,source_col=None)
                cells=[]
                for(r,col),cell in sorted(list(sheet._cells.items())):cells.append(dict(row=r-1,col=col-1,has_value=cell.value is not None,semantic=semantic(r-1,col-1),cell=projector._build_excel_cell(sheet,0,0,r-1,col-1).model_dump()))
                selected,penalty,tables=projector._select_best_gap_candidate(sheet)
                sheets.append(dict(name=sheet.title,state=sheet.sheet_state,cells=cells,default_cell=default_cell,selected_gap=selected,penalty=penalty,tables=[t.model_dump()for t in tables],html=[render_spreadsheet_table(t)for t in tables]))
            c.update(date1904=book.epoch.year==1904,sheets=sheets);book.close()
            # Complete package -> pages comparisons on documents without the
            # drawing/OLE layer that has deliberately not been connected yet.
            with zipfile.ZipFile(BytesIO(data))as z:
                resource_parts=any(n.startswith('xl/drawings/')or n.startswith('xl/embeddings/')or n=='xl/cellimages.xml'for n in z.namelist())
            if not resource_parts:
                from rapid_doc._vendor.docvortex_core.analyzers.native.office.xlsx.xlsx_converter import XlsxConverter
                full=[]
                for hidden,gap,singleton in [(False,None,True),(True,None,True),(False,0,False)]:
                    engine=XlsxConverter(include_hidden_sheets=hidden,gap_tolerance=gap,treat_singleton_as_text=singleton);engine.convert(BytesIO(data));full.append(dict(include_hidden=hidden,gap=gap,singleton=singleton,pages=engine.pages))
                c['full_cells_queries']=full
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    def data(book):out=BytesIO();book.save(out);return out.getvalue()
    def mutate(raw,path,replace):
        out=BytesIO()
        with zipfile.ZipFile(BytesIO(raw))as z,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED)as dest:
            for info in z.infolist():
                content=z.read(info)
                if info.filename==path:content=replace(content.decode()).encode()
                dest.writestr(info,content)
        return out.getvalue()
    for epoch in [False,True]:
        for iso in [False,True]:
            wb=Workbook();wb.iso_dates=iso
            if epoch:
                from openpyxl.utils.datetime import MAC_EPOCH
                wb.epoch=MAC_EPOCH
            ws=wb.active;ws.title='类型';ws.append(['value','kind'])
            for row,v in enumerate([0,1,-2,1.25,1e-7,1e20,False,True,'plain & < > " \'','a\r\nb\rc\nd',datetime(2024,1,2),date(2024,1,3),time(12,34,56,789000),timedelta(days=3,seconds=111,microseconds=123000),'=2+3','DISPIMG unknown','  '],2):ws.append([v,str(type(v))])
            ws['C2']=CellRichText(['plain ',TextBlock(InlineFont(b=True,i=True,u='single',strike=True,vertAlign='superscript'),' styled text '),' trailing'])
            ws['C3']=CellRichText([TextBlock(InlineFont(vertAlign='subscript'),'A'),TextBlock(InlineFont(b=True),'B')]);ws['C4']=CellRichText(['a','b'])
            ws['C5']='escaped _x005F_ text';ws['D2']='link\nlabel';ws['D2'].hyperlink='https://example.com/a?x=1&b=2';ws['D3'].hyperlink='https://example.com/empty'
            ws['D4']='unsafe';ws['D4'].hyperlink='javascript:alert(1)';ws['D5']='internal'
            from openpyxl.worksheet.hyperlink import Hyperlink
            ws['D5'].hyperlink=Hyperlink(ref='D5',location="'类型'!A1")
            for coord in ['A2','B3','C4']:
                ws[coord].font=Font(b=True,i=True,u='double',strike=True,color='12AB34');ws[coord].fill=PatternFill('solid',fgColor='FEDCBA');ws[coord].alignment=Alignment(horizontal='center',vertical='top')
            ws['B5'].fill=GradientFill(stop=['000000','FFFFFF']);ws.merge_cells('F2:H4');ws['F2']='merged';ws['G7']='new link';ws['G7'].hyperlink='mailto:a@example.com'
            hidden=wb.create_sheet('Hidden');hidden.sheet_state='hidden';hidden['A1']='hidden';empty=wb.create_sheet('Empty');empty['XFD1000'].font=Font(b=True)
            capture('types '+str((epoch,iso)),data(wb))
    for fmt in ['yyyy-mm-dd','[h]:mm:ss','mm:ss','[HH]:mm','0.00','0 "days"','mmss.0']:
        wb=Workbook();ws=wb.active
        for r,v in enumerate([0.,.5,59.,60.,61.,-.25,45292.123456,3000000.],1):ws.cell(r,1,v).number_format=fmt
        capture('numeric '+fmt,data(wb))
    wb=Workbook();ws=wb.active;ws['A1']='header';ws['A2']=10;ws['B2']='text';ws['C2']='another';ws.merge_cells('E2:G4');ws['E2']='anchor';base=data(wb)
    mutations=[('float','xl/worksheets/sheet1.xml',lambda s:s.replace('<v>10</v>','<v>1.0</v>')),('cached formula','xl/worksheets/sheet1.xml',lambda s:s.replace('<v>10</v>','<f>2*5</f><v>10</v>')),('uncached formula','xl/worksheets/sheet1.xml',lambda s:s.replace('<v>10</v>','<f>2*5</f><v></v>')),('bad number','xl/worksheets/sheet1.xml',lambda s:s.replace('<v>10</v>','<v>BAD</v>')),('bad ISO','xl/worksheets/sheet1.xml',lambda s:s.replace('t="n"><v>10</v>','t="d"><v>not-date</v>')),('inflated dimension','xl/worksheets/sheet1.xml',lambda s:s.replace('ref="A1:G4"','ref="A1:XFD1048576"')),('short dimension','xl/worksheets/sheet1.xml',lambda s:s.replace('ref="A1:G4"','ref="A1:A1"')),('merged link','xl/worksheets/sheet1.xml',lambda s:s.replace('</worksheet>','<hyperlinks><hyperlink ref="G4" location="Sheet!A1"/></hyperlinks></worksheet>')),('link range','xl/worksheets/sheet1.xml',lambda s:s.replace('</worksheet>','<hyperlinks><hyperlink ref="B6:D7" location="Sheet!A1"/></hyperlinks></worksheet>')),('rich blank','xl/worksheets/sheet1.xml',lambda s:s.replace('<is><t>text</t></is>','<is><r><rPr><b/></rPr><t></t></r></is>')),('plain blank','xl/worksheets/sheet1.xml',lambda s:s.replace('<is><t>text</t></is>','<is></is>')),('default font','xl/styles.xml',lambda s:s.replace('theme="1"','rgb="FF123456"'))]
    for name,path,fn in mutations:capture(name,mutate(base,path,fn))
    for value in ['-0.0','1e16','1e-05','1e20','9.999999999999999e-05','123456789012345678901234567890']:
        capture('numeric raw '+value,mutate(base,'xl/worksheets/sheet1.xml',lambda s,v=value:s.replace('<v>10</v>','<v>'+v+'</v>')))
    capture('large boolean',mutate(base,'xl/worksheets/sheet1.xml',lambda s:s.replace('t="n"><v>10</v>','t="b"><v>123456789012345678901234567890</v>')))
    wb=Workbook();wb.active['A1']='first';ws=wb.create_sheet('Other');ws['A1']='second';named=data(wb)
    for title in ['Sheet','sheet','Sheet1','', 'bad/name']:
        capture('sheet title '+repr(title),mutate(named,'xl/workbook.xml',lambda s,t=title:s.replace('name="Other"','name="'+t+'"')))
    # Shared strings preserve the same rich-text paths as inline strings.
    for content in ['<t>shared _x005F_ text</t>','<r><t>A</t></r><r><t>B</t></r>','<r><rPr><b/><u/></rPr><t> rich &amp; text </t></r>','<r><rPr/><t></t></r>','<t></t>']:
        out=BytesIO()
        with zipfile.ZipFile(BytesIO(base))as z,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED)as dest:
            for info in z.infolist():
                raw=z.read(info).decode()if info.filename in ['xl/worksheets/sheet1.xml','xl/_rels/workbook.xml.rels','[Content_Types].xml']else None
                if info.filename=='xl/worksheets/sheet1.xml':raw=raw.replace('t="n"><v>10</v>','t="s"><v>0</v>')
                if info.filename=='xl/_rels/workbook.xml.rels':raw=raw.replace('</Relationships>','<Relationship Id="shared" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/></Relationships>')
                if info.filename=='[Content_Types].xml':raw=raw.replace('</Types>','<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>')
                dest.writestr(info,raw.encode()if raw is not None else z.read(info))
            dest.writestr('xl/sharedStrings.xml','<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si>'+content+'</si></sst>')
        shared=out.getvalue();capture('shared '+content,shared)
        if content.startswith('<t>shared'):
            capture('shared manifest missing',mutate(shared,'[Content_Types].xml',lambda s:s.replace('<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>','')))
            capture('shared relation missing',mutate(shared,'xl/_rels/workbook.xml.rels',lambda s:s.replace('<Relationship Id="shared" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>','')))
    for path in sorted((ROOT/'demo/xlsx').glob('*.xlsx'))+[ROOT/'demo/ms_office_docs/xlsx_01.xlsx']:capture('real '+path.name,path.read_bytes())
    target=ROOT/'tests/fixtures/office_rust_reference_068/xlsx_cells.json.gz';target.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True,default=str).encode(),mtime=0));print('XLSX cell/package references:',len(cases),'workbooks;',sum(len(c.get('sheets',[]))for c in cases),'worksheets;',sum('error'in c for c in cases),'errors')
if __name__=='__main__':main()
