"""Freeze original chart parsing, workbook priority and HTML outputs."""
import dataclasses,gzip,json,sys,random
from io import BytesIO
from pathlib import Path
from html import escape
from datetime import datetime,time,date,timedelta
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
C='http://schemas.openxmlformats.org/drawingml/2006/chart';A='http://schemas.openxmlformats.org/drawingml/2006/main'
def cache(values,kind='numRef',formula=None):
    cache='strCache' if kind=='strRef' else 'numCache'
    return f'<c:{kind}>'+('' if formula is None else '<c:f>'+escape(formula)+'</c:f>')+f'<c:{cache}>'+''.join(f'<c:pt idx="{i}"><c:v>{escape(str(v))}</c:v></c:pt>' for i,v in enumerate(values))+f'</c:{cache}></c:{kind}>'
def series(name='A',x=None,y=None,size=None,formulas=False):
    return '<c:ser><c:tx><c:v>'+escape(name)+'</c:v></c:tx><c:cat>'+cache(x or [],'strRef',"'Data'!$A$2:$A$4" if formulas else None)+'</c:cat><c:xVal>'+cache(x or [],formula="Data!A2:A4" if formulas else None)+'</c:xVal><c:val>'+cache(y or [],formula="Data!B2:B4" if formulas else None)+'</c:val><c:yVal>'+cache(y or [],formula="Data!B2:B4" if formulas else None)+'</c:yVal><c:bubbleSize>'+cache(size or [],formula="Data!C2:C4" if formulas else None)+'</c:bubbleSize></c:ser>'
def chart(plot,body,dateaxis=False,date1904=False,other=''):
    return f'<c:chartSpace xmlns:c="{C}" xmlns:a="{A}"><c:date1904 val="{int(date1904)}"/><c:chart><c:title><c:tx><c:rich><a:p><a:r><a:t> Title &amp; text </a:t></a:r></a:p></c:rich></c:tx></c:title><c:plotArea><c:{plot}>{body}</c:{plot}>{other}<c:'+('dateAx' if dateaxis else 'catAx')+'><c:title><a:t> Axis </a:t></c:title></c:'+('dateAx' if dateaxis else 'catAx')+'><c:valAx><c:axPos val="b"/><c:title><a:t>X axis</a:t></c:title></c:valAx><c:valAx><c:axPos val="l"/><c:title><a:t>Y axis</a:t></c:title></c:valAx></c:plotArea></c:chart></c:chartSpace>'
def main():
    from rapid_doc._vendor.docvortex_core.analyzers.native.office import ooxml_chart as original
    from openpyxl import Workbook
    from openpyxl.utils.datetime import MAC_EPOCH
    from loguru import logger
    logger.remove();cases=[]
    def capture(name,text,wb=None):
        raw=text.encode() if isinstance(text,str) else text
        spec=original.parse_chart_spec_from_ooxml(raw)
        c=dict(name=name,chart_hex=raw.hex(),workbook_hex=wb.hex() if wb is not None else None,spec=dataclasses.asdict(spec) if spec else None,cache=original.render_chart_html_from_cache(spec) if spec else '')
        for key,call in [('workbook',lambda:original.render_chart_html_from_workbook(spec,wb) if spec and wb else ''),('fallback',lambda:original.html_table_from_excel_bytes(wb) if wb else ''),('html',lambda:original.extract_chart_html_from_ooxml(raw,wb))]:
            try:c[key]=call()
            except Exception as e:c[key+'_error']=type(e).__name__
        cases.append(c)
    plots=['areaChart','area3DChart','barChart','bar3DChart','bubbleChart','doughnutChart','lineChart','line3DChart','ofPieChart','pieChart','pie3DChart','radarChart','scatterChart','stockChart','surfaceChart','surface3DChart']
    for plot in plots:
        for shared in [False,True]:
            capture(plot+str(shared),chart(plot,series(' A & <B> ',['a','b','c'],['1','2',''],['4','5','6'])+series('', ['a','b','c'] if shared else ['3','4'],['-2','0.1'],['7','8'])))
    for epoch in [False,True]:
        capture('date '+str(epoch),chart('lineChart',series('dates',['0','0.5','1','59','60','61','-1','45292.25','NaN','not a date'],['1','2']),True,epoch))
    for indices in [['-1','0','2'],['1','1','3'],['bad','+2','１'],['100000'],['99999'],['-5']]:
        raw='<c:ser><c:val><c:numRef><c:numCache>'+''.join(f'<c:pt idx="{i}"><c:v>{n}</c:v></c:pt>' for n,i in enumerate(indices))+'</c:numCache></c:numRef></c:val></c:ser>';capture('indices '+str(indices),chart('barChart',raw))
    multi='<c:ser><c:cat><c:multiLvlStrRef><c:f>Data!A1:A3</c:f><c:multiLvlStrCache><c:lvl><c:pt idx="0"><c:v>Group</c:v></c:pt><c:pt idx="2"><c:v>Other</c:v></c:pt></c:lvl><c:lvl><c:pt idx="0"><c:v>A</c:v></c:pt><c:pt idx="1"><c:v>B</c:v></c:pt></c:lvl></c:multiLvlStrCache></c:multiLvlStrRef></c:cat><c:val>'+cache([1,2,3])+'</c:val></c:ser>';capture('multilevel',chart('barChart',multi))
    for other in ['<c:lineChart>'+series('second',['a'],[2])+'</c:lineChart>','<c:scatterChart>'+series('second',['a'],[2])+'</c:scatterChart>']:
        capture('combo '+other[:18],chart('barChart',series('first',['b'],[1]),other=other))
    for raw in [b'',b'bad',b'<chart/>',chart('lineChart','').encode()]:capture('empty '+str(len(raw)),raw)
    for epoch in [False,True]:
        wb=Workbook();empty=wb.active;empty.title='Empty';ws=wb.create_sheet('Data');
        if epoch:wb.epoch=MAC_EPOCH
        ws.append(['category','value','size']);ws.append(['first',10,3]);ws.append(['second',20,4]);ws.append(['third',30,5]);
        out=BytesIO();wb.save(out);data=out.getvalue()
        for plot in ['barChart','scatterChart','bubbleChart']:
            capture('workbook priority '+plot+str(epoch),chart(plot,series('cached',[100,200,300],[9,9,9],[7,7,7],True)),data)
        capture('workbook generic '+str(epoch),b'<chart/>',data)
        capture('bad reference cache '+str(epoch),chart('barChart',series('cached',[100],[9]).replace('<c:numRef>','<c:numRef><c:f>missing!A1:A2</c:f>')),data)
        capture('bad workbook cache '+str(epoch),chart('barChart',series('cached',[100],[9])),b'not workbook')
    wb=Workbook();ws=wb.active;ws.title="O'Brien";ws.append([' name & <x> ',None,'flag']);ws.append([datetime(2024,1,2),0,True]);ws.append([date(2024,1,3),1.23456,False]);ws.append([time(12,34,56),None,'=2+3']);ws.append([None,None,None]);ws.append(['last',-2,1e-5]);out=BytesIO();wb.save(out);data=out.getvalue()
    for f in ["'O''Brien'!A2:A3","='O''Brien'!$B$2:$B$6","'O''Brien'!A1:C1","'O''Brien'!A1:C3","'O''Brien'!A:A","'O''Brien'!C4:C4"]:
        capture('typed cells '+f,chart('lineChart',series('',[],[],formulas=True).replace("'Data'!$A$2:$A$4",f).replace('Data!B2:B4',f)),data)
    capture('typed workbook fallback',b'<chart/>',data)
    # Cell types and number formats are read by openpyxl before chart rendering.
    # Include duration formats, ISO dates and caches independently of chart XML.
    for iso in [False,True]:
        wb=Workbook();wb.iso_dates=iso;ws=wb.active;ws.title='Data'
        ws.append(['category','value','size'])
        values=[datetime(2024,1,2,3,4,5,123000),date(2024,1,3),time(12,34,56,789000),timedelta(days=2,seconds=1234,microseconds=567000),timedelta(days=-1,seconds=1),0.5,60,1.2345e-7]
        for i,v in enumerate(values,2):ws.append([v,i,True])
        out=BytesIO();wb.save(out);data=out.getvalue()
        capture('ISO typed fallback '+str(iso),b'<chart/>',data)
        for hint in [False,True]:
            capture('ISO typed category '+str((iso,hint)),chart('lineChart',series('',[],[],formulas=True).replace("'Data'!$A$2:$A$4",'Data!A2:A9').replace('Data!B2:B4','Data!B2:B9'),hint),data)
    for fmt in ['[h]:mm:ss','[mm]:ss.000','[s].000','[HH]:mm','yyyy-mm-dd','0.00 "days"','0.0\\m','0 _m','[$-409]d-mmm-yy','mm:ss.0','0 "days','[days','0 "m\nm"']:
        wb=Workbook();ws=wb.active;ws.title='Data';ws.append(['category','value'])
        for i,v in enumerate([0.,.5,1.23456789,-.25,45292.,3000000.],2):ws.append([v,i]);ws.cell(i,1).number_format=fmt
        out=BytesIO();wb.save(out);data=out.getvalue()
        capture('number format '+fmt,b'<chart/>',data)
    # Mutate source XML to test cached formulas, advertised dimensions and errors.
    from zipfile import ZipFile,ZIP_DEFLATED
    wb=Workbook();ws=wb.active;ws.title='Data';ws.append(['category','value']);ws.append(['a',10]);ws.append(['b',20]);out=BytesIO();wb.save(out);base=out.getvalue()
    for label,old,new in [('dimension','ref="A1:B3"','ref="A1:A1"'),('cached formula','<v>10</v>','<f>5+5</f><v>10</v>'),('empty formula','<v>10</v>','<f>5+5</f><v></v>'),('invalid ISO','t="n"><v>10</v>','t="d"><v>not-date</v>'),('ISO duration','t="n"><v>10</v>','t="d"><v>PT26H1M2.123S</v>')]:
        out=BytesIO()
        with ZipFile(BytesIO(base)) as source,ZipFile(out,'w',ZIP_DEFLATED) as dest:
            for member in source.infolist():
                raw=source.read(member)
                if member.filename=='xl/worksheets/sheet1.xml':raw=raw.decode().replace(old,new).encode()
                dest.writestr(member,raw)
        capture('worksheet '+label,chart('barChart',series('cache',[],[],formulas=True).replace('Data!B2:B4','Data!B2:B3')),out.getvalue())
    # Independent existing Office charts and their own relationship workbooks.
    from docx import Document
    for path in ['demo/docx/demo1-docx.docx','demo/docx/test.docx','demo/docx/drawingml.docx']:
        doc=Document(ROOT/path)
        for rel in doc.part.rels.values():
            if rel.reltype.endswith('/chart'):
                part=rel.target_part;embedded=next((r.target_part.blob for r in part.rels.values() if r.reltype.endswith('/package') and not r.is_external),None);capture('real '+path+' '+str(part.partname),part.blob,embedded)
    serials=[]
    rng=random.Random(310313)
    for v in [0.,0.000000001,0.5,0.9999999999,1.,59.,60.,61.,-0.5,-100.,45292.125,1e8,float('inf'),float('nan')]+[rng.uniform(-693000,2000000) for _ in range(70)]+[(i+.5)/86400000 for i in range(20)]:
        for epoch in [False,True]:serials.append(dict(value=repr(v),epoch=epoch,result=original._excel_serial_to_iso(v,date_1904=epoch)))
    target=ROOT/'tests/fixtures/office_rust_reference_068/office_charts.json.gz';target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,serials=serials),ensure_ascii=True).encode(),mtime=0));print('Chart references:',len(cases),'queries;',len(serials),'date conversions')
if __name__=='__main__':main()
