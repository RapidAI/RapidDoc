"""Freeze complete original DOCX outputs for the implemented body routes."""
import gzip,json,sys,random
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
def main():
    from docx import Document
    from docx.oxml import parse_xml
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.section import WD_SECTION_START
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from loguru import logger
    logger.remove();cases=[]
    def node(xml):return parse_xml(('<root xmlns:w="'+W+'" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'+xml+'</root>').encode())[0]
    def add(doc,xml):doc.element.body.insert(len(doc.element.body)-1,node(xml))
    def capture(name,doc):
        if isinstance(doc,bytes):raw=doc
        else:
            data=BytesIO();doc.save(data);raw=data.getvalue()
        c=dict(name=name,package_hex=raw.hex())
        engine=DocxConverter()
        try:
            engine.convert(BytesIO(raw));c['pages']=engine.pages
            def starts(v):
                if isinstance(v,dict):
                    return ([str(v['start'])] if 'start' in v else [])+sum((starts(x) for x in v.values()),[])
                if isinstance(v,list):return sum((starts(x) for x in v),[])
                return []
            c['starts']=starts(engine.pages)
            from export_office_raster_reference import summary
            from export_office_path_pixels import normalized_svg
            import base64
            def normalize(v):
                if isinstance(v,dict):
                    for key,value in list(v.items()):
                        if key=='image_base64' and isinstance(value,str):
                            v[key]=dict(svg=normalized_svg(base64.b64decode(value.split(',')[1]))) if value.startswith('data:image/svg+xml;') else summary(value)
                        else:normalize(value)
                elif isinstance(v,list):
                    for x in v:normalize(x)
            normalize(c['pages'])
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    def new():
        d=Document()
        for n in range(5):
            name='TOC '+str(n)
            if name not in d.styles:d.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        d.styles.add_style('CaptionCustom',WD_STYLE_TYPE.PARAGRAPH)
        numbering=d.part.numbering_part.element
        numbering.append(node('<w:abstractNum w:abstractNumId="99">'+''.join('<w:lvl w:ilvl="'+str(level)+'"><w:start w:val="'+str(level+2)+'"/><w:numFmt w:val="'+('bullet' if level==2 else 'decimal')+'"/><w:lvlText w:val="%1."/></w:lvl>' for level in range(5))+'</w:abstractNum>'))
        for num in [99,100]:numbering.append(node('<w:num w:numId="'+str(num)+'"><w:abstractNumId w:val="99"/>'+('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="7"/></w:lvlOverride>' if num==100 else '')+'</w:num>'))
        return d
    def item(doc,text,level=0,num=99):add(doc,f'<w:p><w:pPr><w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="{num}"/></w:numPr></w:pPr><w:r><w:t>{text}</w:t></w:r></w:p>')
    def toc(doc,text,level=1,anchor=None):
        visible='<w:r><w:t>'+text+'</w:t></w:r>'
        if anchor:visible='<w:hyperlink w:anchor="'+anchor+'">'+visible+'</w:hyperlink>'
        return '<w:p><w:pPr><w:pStyle w:val="TOC'+str(level)+'"/></w:pPr>'+visible+'</w:p>'
    d=new();d.add_heading(' Title ',0);d.add_heading(' chapter ',1);d.add_paragraph(' plain 中文 😀 ');d.add_paragraph('');capture('titles and normal',d)
    for levels in [[0,0,1,1,2,0],[3,2,1,0],[0,3,1,0],[2,2,4,2],[0,1,0,1],[0,-1,0],[-1,-1,0]]:
        d=new()
        for i,level in enumerate(levels):item(d,'item '+str(i),level)
        capture('nest '+str(levels),d)
    d=new();item(d,'one');d.add_paragraph('boundary');item(d,'resume');d.add_table(1,1).cell(0,0).text='table boundary';item(d,'resume after table');item(d,'new num',num=100);item(d,'old num again');capture('boundaries and counters',d)
    d=new();item(d,'parent',0);item(d,'child',1);d.add_paragraph('body');item(d,'next parent',0);item(d,'next child',1);capture('heading style list detection',d)
    d=new();item(d,'a',0);item(d,'b',1);item(d,'c',0);item(d,'d',1);item(d,'e',2);capture('parent restart bullet child',d)
    d=new();d.add_paragraph('a','List Number');d.add_paragraph('b','List Number');d.add_paragraph('c','List Bullet');capture('builtin numbering',d)
    d=new();item(d,'a');d.add_heading('heading boundary',1);item(d,'b');capture('heading does not close list',d)
    for level in [0,1,2]:
        d=new();add(d,toc(d,'cover reused style',level));d.add_paragraph('real body');add(d,toc(d,'1. Root',level,'_TocA'));add(d,toc(d,'1.1 Child',level+2,'_TocB'));add(d,toc(d,'unlinked continuation',level+1));d.add_paragraph('end');capture('plain toc '+str(level),d)
    for levels in [[1,2,3,1],[3,2,1],[1,4,2,1],[0,1,0]]:
        d=new();xml=''.join(toc(d,'1.1 item '+str(i),level,'_Toc'+str(i)) for i,level in enumerate(levels));add(d,'<w:sdt><w:sdtPr><w:tag w:val="TableOfContents"/></w:sdtPr><w:sdtContent>'+xml+'</w:sdtContent></w:sdt>');d.add_paragraph('after TOC');capture('sdt toc '+str(levels),d)
    d=new();add(d,'<w:sdt><w:sdtPr><w:docPartObj><w:docPartGallery w:val="Table of Contents"/></w:docPartObj></w:sdtPr><w:sdtContent>'+toc(d,'Figure 1. A',3)+toc(d,'表 2. B',4)+toc(d,'heading',1)+'</w:sdtContent></w:sdt>');capture('flat figure list',d)
    d=new();add(d,toc(d,'target',1,'_TocB'));add(d,'<w:p><w:bookmarkStart w:id="0" w:name="ordinary"/><w:bookmarkStart w:id="1" w:name="_TocA"/><w:bookmarkStart w:id="2" w:name="_TocB"/><w:r><w:t>heading target</w:t></w:r></w:p>');capture('bookmark aliases',d)
    d=new();add(d,'<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr><w:r><w:fldChar w:fldCharType="begin"/><w:instrText> HYPERLINK \\l "_TocX" </w:instrText><w:fldChar w:fldCharType="separate"/><w:t>complex TOC</w:t><w:fldChar w:fldCharType="end"/></w:r></w:p>');capture('field toc anchor',d)
    d=new();add(d,'<w:p><w:r><w:rPr><w:b/><w:u/></w:rPr><w:t xml:space="preserve"> bold </w:t></w:r><w:hyperlink w:anchor="local"><w:r><w:rPr><w:i/></w:rPr><w:t>linked</w:t></w:r></w:hyperlink><w:r><w:t xml:space="preserve"> tail </w:t></w:r></w:p>');capture('rich linked text',d)
    for display in [False,True]:
        d=new();add(d,'<w:p>'+('' if display else '<w:r><w:t>A😀 </w:t></w:r>')+'<m:oMath><m:f><m:num><m:r><m:t>x</m:t></m:r></m:num><m:den><m:r><m:t>2</m:t></m:r></m:den></m:f></m:oMath>'+('' if display else '<w:r><w:rPr><w:b/></w:rPr><w:t> B</w:t></w:r>')+'</w:p>');capture('math '+str(display),d)
    d=new();add(d,'<w:p><w:pPr><w:pStyle w:val="CaptionCustom"/></w:pPr><w:r><w:instrText>SEQ Figure</w:instrText><w:t>Figure 1</w:t></w:r></w:p>');capture('caption field',d)
    for start in [WD_SECTION_START.NEW_PAGE,WD_SECTION_START.CONTINUOUS]:
        d=new();d.add_paragraph('before');d.add_section(start);d.add_paragraph('after');capture('section '+str(start),d)
    d=new();d.add_paragraph('before');add(d,'<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/><w:pgMar w:top="0" w:bottom="0" w:left="0" w:right="0" w:header="0" w:footer="0"/></w:sectPr></w:pPr></w:p>');d.add_paragraph('after');capture('layout only section',d)
    d=new();add(d,'<w:sdt><w:sdtContent><w:p><w:r><w:t>content control</w:t></w:r></w:p><w:p><w:r><w:t>next</w:t></w:r></w:p></w:sdtContent></w:sdt>');capture('plain block sdt',d)
    for digits in ['123','１２３','²','①','½',' 123 ','']:
        d=new();d.add_paragraph('body');d.sections[0].header.paragraphs[0].text=digits;d.sections[0].footer.paragraphs[0].text=digits;capture('margin digits '+digits,d)
    d=new();d.add_paragraph('body');sec=d.sections[0];sec.header.paragraphs[0].text='first line';sec.header.add_paragraph('second line');sec.footer.paragraphs[0].text='footer';capture('margin paragraphs',d)
    for odd_even in [False,True]:
        d=new();d.settings.odd_and_even_pages_header_footer=odd_even;sec=d.sections[0];sec.different_first_page_header_footer=True
        sec.header.paragraphs[0].text='default';sec.even_page_header.paragraphs[0].text='even';sec.first_page_header.paragraphs[0].text='first'
        sec.footer.paragraphs[0].text='footer';sec.even_page_footer.paragraphs[0].text='footer';sec.first_page_footer.paragraphs[0].text='first footer'
        d.add_paragraph('section one');d.add_section(WD_SECTION_START.NEW_PAGE);d.add_paragraph('section two');capture('margin variants inheritance '+str(odd_even),d)
    d=new();d.sections[0].header.paragraphs[0]._p.append(node('<w:hyperlink w:anchor="local"><w:r><w:rPr><w:b/></w:rPr><w:t>link</w:t></w:r></w:hyperlink>'));d.sections[0].header.paragraphs[0]._p.append(node('<m:oMath><m:r><m:t>x</m:t></m:r></m:oMath>'));d.add_paragraph('body');capture('margin linked math',d)
    d=new();d.add_paragraph('one');d.sections[0].header.paragraphs[0].text='header one';sec=d.add_section(WD_SECTION_START.NEW_PAGE);sec.header.is_linked_to_previous=False;sec.header.paragraphs[0].text='header two';d.add_paragraph('two');capture('margin override',d)
    d=new();d.sections[0].header.paragraphs[0].text='header';add(d,'<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/><w:pgMar w:top="0" w:bottom="0" w:left="0" w:right="0" w:header="0" w:footer="0"/></w:sectPr></w:pPr></w:p>');d.add_paragraph('body');capture('margin section without logical page',d)
    d=new();table=d.add_table(1,1);table.cell(0,0)._tc.append(node('<w:p><w:r><w:t>formula </w:t></w:r><m:oMath><m:r><m:t>x</m:t></m:r></m:oMath></w:p>'));capture('table equation body',d)
    d=new();d.part.numbering_part.element.find('{'+W+'}abstractNum[@{'+W+'}abstractNumId="99"]').find('{'+W+'}lvl').find('{'+W+'}start').set('{'+W+'}val','9'*80);item(d,'large start');item(d,'large next');capture('arbitrary precision list start',d)
    def textbox(doc,groups,modern=False):
        shapes=[]
        for paragraphs in groups:
            content=''.join('<w:p><w:r><w:t>'+s+'</w:t></w:r></w:p>' for s in paragraphs)
            if modern:shapes.append('<wps:wsp><wps:txbx><w:txbxContent>'+content+'</w:txbxContent></wps:txbx></wps:wsp>')
            else:shapes.append('<v:shape><v:textbox><w:txbxContent>'+content+'</w:txbxContent></v:textbox></v:shape>')
        add(doc,'<w:p xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><w:r><w:pict>'+''.join(shapes)+'</w:pict></w:r><w:r><w:t>outside textbox</w:t></w:r></w:p>')
    for modern in [False,True]:
        for groups in [[['one','two']],[['same'],['same']],[['same','other'],['same','other']],[['different'],['other']],[['','blank after']]]:
            d=new();d.add_paragraph('before');textbox(d,groups,modern);d.add_paragraph('after');capture('textbox '+str(modern)+str(groups),d)
    d=new();add(d,'<w:pict xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:txBody><a:r><a:t>shape text</a:t></a:r></a:txBody></w:pict>');capture('top-level shape text fallback',d)
    d=new();add(d,'<w:pict xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:txBody><a:p><a:r><a:t>original collected a:t without w:p</a:t></a:r></a:p></a:txBody></w:pict>');capture('shape paragraph fallback semantics',d)
    d=new();add(d,'<w:drawing xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><wps:txbx><w:p><w:r><w:t>alternate text container</w:t></w:r></w:p></wps:txbx></w:drawing>');capture('alternative textbox container',d)
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    raster=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/office_raster.json.gz').read_bytes()))
    samples=[c for c in raster['cases'] if c['name'].startswith('(3, 2)') and c['expected'] and 'error' not in c['expected']]
    vectors=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/emfplus.json.gz').read_bytes()))
    samples += [dict(name='vector '+c['name'],hex=c['hex']) for c in vectors['cases'] if c['name'] in ['clear','empty','path 0','raw image 26200a 12','GetDC bridge']]
    samples += [dict(name='invalid vector',hex=b'bad vector'.hex())]
    svg=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/office_svg.json.gz').read_bytes()))
    samples += [dict(name='SVG '+c['name'],hex=c['hex']) for c in svg['cases'][:3]+svg['cases'][-14:-8]]
    for sample in samples:
        for anchored in [False,True]:
            d=new();data=bytes.fromhex(sample['hex']);vector=sample['name'].startswith(('vector ','invalid vector'));mime='image/emf' if vector else 'image/png';extension='emf' if vector else 'png'
            part=Part(PackURI('/word/media/picture.'+extension),mime,data,d.part.package);rid=d.part.relate_to(part,RT.IMAGE)
            container='anchor' if anchored else 'inline'
            drawing='<w:r xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:drawing><wp:'+container+'><wp:docPr id="1" name="picture" descr=" label &amp; text "/><a:graphic><a:graphicData><a:blip r:embed="'+rid+'"/></a:graphicData></a:graphic></wp:'+container+'></w:drawing></w:r>'
            add(d,'<w:p><w:r><w:t>paragraph text</w:t></w:r>'+drawing+drawing+'</w:p>');d.add_paragraph('next');capture('body image '+str(anchored)+' '+sample['name'],d)
    d=new();part=Part(PackURI('/word/media/picture.png'),'image/png',bytes.fromhex(samples[0]['hex']),d.part.package);rid=d.part.relate_to(part,RT.IMAGE)
    for _ in range(2):add(d,'<w:p xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:v="urn:schemas-microsoft-com:vml"><w:r><w:pict><v:shape alt=" VML alt "><v:imagedata r:id="'+rid+'"/></v:shape></w:pict></w:r></w:p>')
    capture('VML repeated part distinct paragraphs',d)
    from export_office_chart_reference import chart as chart_xml, series as chart_series
    from openpyxl import Workbook
    for plot in ['barChart','scatterChart','bubbleChart']:
        for embedded in [False,True]:
            d=new();raw=chart_xml(plot,chart_series('cached',['a','b'],[9,8],[7,6],formulas=embedded)).encode()
            part=Part(PackURI('/word/charts/chart1.xml'),'application/vnd.openxmlformats-officedocument.drawingml.chart+xml',raw,d.part.package)
            if embedded:
                wb=Workbook();ws=wb.active;ws.title='Data';ws.append(['name','value','size']);ws.append(['first',1,4]);ws.append(['second',2,5]);ws.append(['third',3,6]);buf=BytesIO();wb.save(buf)
                workbook=Part(PackURI('/word/embeddings/data.xlsx'),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',buf.getvalue(),d.part.package);part.relate_to(workbook,RT.PACKAGE)
            rid=d.part.relate_to(part,RT.CHART)
            drawing='<w:r xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:drawing><c:chart r:id="'+rid+'"/></w:drawing></w:r>'
            add(d,'<w:p>'+drawing+drawing+'<w:r><w:t>after chart</w:t></w:r></w:p>');capture('chart body '+plot+str(embedded),d)
    for broken in ['id','relation','xml','workbook']:
        d=new();raw=b'broken chart' if broken=='xml' else chart_xml('barChart',chart_series('A',['x'],[7],formulas=True)).encode();part=Part(PackURI('/word/charts/chart1.xml'),'application/vnd.openxmlformats-officedocument.drawingml.chart+xml',raw,d.part.package)
        if broken=='workbook':
            wb=Part(PackURI('/word/embeddings/data.xlsx'),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',b'broken workbook',d.part.package);part.relate_to(wb,RT.PACKAGE)
        rid=d.part.relate_to(part,RT.IMAGE if broken=='relation' else RT.CHART)
        add(d,'<w:p xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:r><w:drawing><c:chart'+('' if broken=='id' else ' r:id="'+rid+'"')+'/></w:drawing></w:r><w:r><w:t>retained text</w:t></w:r></w:p>');capture('chart broken '+broken,d)
    chart_cases=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/office_charts.json.gz').read_bytes()))['cases']
    for c in chart_cases:
        if c['name'] not in ['worksheet invalid ISO','worksheet ISO duration','worksheet cached formula','worksheet dimension','ISO typed category (True, True)']:continue
        d=new();part=Part(PackURI('/word/charts/chart1.xml'),'application/vnd.openxmlformats-officedocument.drawingml.chart+xml',bytes.fromhex(c['chart_hex']),d.part.package)
        wb=Part(PackURI('/word/embeddings/data.xlsx'),'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',bytes.fromhex(c['workbook_hex']),d.part.package);part.relate_to(wb,RT.PACKAGE)
        rid=d.part.relate_to(part,RT.CHART)
        add(d,'<w:p xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:r><w:drawing><c:chart r:id="'+rid+'"/></w:drawing></w:r><w:r><w:t>retained after workbook</w:t></w:r></w:p>');capture('chart cells '+c['name'],d)
    rng=random.Random(31031)
    for trial in range(30):
        d=new()
        for j in range(18):
            choice=rng.randrange(6)
            if choice<3:item(d,'item '+str(j),rng.choice([0,1,2,3]),rng.choice([99,100]))
            elif choice==3:d.add_paragraph('body '+str(j))
            elif choice==4:d.add_heading('heading '+str(j),rng.randrange(1,4))
            else:d.add_table(1,1).cell(0,0).text='table '+str(j)
        capture('mixed '+str(trial),d)
    for path in ['demo/docx/equations.docx','demo/docx/section.docx','demo/docx/空白.docx','demo/docx/textbox.docx','demo/docx/demo1-docx.docx','demo/docx/test.docx','demo/docx/drawingml.docx']:
        capture('real '+path,(ROOT/path).read_bytes())
    for name in ['ole.docx','ole_v5.docx','gif_v3.docx','gif_v5.docx']:
        capture('embedded '+name,(ROOT/'tests/fixtures/office_rust_reference_068/mtef_packages'/name).read_bytes())
    path=ROOT/'tests/fixtures/office_rust_reference_068/docx_body.json.gz';path.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0));print('DOCX body references:',len(cases),'complete original outputs;',sum('error' in c for c in cases),'errors')
if __name__=='__main__':main()
