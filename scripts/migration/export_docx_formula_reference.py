"""Current Python token arbitration, Word cell rebuild and injected table HTML."""
from pathlib import Path
from xml.sax.saxutils import escape,quoteattr
import gzip
import json
import sys
from io import BytesIO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M='http://schemas.openxmlformats.org/officeDocument/2006/math'
NS=f'xmlns:w="{W}" xmlns:m="{M}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'


def main():
    from lxml import etree
    from docx import Document
    from docx.oxml import parse_xml
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.equationxml import DocxEquationXmlDecoder
    from rapid_doc._vendor.docvortex_core.foundation._hyperlink import sanitize_hyperlink_target, OFFICE_EXTERNAL_HYPERLINK_SCHEMES
    from bs4 import BeautifulSoup
    from loguru import logger
    from scripts.migration.export_docx_table_html_reference import capture as mammoth
    logger.remove()
    document=Document()
    document.part.rels.add_relationship(RT.HYPERLINK,'https://example.org/a?x=1&y=2',rId='link',is_external=True)
    document.styles['Normal'].font.bold=True
    converter=DocxConverter()
    converter.docx_obj=document
    styles=etree.tostring(document.styles.element,encoding='unicode')
    relationships=[dict(id='link',kind=RT.HYPERLINK,target='https://example.org/a?x=1&y=2',external=True)]
    def r(text,prop=''):return f'<w:r><w:rPr>{prop}</w:rPr><w:t>{escape(text)}</w:t></w:r>'
    def math(text='x+y'):return '<m:oMath><m:r><m:t>'+escape(text)+'</m:t></m:r></m:oMath>'
    def p(body,prop=''):return f'<w:p><w:pPr>{prop}</w:pPr>{body}</w:p>'
    def table(body):return '<w:tbl><w:tr><w:tc>'+body+'</w:tc></w:tr></w:tbl>'
    eqdoc=f'<w:wordDocument xmlns:w="http://schemas.microsoft.com/office/word/2003/wordml" xmlns:m="{M}"><w:body><w:p><m:oMathPara>{math("a/b")}</m:oMathPara></w:p></w:body></w:wordDocument>'
    shape=f'<v:shape equationxml={quoteattr(eqdoc)}/>'
    paragraphs=[p(r('A')+math()+r('B')),
        p(r(' under\nline ','<w:u w:val="single"/><w:i/><w:strike/>')+math('a<b & c')),
        p(r('top','<w:vertAlign w:val="superscript"/>')+math('𝔸+ℝ')),
        p('<w:hyperlink r:id="link" w:anchor="frag">'+r('linked')+math()+'</w:hyperlink>'),
        p('<w:hyperlink w:anchor="local">'+r('anchor')+math()+'</w:hyperlink>'),
        p(r('hidden','<w:vanish/>')+math()),
        p('<w:fldSimple>'+r('simple')+math()+'</w:fldSimple>'),
        p('<w:sdt><w:sdtContent>'+r('control')+math()+'</w:sdtContent></w:sdt>'),
        p('<w:customXml>'+r('custom')+math()+'</w:customXml>'),
        p('<w:r><w:t>A</w:t><w:tab/><w:ptab/><w:cr/><w:br/><w:noBreakHyphen/><w:softHyphen/><w:t>B</w:t></w:r>'+math()),
        p('<mc:AlternateContent><mc:Choice>'+r('choice')+math('choice')+'</mc:Choice><mc:Fallback>'+r('fallback')+math('fallback')+'</mc:Fallback></mc:AlternateContent>'),
        p('<mc:AlternateContent><mc:Choice>'+r('text choice')+'</mc:Choice><mc:Fallback>'+math('fallback')+'</mc:Fallback></mc:AlternateContent>'),
        p('<mc:AlternateContent><mc:Choice>'+shape+'</mc:Choice><mc:Fallback>'+math('preferred')+'</mc:Fallback></mc:AlternateContent>'),
        p('<w:object>'+shape+math('preferred')+'</w:object>'),
        p(shape),p('<m:oMath><m:acc/></m:oMath>'),p('<w:txbxContent>'+p(math('outside'))+'</w:txbxContent>'+math('inside')),
        p('<m:oMathPara>'+math('first')+math('second')+'</m:oMathPara>'),
    ]
    cases=[]
    numbering=None
    def capture(body,name):
        xml=f'<w:document {NS}><w:body>{body}</w:body></w:document>'
        node=parse_xml(xml)[0][0]
        case=dict(name=name,document=xml,styles=styles,numbering=numbering,relationships=relationships,tokens=converter._docx_formula_tokens(node,document.part))
        def render():
            if etree.QName(node).localname=='p':return converter._build_paragraph_html_with_equations(node,document.part)
            raw=mammoth(etree.tostring(node,encoding='unicode'),name,styles,numbering,relationships=relationships)
            assert 'html' in raw,raw
            soup=BeautifulSoup(raw['html'],'html.parser')
            injected=str(converter._inject_equations_into_table(soup.table,node,document.part))
            return converter._normalize_table_links(converter._normalize_table_colspans(injected))
        key='paragraph' if etree.QName(node).localname=='p' else 'table'
        case[key]=render()
        # Keep the original snapshot. A second reference isolates one deliberate
        # HTML-boundary correction: escaping equation text before raw insertion.
        # The original Python serializer otherwise mistakes e.g. "a<b" for a tag.
        original=converter._docx_formula_tokens
        def escaped(node,part):
            converter._docx_formula_tokens=original
            try:
                result=original(node,part)
                return [(kind,escape(value)) if kind!='text' else (kind,value) for kind,value in result]
            finally:converter._docx_formula_tokens=escaped
        try:
            converter._docx_formula_tokens=escaped
            case['safe_'+key]=render()
        finally:converter._docx_formula_tokens=original
        cases.append(case)
    for index,paragraph in enumerate(paragraphs):capture(paragraph,f'paragraph {index}')
    # Ordinary cell contents and malformed row/cell count guards use the actual
    # Mammoth output before injection, including nested tables and merged cells.
    for index,paragraph in enumerate(paragraphs):capture(table(paragraph),f'table {index}')
    capture('<w:tbl><w:tr><w:tc>'+p(r('outer'))+table(p(math()))+'</w:tc></w:tr></w:tbl>','nested guard')
    capture('<w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="2"/></w:tcPr>'+p(math())+'</w:tc></w:tr><w:tr><w:tc>'+p(r('a'))+'</w:tc><w:tc>'+p(r('b'))+'</w:tc></w:tr></w:tbl>','colspan after injection')
    for target in ['javascript:alert(1)',' HTTPS://example.org ','//evil','../relative#x','#local','mailto:user@example.org']:
        document.part.rels['link']._target=target
        # Emit a separate immutable relationship reference for each target.
        relationships=[dict(id='link',kind=RT.HYPERLINK,target=target,external=True)]
        capture(table(p('<w:hyperlink r:id="link">'+r('visible','<w:i/>')+math()+'</w:hyperlink>')),f'link {target}')
    manifest=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/manifest.json').read_text())
    real_tables=0
    for source in manifest['cases']:
        if source['format']!='docx':continue
        converter=DocxConverter()
        payload=converter._sanitize_missing_internal_relationships((ROOT/source['input']).read_bytes())
        document=Document(BytesIO(payload))
        converter.docx_obj=document
        styles=etree.tostring(document.styles.element,encoding='unicode')
        numbering_root=converter._get_numbering_root()
        numbering=etree.tostring(numbering_root,encoding='unicode') if numbering_root is not None else None
        relationships=[dict(id=r.rId,kind=r.reltype,target=r.target_ref,external=r.is_external) for r in document.part.rels.values()]
        for index,node in enumerate(document.element.iter(f'{{{W}}}tbl')):
            names={etree.QName(n).localname for n in node.iter() if isinstance(n.tag,str)}
            if 'oMath' in names and not names.intersection({'blip','imagedata','OLEObject','shape'}):
                capture(etree.tostring(node,encoding='unicode'),source['id']+f' real table {index}')
                real_tables+=1
    equationxml=[]
    for name,payload in [('valid',eqdoc),('empty',''),('spaces','  '),('malformed','<x>'),('wrong root',eqdoc.replace('wordDocument','other')),
                         ('comment',eqdoc.replace('<w:body>','<!-- forbidden --><w:body>')),
                         ('two paragraphs',eqdoc.replace('</w:body>','<w:p/></w:body>')),
                         ('extra equation',eqdoc.replace('</w:body>',math()+'</w:body>')),
                         ('doctype','<!DOCTYPE w:wordDocument>'+eqdoc),
                         ('utf8 declaration','<?xml version="1.0" encoding="UTF-8"?>'+eqdoc),
                         ('contradicting latin1 declaration','<?xml version="1.0" encoding="ISO-8859-1"?>'+eqdoc.replace('a/b','é')),
                         ('contradicting utf16 declaration','<?xml version="1.0" encoding="UTF-16"?>'+eqdoc),
                         ('malformed OMML',eqdoc.replace(math('a/b'),'<m:oMath><m:acc/></m:oMath>'))]:
        decoder=DocxEquationXmlDecoder()
        value=decoder.decode(payload)
        assert decoder.decode(payload)==value
        equationxml.append(dict(name=name,input=payload,expected=value,total_bytes=decoder.total_bytes))
    links=[]
    targets=['https://example.org','http://','HTTP:relative','ftp://user@host:bad/path','mailto:x','mailto:?q','tel:12','javascript:hi','data:text/plain,x','1:relative','x:y', '../a','/root','//host','#','#  ','#ok','.',None,'a\\b','a\nb','http://[::1]:abc','http://[127.0.0.1]','http://[v1.name]','http://[V1.name]','http://[fe80::1%zone]','http://[fe80::1%]','http://[::1]suffix','http://user[::1]@host','http://user@:80']
    targets += ['https://exa'+c+'mple.org' for c in json.loads((ROOT/'rust/rapiddoc-office/src/link_nfkc_forbidden.json').read_text())]
    for target in targets:
        links.append(dict(input=target,expected=sanitize_hyperlink_target(target,allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,allow_relative=True,allow_fragment=True)))
    fixture=dict(cases=cases,equationxml=equationxml,links=links,real_tables=real_tables)
    target=ROOT/'tests/fixtures/office_rust_reference_068/docx_formulas.json.gz'
    target.write_bytes(gzip.compress(json.dumps(fixture,ensure_ascii=False).encode(),mtime=0))
    print('DOCX formula references:',len(cases),'token/render cases;',real_tables,'real tables;',len(equationxml),'Equation XML;',len(links),'links')


if __name__=='__main__':main()
