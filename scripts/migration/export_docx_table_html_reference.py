"""Freeze the original Mammoth reader + default HTML conversion, no replacement algorithm."""
from io import BytesIO
from pathlib import Path
import gzip
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = f'xmlns:w="{W}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'


def capture(xml, name, styles=None, numbering=None, relationships=(), source=None):
    from lxml import etree
    from mammoth.docx import body_xml, numbering_xml, styles_xml, office_xml, relationships_xml
    from mammoth.conversion import convert_document_element_to_html
    from mammoth.options import read_options
    from contextlib import ExitStack
    from mammoth.zips import open_zip
    from mammoth.docx import content_types_xml
    def tree(value): return office_xml.read(BytesIO(value.encode()))
    style = styles_xml.read_styles_xml_element(tree(styles)) if styles else styles_xml.Styles.EMPTY
    numbers = numbering_xml.read_numbering_xml_element(tree(numbering), style) if numbering else numbering_xml.Numbering.EMPTY
    rels = relationships_xml.Relationships([relationships_xml.Relationship(r['id'], r['target'], r['kind']) for r in relationships])
    case = dict(name=name, xml=xml, styles=styles, numbering=numbering, relationships=list(relationships),source=source)
    case['pending_resources'] = sorted({etree.QName(node).localname for node in etree.fromstring(xml.encode()).iter()
        if isinstance(node.tag,str) and (etree.QName(node).namespace or '').endswith('/math')
        and etree.QName(node).localname in {'oMath','oMathPara'}})
    try:
        with ExitStack() as stack:
            package = stack.enter_context(open_zip(BytesIO((ROOT/source).read_bytes()),'r')) if source else None
            types = content_types_xml.read_content_types_xml_element(tree(package.read_str('[Content_Types].xml'))) if package else None
            result = body_xml.reader(styles=style, numbering=numbers, relationships=rels,docx_file=package,content_types=types).read_all([tree(xml)])
            options = read_options({}).value
            case['html'] = ''.join(convert_document_element_to_html(item, **options).value for item in result.value)
    except (ValueError, IndexError, KeyError, AttributeError, TypeError) as error:
        case['error'] = type(error).__name__
    return case


def main():
    from lxml import etree
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.tables import _DocxTables
    fixture = ROOT / 'tests/fixtures/office_rust_reference_068'
    tables = json.loads(gzip.decompress((fixture/'docx_tables.json.gz').read_bytes()))
    cases = [capture(t['xml'], t['name']) for t in tables['tables']]
    contexts = json.loads(gzip.decompress((fixture/'docx_context.json.gz').read_bytes()))
    for context in contexts['cases'][:4]:
        document = etree.fromstring(context['document'].encode())
        for index, table in enumerate(document.iter(f'{{{W}}}tbl')):
            cases.append(capture(etree.tostring(table, encoding='unicode'), context['name']+f' context table {index}',
                                 context['styles'], context['numbering'], context['relationships'],context['source']))
    def run(text, prop=''): return f'<w:r><w:rPr>{prop}</w:rPr><w:t>{text}</w:t></w:r>'
    def paragraph(content, prop=''): return f'<w:p><w:pPr>{prop}</w:pPr>{content}</w:p>'
    def table(content): return f'<w:tbl {NS}><w:tr><w:tc>{content}</w:tc></w:tr></w:tbl>'
    styles = f'''<w:styles xmlns:w="{W}">
      <w:style w:type="character" w:styleId="strong"><w:name w:val="Strong"/></w:style>
      <w:style w:type="character" w:styleId="unicode-strong"><w:name w:val="ſtRoNg"/></w:style>
      <w:style w:type="character" w:styleId="inherited"><w:name w:val="Local"/><w:rPr><w:b/></w:rPr></w:style>
      <w:style w:type="paragraph" w:styleId="mixed-heading"><w:name w:val="hEaDiNg 3"/></w:style>
      <w:style w:type="paragraph" w:styleId="custom"><w:name w:val="heading 2"/></w:style>
      <w:style w:type="paragraph" w:styleId="list"><w:name w:val="List"/></w:style>
      <w:style w:type="numbering" w:styleId="linked"><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr></w:style>
    </w:styles>'''
    numbering = f'''<w:numbering xmlns:w="{W}">
      <w:abstractNum w:abstractNumId="1">
        <w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/><w:pStyle w:val="list"/></w:lvl>
        <w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl>
        <w:lvl w:ilvl="2"><w:numFmt w:val="decimal"/></w:lvl>
      </w:abstractNum>
      <w:abstractNum w:abstractNumId="2"><w:numStyleLink w:val="linked"/></w:abstractNum>
      <w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num>
      <w:num w:numId="2"><w:abstractNumId w:val="2"/></w:num>
    </w:numbering>'''
    rels = [dict(id='link',target='https://example.org/a?x=1&y=2#old',kind='hyperlink')]
    generated = []
    for prop in ['', '<w:b/>','<w:i/>','<w:strike/>','<w:u w:val="single"/>',
                 '<w:b/><w:i/><w:strike/>','<w:vertAlign w:val="superscript"/>',
                 '<w:vertAlign w:val="subscript"/>','<w:rStyle w:val="strong"/>',
                 '<w:rStyle w:val="unicode-strong"/>','<w:rStyle w:val="inherited"/>',
                 '<w:b w:val="false"/><w:i w:val="off"/>','<w:vanish/>']:
        generated.append(table(paragraph(run('A &amp; &lt; &quot;',prop)+run('B',prop))))
    for style in ['Heading1','Heading6','Heading7','Heading','custom','list','mixed-heading']:
        generated.append(table(paragraph(run('Title'),f'<w:pStyle w:val="{style}"/>')))
    def list_p(text, level='0', num='1'):
        return paragraph(run(text),f'<w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="{num}"/></w:numPr>')
    generated += [table(''.join(list_p(str(i), str(level)) for i, level in enumerate(levels)))
                  for levels in [[0,1,1,2,0],[1,0,2,1],[4,4,5,0],[0,0,0]]]
    generated += [table(list_p('style link','1','2')),table(list_p('raw zero','00')),
        table(paragraph(run('deleted'),'<w:rPr><w:del/></w:rPr>')+paragraph(run('joined'))),
        table(paragraph(run('never flushed'),'<w:rPr><w:del/></w:rPr>')),
        table(paragraph('<w:hyperlink r:id="link" w:anchor="new" w:tgtFrame="_blank">'+run('Link','<w:b/>')+'</w:hyperlink>')),
        table(paragraph('<w:hyperlink w:anchor="toc">'+run('Anchor')+'</w:hyperlink>'+run(' tail'))),
        table(paragraph('<w:bookmarkStart w:name="bookmark"/>'+run('A')+'<w:bookmarkStart w:name="_GoBack"/>')),
        table(paragraph('<w:r><w:t>A</w:t><w:tab/><w:br/><w:br w:type="page"/><w:t>B</w:t></w:r>')),
        table(paragraph('<w:r><w:fldChar w:fldCharType="begin"/><w:instrText>HYPERLINK "https://example.org"</w:instrText><w:fldChar w:fldCharType="separate"/><w:t>A</w:t><w:fldChar w:fldCharType="end"/></w:r>')),
        table(paragraph('<w:r><w:fldChar w:fldCharType="begin"/><w:instrText>HYPERLINK \\l "anchor"</w:instrText><w:fldChar w:fldCharType="separate"/></w:r>'+run('A','<w:b/>')+run('B','<w:i/>')+'<w:r><w:fldChar w:fldCharType="end"/></w:r>')),
        table(paragraph('<w:r><w:fldChar w:fldCharType="end"/></w:r>')),
        table(paragraph(run('Nested'))+f'<w:tbl><w:tr><w:tc>{paragraph(run("Inner"))}</w:tc></w:tr></w:tbl>'),
        table(paragraph('<w:r><w:pict><w:txbxContent>'+paragraph(run('Text box'))+'</w:txbxContent></w:pict></w:r>')),
    ]
    for code in ['0061','F061','0x_61','6_1','０x61','+0061','-1','F'*40,'invalid','0x','_61','61_']:
        generated.append(table(paragraph(f'<w:r><w:sym w:font="Symbol" w:char="{code}"/></w:r>')))
    for index, xml in enumerate(generated):
        cases.append(capture(xml,f'HTML generated {index}',styles,numbering,rels))
    for case in cases:
        if 'html' in case:
            case['normalized_html'] = _DocxTables()._normalize_table_colspans(case['html'])
    normalizations=[]
    def row(*spans):return '<tr>'+''.join(f'<td colspan="{span}">X</td>' for span in spans)+'</tr>'
    for name,markup in [
        ('majority','<table>'+row(1,1)+row(3,2)+row(1,1)+'</table>'),
        ('tie first','<table>'+row(3,2)+row(1,1)+'</table>'),
        ('reduce left first','<table>'+row(1,1,1)+row(2,4)+row(1,1,1)+'</table>'),
        ('negative and zero','<table>'+row(-1,0)+row(3,2)+row(-1,0)+'</table>'),
        ('large integer','<table>'+row(1)+row(10**45)+row(1)+'</table>'),
        ('invalid','<table>'+row('bad')+row(2)+'</table>'),
        ('invalid late returns original','<table>'+row(1)+row(3)+row(1)+'</table><table>'+row('bad')+'</table>'),
        ('rowspan guard','<table><tr><td rowspan="2">X</td><td>X</td></tr>'+row(5)+row(1,1)+'</table>'),
        ('guard skips invalid colspan but later table still corrected','<table><tr><td rowspan="2" colspan="bad">X</td></tr></table><table>'+row(1)+row(3)+row(1)+'</table>'),
        ('guard short circuits invalid later rowspan','<table><tr><td rowspan="2">X</td><td rowspan="bad">Y</td></tr></table><table>'+row(1)+row(3)+row(1)+'</table>'),
        ('header','<table><thead>'+row(1,1)+'</thead><tbody>'+row(5)+row(1,1)+'</tbody></table>'),
        ('nested','<table><tr><td><table>'+row(1)+row(3)+row(1)+'</table></td></tr>'+row(1,1,1)+'</table>'),
    ]:
        normalizations.append(dict(name=name,html=markup,expected=_DocxTables()._normalize_table_colspans(markup)))
    target=fixture/'docx_table_html.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases,normalizations=normalizations),ensure_ascii=False).encode(),mtime=0))
    print('DOCX table HTML:',len(cases),'cases;',sum('error' in c for c in cases),'expected errors;',len(normalizations),'normalization edge cases')


if __name__ == '__main__': main()
