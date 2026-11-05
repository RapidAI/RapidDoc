"""Capture local table signatures and Mammoth's original span recovery."""
from copy import deepcopy
from io import BytesIO
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def capture(xml,name):
    from lxml import etree
    from mammoth.docx import body_xml,numbering_xml,office_xml
    from mammoth import documents
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.tables import _DocxTables
    node=etree.fromstring(xml.encode())
    signature=_DocxTables._xml_table_signature(node)
    # Span recovery is independent of cell rich content. Use the original reader
    # on a structural view, so absent images/relationships do not replace it with
    # a mock algorithm. Full table HTML is a separate acceptance requirement.
    structure=deepcopy(node)
    for cell in structure.iter(f'{{{W}}}tc'):
        for content in list(cell):
            if content.tag!=f'{{{W}}}tcPr':cell.remove(content)
    try:
        parsed=body_xml.reader(numbering=numbering_xml.Numbering.EMPTY).read_all([office_xml.read(BytesIO(etree.tostring(structure)))])
        table=next(item for item in parsed.value if isinstance(item,documents.Table))
        layout=dict(rows=[dict(header=row.is_header,cells=[dict(colspan=cell.colspan,rowspan=cell.rowspan) for cell in row.children if isinstance(cell,documents.TableCell)]) for row in table.children if isinstance(row,documents.TableRow)],
                    merge_disabled=any('unexpected non-' in message.message for message in parsed.messages))
    except ValueError as error:
        layout=dict(error=type(error).__name__)
    return dict(name=name,xml=xml,signature=signature,layout=layout)


def main():
    from lxml import etree
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.tables import _DocxTables
    old=json.loads(gzip.decompress((ROOT/'tests/fixtures/office_rust_reference_068/docx_context.json.gz').read_bytes()))
    tables=[]
    for case in old['cases']:
        document=etree.fromstring(case['document'].encode())
        for index,node in enumerate(document.iter(f'{{{W}}}tbl')):
            tables.append(capture(etree.tostring(node,encoding='unicode'),f"{case['name']} table {index}"))
    def cell(text,span=1,merge=None):
        props=f'<w:gridSpan w:val="{span}"/>'
        if merge is not None:props+=f'<w:vMerge{(" w:val="+chr(34)+merge+chr(34)) if merge!="missing" else ""}/>'
        return f'<w:tc><w:tcPr>{props}</w:tcPr><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:tc>'
    def row(cells,props=''):return f'<w:tr><w:trPr>{props}</w:trPr>{cells}</w:tr>'
    cases=[
        row(cell('a')+cell('b'))+row(cell('c')+cell('d')),
        row(cell('a',2,'restart')+cell('b'))+row(cell('ignored',2,'continue')+cell('d')),
        row(cell('orphan',1,'continue')),
        row(cell('a')+cell('b'))+row(cell('c'))+row(cell('d')+cell('continue',1,'continue')),
        row(cell('a'),'<w:tblHeader w:val="false"/>')+row(cell('ignored',1,'missing')),
        row(cell('deleted'),'<w:del/>')+row(cell('a',1,'continue')),
        row(cell('zero',0)+cell('negative',-1))+row(cell('tail')),
        row(cell('invalid','invalid')),
        '<w:p><w:r><w:t>Unexpected</w:t></w:r></w:p>'+row(cell('a'))+row(cell('b',1,'continue')),
        row('<w:p/>'+cell('a'))+row(cell('b',1,'continue')),
        '<w:ins>'+row(cell('inserted'))+'</w:ins>',
        '<w:sdt><w:sdtContent>'+row(cell('control'))+'</w:sdtContent></w:sdt>',
        '<w:customXml>'+row(cell('ignored'))+'</w:customXml>',
        row(cell('empty value',1,'')),
        '<mc:AlternateContent><mc:Choice Requires="w14">'+row(cell('choice'))+'</mc:Choice><mc:Fallback>'+row(cell('fallback'))+'</mc:Fallback></mc:AlternateContent>',
        row('<w:tc><w:p><w:r><w:t>A B</w:t><w:noBreakHyphen/><w:softHyphen/><w:sym w:font="Symbol" w:char="F061"/></w:r><w:fldSimple><w:r><w:t>Ignored</w:t></w:r></w:fldSimple><m:oMath><m:r><m:t>Math</m:t></m:r></m:oMath></w:p></w:tc>'),
        row('<w:tc><w:sdt><w:sdtPr><w14:checkbox/></w:sdtPr><w:sdtContent><w:p><w:r><w:t>Check</w:t><w:t>Label</w:t></w:r></w:p></w:sdtContent></w:sdt></w:tc>'),
    ]
    declaration=f'xmlns:w="{W}" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
    for index,body in enumerate(cases):tables.append(capture(f'<w:tbl {declaration}>{body}</w:tbl>',f'generated {index}'))
    fragments=[]
    for font in ['Symbol','Wingdings','Webdings','unknown']:
        for code in ['0061','F061','F0A3','F0a3','0x61','invalid','', 'F0','-1']:
            xml=f'<w:sym xmlns:w="{W}" w:font="{font}" w:char="{code}"/>'
            fragments.append(dict(xml=xml,expected=_DocxTables._xml_table_char_fragment(etree.fromstring(xml.encode()))))
    matches=[]
    signatures=[t['signature'] for t in tables]
    for index,signature in enumerate(signatures):
        for other in [signature,dict(signature,text=signature['text']+'tail'),dict(signature,text=''),dict(signature,row_count=signature['row_count']+1,cell_count=signature['cell_count']+1)]:
            matches.append(dict(xml=signature,html=other,expected=_DocxTables._table_signatures_match(signature,other)))
    target=ROOT/'tests/fixtures/office_rust_reference_068/docx_tables.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(tables=tables,fragments=fragments,matches=matches),ensure_ascii=False).encode(),mtime=0))
    print('DOCX table references:',len(tables),'tables,',len(fragments),'symbols,',len(matches),'matching cases')


if __name__=='__main__':main()
