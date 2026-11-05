"""Freeze current DOCX paragraph/style/field/numbering behavior; no production code."""
from io import BytesIO
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def capture(data, name, source=None):
    from docx import Document
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run
    from lxml import etree
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    def number(value):return None if value is None else str(value)
    converter = DocxConverter()
    document = Document(BytesIO(converter._sanitize_missing_internal_relationships(data)))
    converter.docx_obj = document
    paragraphs = []
    for node in document.element.iter(f"{{{W}}}p"):
        paragraph = Paragraph(node, document)
        try:
            contents = list(converter._iter_paragraph_inner_content(paragraph))
            label, heading = converter._get_label_and_level(paragraph)
            num_id, level = converter._get_numId_and_ilvl(paragraph)
            runs = []
            for content in contents:
                for run in ([content] if isinstance(content, Run) else content.runs):
                    hidden = converter._is_hidden_run(run)
                    runs.append(dict(text=run.text or "", hidden=hidden,
                                     format=None if hidden else converter._get_format_from_run(run).model_dump(mode="json")))
            elements = [dict(text=text, format=fmt.model_dump(mode="json") if fmt is not None else None,
                             hyperlink=str(link) if link is not None else None)
                        for text, fmt, link in converter._get_paragraph_elements(paragraph)]
            paragraphs.append(dict(text=converter._get_paragraph_text(paragraph), label=label,
                                   heading_level=heading, num_id=num_id, level=level,
                                   outline_level=converter._get_effective_outline_level(paragraph),
                                   toc_level=converter._get_toc_item_level(paragraph),
                                   numbered=bool(num_id is not None and level is not None and converter._is_numbered_list(num_id,level)),
                                   start=number(converter._get_numbering_level_start(num_id,level)) if num_id is not None and level is not None else None,
                                   elements=elements,runs=runs,
                                   field_instructions=converter._complex_field_instructions(node)))
        except Exception as error:
            paragraphs.append(dict(error=type(error).__name__))
    numbering = converter._get_numbering_root()
    counters = []
    for num_id, level in [(42,0),(42,1),(42,1),(42,0),(42,1),(43,0),(42,2),(42,1),(999,0),(44,0),(44,0)]:
        try:
            counters.append(dict(id=num_id,level=level,start=number(converter._get_numbering_level_start(num_id,level)),
                                 numbered=converter._is_numbered_list(num_id,level),next=number(converter._advance_list_counter(num_id,level))))
        except Exception as error:
            counters.append(dict(id=num_id,level=level,error=type(error).__name__))
    return dict(name=name,source=source,document=etree.tostring(document.element,encoding="unicode"),
                styles=etree.tostring(document.styles.element,encoding="unicode"),
                numbering=etree.tostring(numbering,encoding="unicode") if numbering is not None else None,
                relationships=[dict(id=rel.rId,kind=rel.reltype,target=rel.target_ref,external=rel.is_external,resolved=None)
                               for rel in document.part.rels.values()],paragraphs=paragraphs,counters=counters)


def synthetic():
    from docx import Document
    from docx.oxml import OxmlElement, parse_xml
    from docx.oxml.ns import nsdecls
    from docx.enum.style import WD_STYLE_TYPE
    document = Document()
    body = document.element.body
    for child in list(body):
        body.remove(child)
    def paragraph(content):
        body.append(parse_xml(f'<w:p {nsdecls("w","r")}>{content}</w:p>'))
    normal = document.styles['Normal']
    normal.font.bold = True
    parent = document.styles.add_style('CustomBase',WD_STYLE_TYPE.PARAGRAPH)
    parent.font.italic = True
    parent.element.get_or_add_pPr().append(parse_xml(f'<w:numPr {nsdecls("w")}><w:numId w:val="42"/></w:numPr>'))
    child = document.styles.add_style('CustomChild',WD_STYLE_TYPE.PARAGRAPH)
    child.base_style=parent
    character = document.styles.add_style('StrongChar',WD_STYLE_TYPE.CHARACTER)
    character.font.bold=False
    custom = document.styles.add_style('CustomOutline',WD_STYLE_TYPE.PARAGRAPH)
    custom.element.get_or_add_pPr().append(parse_xml(f'<w:outlineLvl {nsdecls("w")} w:val="3"/>'))
    linkstyle = document.styles.add_style('LinkCharacter',WD_STYLE_TYPE.CHARACTER)
    linkstyle.name='Custom Hyperlink Style';linkstyle.font.underline=True
    tocstyle=document.styles.add_style('TOC 1',WD_STYLE_TYPE.PARAGRAPH)
    tocstyle.style_id='TOC1'
    for props in ['', '<w:b w:val="false"/>','<w:i/>','<w:u w:val="single"/>','<w:u w:val="words"/>',
                  '<w:em w:val="dot"/>','<w:strike/>','<w:vertAlign w:val="subscript"/>','<w:vertAlign w:val="superscript"/>']:
        paragraph(f'<w:r><w:rPr>{props}</w:rPr><w:t>Alpha</w:t></w:r><w:r><w:t xml:space="preserve"> </w:t></w:r><w:r><w:rPr>{props}</w:rPr><w:t>Beta</w:t></w:r>')
        paragraph(f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">   </w:t></w:r>')
    for style in ['CustomChild','CustomOutline','Heading1','MissingStyle','TOC1']:
        paragraph(f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:rPr><w:rStyle w:val="StrongChar"/></w:rPr><w:t>Styled</w:t></w:r>')
    for label in ['Heading١','Heading２','3Heading','TOC ١','目录 ２','Prefix heading2']:
        style=document.styles.add_style(label,WD_STYLE_TYPE.PARAGRAPH)
        paragraph(f'<w:pPr><w:pStyle w:val="{style.style_id}"/></w:pPr><w:r><w:t>Unicode style</w:t></w:r>')
    for name in ['CycleA','CycleB']:
        document.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
    document.styles['CycleA'].base_style=document.styles['CycleB'];document.styles['CycleB'].base_style=document.styles['CycleA']
    paragraph('<w:pPr><w:pStyle w:val="CycleA"/></w:pPr><w:r><w:t>Cycle</w:t></w:r>')
    for wrapper in ['ins','moveTo','smartTag','customXml','bdo','dir','fldSimple','del','moveFrom']:
        paragraph(f'<w:{wrapper}><w:r><w:t>{wrapper}</w:t></w:r></w:{wrapper}><w:sdt><w:sdtContent><w:r><w:t>Control</w:t></w:r></w:sdtContent></w:sdt>')
    paragraph('<w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/><w:noBreakHyphen/><w:ptab/><w:br w:type="page"/><w:t>C</w:t></w:r>')
    paragraph('<w:r><w:rPr><w:vanish w:val="false"/></w:rPr><w:t>Hidden</w:t></w:r><w:r><w:t>Visible</w:t></w:r>')
    for instruction, closed in [(' HYPERLINK &quot;https://example.test/a&quot; ',True),(' HYPERLINK \\l &quot;section&quot; ',True),(' PAGE ',True),(' HYPERLINK &quot;https://example.test/open&quot; ',False)]:
        paragraph('<w:r><w:fldChar w:fldCharType="begin"/></w:r>'+f'<w:r><w:instrText>{instruction}</w:instrText></w:r>'+'<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>FieldResult</w:t></w:r>'+('<w:r><w:fldChar w:fldCharType="end"/></w:r>' if closed else ''))
    paragraph('<w:r><w:instrText>ORPHAN</w:instrText><w:t>OrphanResult</w:t></w:r>')
    paragraph('<w:hyperlink w:anchor="mark"><w:r><w:rPr><w:rStyle w:val="LinkCharacter"/></w:rPr><w:t xml:space="preserve">  link  </w:t></w:r></w:hyperlink><w:r><w:t>Tail</w:t></w:r>')
    paragraph('<w:pPr><w:pStyle w:val="TOC1"/></w:pPr><w:hyperlink w:anchor="tocmark"><w:r><w:t>Internal</w:t></w:r></w:hyperlink><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> HYPERLINK \\l &quot;mark&quot; </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Field</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>')
    paragraph('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> HYPERLINK &quot;https://example.test/outer&quot; </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Outer</w:t></w:r><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText> HYPERLINK &quot;https://example.test/inner&quot; </w:instrText></w:r><w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Inner</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r>')
    paragraph('<w:hyperlink w:anchor="mark"><w:r><w:rPr><w:vanish w:val="false"/></w:rPr><w:t>HiddenLink</w:t></w:r><w:r><w:t>VisibleLink</w:t></w:r></w:hyperlink>')
    from docx.opc.constants import RELATIONSHIP_TYPE
    for target in ['https://example.test/a#b','relative/manual.docx','mailto:user@example.test','../relative.docx']:
        rid=document.part.relate_to(target,RELATIONSHIP_TYPE.HYPERLINK,is_external=True)
        paragraph(f'<w:hyperlink r:id="{rid}"><w:r><w:t>Linked</w:t></w:r></w:hyperlink>')
    numbering=document.part.numbering_part.element
    numbering.append(parse_xml(f'<w:abstractNum {nsdecls("w")} w:abstractNumId="99"><w:lvl w:ilvl="0"><w:start w:val="3"/><w:numFmt w:val="decimal"/><w:pStyle w:val="CustomChild"/></w:lvl><w:lvl w:ilvl="1"><w:start w:val="5"/><w:numFmt w:val="lowerLetter"/></w:lvl><w:lvl w:ilvl="2"><w:start w:val="1"/><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>'))
    numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="42"><w:abstractNumId w:val="99"/><w:lvlOverride w:ilvl="1"><w:startOverride w:val="9"/></w:lvlOverride></w:num>'))
    numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="43"><w:abstractNumId w:val="99"/></w:num>'))
    numbering.append(parse_xml(f'<w:abstractNum {nsdecls("w")} w:abstractNumId="100"><w:lvl w:ilvl="0"><w:start w:val="1234567890123456789012345678901234567890"/><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>'))
    numbering.append(parse_xml(f'<w:num {nsdecls("w")} w:numId="44"><w:abstractNumId w:val="100"/></w:num>'))
    for props in ['<w:numId w:val="42"/><w:ilvl w:val="1"/>','<w:numId w:val="42"/>','<w:numId w:val="0"/>','<w:numId w:val="999"/>','']:
        paragraph(f'<w:pPr><w:pStyle w:val="CustomChild"/><w:numPr>{props}</w:numPr></w:pPr><w:r><w:t>Number</w:t></w:r>')
    for invalid in ['<w:u/>','<w:b w:val="invalid"/>','<w:u w:val="invalid"/>','<w:vertAlign w:val="unknown"/>','<w:vertAlign/>']:
        paragraph(f'<w:r><w:rPr>{invalid}</w:rPr><w:t>Malformed</w:t></w:r>')
    out=BytesIO();document.save(out)
    return out.getvalue()


def main():
    manifest=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/manifest.json').read_bytes())
    cases=[capture((ROOT/case['input']).read_bytes(),case['id'],case['input']) for case in manifest['cases'] if case['format']=='docx']
    cases.append(capture(synthetic(),'style_field_numbering_combinations'))
    payload=dict(cases=cases)
    target=ROOT/'tests/fixtures/office_rust_reference_068/docx_context.json.gz'
    target.write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False).encode(),mtime=0))
    print('DOCX context references:',len(cases),'documents,',sum(len(c['paragraphs']) for c in cases),'paragraphs')


if __name__=='__main__':main()
