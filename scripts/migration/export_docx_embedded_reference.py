"""Original DOCX per-part MTEF arbitration, preview filtering and table injection."""
from pathlib import Path
from io import BytesIO
import gzip,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/native_office')]
def main():
    import _mtef_test_utils as v3
    import _mtef_v5_test_utils as v5
    import _image_mtef_test_utils as images
    from docx import Document
    from docx.oxml import parse_xml
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from lxml import etree
    from xml.sax.saxutils import escape
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from loguru import logger
    from PIL import Image
    logger.remove()
    fixture=ROOT/'tests/fixtures/office_rust_reference_068'
    target=fixture/'mtef_packages';target.mkdir(exist_ok=True)
    sources=[]
    png=BytesIO();Image.new('RGB',(1,1),'white').save(png,format='PNG')
    for mode in ['ole','ole_v5','icon','linked','invalid_ole','gif_v3','gif_v5']:
        document=Document();paragraph=document.add_table(1,1).cell(0,0).paragraphs[0];paragraph.add_run('Before ')
        if mode.startswith('gif'):
            payload=images.build_gif_with_mtef(v5.v5_equation(v5.v5_text('x+y')) if mode=='gif_v5' else v3.mtef_equation(v3.mtef_text('x+y')))
            rid,_=document.part.get_or_add_image(BytesIO(payload))
            body=f'<w:pict><v:shape><v:imagedata r:id="{rid}"/></v:shape></w:pict>'
        else:
            data=v3.equation_native(v5.v5_equation(v5.v5_text('a+b'))) if mode=='ole_v5' else v3.equation_native(v3.mtef_equation(v3.mtef_text('a+b')))
            blob=v3._build_nested_cfb([dict(name='Root Entry',kind=5,child=1),dict(name='Equation Native',kind=2,data=data)])
            if mode=='invalid_ole':blob=b'invalid OLE object'
            part=Part(PackURI('/word/embeddings/formula.bin'),'application/vnd.openxmlformats-officedocument.oleObject',blob,document.part.package)
            rid=document.part.relate_to(part,RT.OLE_OBJECT)
            preview,_=document.part.get_or_add_image(BytesIO(png.getvalue()))
            options=' DrawAspect="Icon"' if mode=='icon' else ' Type="Link"' if mode=='linked' else ''
            body=f'<w:object><v:shape><v:imagedata r:id="{preview}"/></v:shape><o:OLEObject ProgID="Equation.3" r:id="{rid}"{options}/></w:object>'
        paragraph._p.append(parse_xml('<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">'+body+'</w:r>'))
        paragraph.add_run(' After')
        path=target/(mode+'.docx');document.save(path)
        sources.append((mode,path))
    manifest=json.loads((fixture/'manifest.json').read_text())
    sources.extend((c['id'],ROOT/c['input']) for c in manifest['cases'] if c['format']=='docx')
    cases=[]
    for name,path in sources:
        converter=DocxConverter();payload=converter._sanitize_missing_internal_relationships(path.read_bytes())
        document=Document(BytesIO(payload));converter.docx_obj=document;converter._fallback_docx_bytes=payload
        tables=[]
        try:
            for index,node in enumerate(document.element.body.findall('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl')):
                tokens=converter._docx_formula_tokens(node,document.part)
                previews=[converter._picture_is_equation_preview(n,document.part) for n in node.iter() if isinstance(n.tag,str) and etree.QName(n).localname in {'blip','imagedata'}]
                raw=converter._mammoth_fallback_html(node)
                def render():
                    injected=converter._inject_equations_into_table_html(raw,node,document.part)
                    return converter._normalize_table_links(converter._normalize_table_colspans(injected))
                original=converter._docx_formula_tokens
                def escaped(n,p):
                    converter._docx_formula_tokens=original
                    try:return [(kind,escape(value)) if kind!='text' else (kind,value) for kind,value in original(n,p)]
                    finally:converter._docx_formula_tokens=escaped
                legacy=render()
                try:
                    converter._docx_formula_tokens=escaped;safe=render()
                finally:converter._docx_formula_tokens=original
                tables.append(dict(index=index,tokens=tokens,previews=previews,html=legacy,safe_html=safe))
            cases.append(dict(name=name,path=str(path.relative_to(ROOT)).replace('\\','/'),package_hex=payload.hex(),tables=tables,image_bytes=converter._image_equation_decoder.total_bytes,ole_bytes=converter._ooxml_equation_decoder.total_bytes))
        finally:converter._close_mammoth_fallback_context()
    (fixture/'docx_embedded.json.gz').write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('DOCX embedded references:',len(cases),'packages;',sum(len(c['tables']) for c in cases),'tables;',sum(len(t['previews']) for c in cases for t in c['tables']),'preview queries')
if __name__=='__main__':main()
