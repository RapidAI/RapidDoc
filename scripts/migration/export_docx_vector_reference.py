"""Original DOCX resource conversion with constructed WMF/EMF parts."""
import base64,gzip,json,sys,re
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from export_office_path_pixels import normalized_svg
def main():
    from docx import Document
    from docx.oxml import parse_xml
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from loguru import logger
    from metafile_render import render_metafile
    logger.remove()
    ref=ROOT/'tests/fixtures/office_rust_reference_068'
    paths=json.loads(gzip.decompress((ref/'metafile_paths.json.gz').read_bytes()))
    cases=[]
    # Drawings include stock pens, clip fallback, images, negative extents and
    # unknown-ROP source fallback; parts repeat twice to exercise source reuse.
    sources=[c for c in paths['final'] if c['name'].startswith('mixed ')][:12]
    sources += [c for c in paths['final'] if c['name'] in ['WMF rectangle','WMF standard bounds inferred','EMF shape 43','EMF partial diagnostics']]
    sources += [c for c in paths['final'] if c['name'].startswith('EMF text align ')][:8]
    sources += [c for c in paths['final'] if c['name'].startswith('WMF ExtTextOut ')][:8]
    sources += [c for c in paths['final'] if c['name'].startswith('EMF ANSI charset ')][:6]
    sources += [c for c in paths['final'] if c['name'].startswith('EMF font substitution ')]
    plus=json.loads(gzip.decompress((ref/'emfplus.json.gz').read_bytes()))
    sources += [c for c in plus['cases'] if 'document' in c and all('error' not in out for out in c['outputs'].values())]
    sources += [c for c in plus['cases'] if 'error' in c]
    sources += [dict(name='invalid vector '+data.hex(),hex=data.hex()) for data in [b'',b'not vector',b'\xd7\xcd\xc6\x9a',b'\x01\0\x09\0']]
    for index,c in enumerate(sources):
        data=bytes.fromhex(c['hex']);fmt='emf' if data[:4]==b'\x01\0\0\0' else 'wmf'
        document=Document();cell=document.add_table(1,1).cell(0,0);p=cell.paragraphs[0]
        part=Part(PackURI(f'/word/media/vector.{fmt}'),f'image/{fmt}',data,document.part.package);rid=document.part.relate_to(part,RT.IMAGE)
        body='<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office"><w:pict><v:shape><v:imagedata r:id="'+rid+'" o:title="vector &amp; label"/></v:shape></w:pict></w:r>'
        drawing='<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:drawing><wp:inline><wp:docPr id="1" name="vector" descr="vector &amp; label"/><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="'+rid+'"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        p.add_run('Before ');p._p.append(parse_xml(body));p.add_run(' Between ');p._p.append(parse_xml(drawing));p.add_run(' After')
        raw=BytesIO();document.save(raw);raw=raw.getvalue()
        converter=DocxConverter();converter.docx_obj=document;converter._fallback_docx_bytes=raw
        try:html=converter._mammoth_fallback_html(document.element.body.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl'))
        finally:converter._close_mammoth_fallback_context()
        images=[]
        def replace(match):
            images.append(normalized_svg(base64.b64decode(match.group(1))));return 'data:image/svg+xml;base64,SVG['+str(len(images)-1)+']'
        html=re.sub(r'data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)',replace,html)
        assert len(images)==2 or html.count('data:image/jpeg;base64,')==2,c['name']
        try:
            rendered=render_metafile(data,output_format='svg',dpi=200,backend='auto')
            report=dict(width=rendered.width,height=rendered.height,partial=rendered.partial,diagnostics=[dict(code=d.code,level=d.level,message=d.message,record_type=d.record_type,record_index=d.record_index,offset=d.offset) for d in rendered.diagnostics])
        except Exception:
            report=None
        cases.append(dict(name=c['name'],package_hex=raw.hex(),html=html,images=images,report=report))
    (ref/'docx_vectors.json.gz').write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print('DOCX vector references:',len(cases),'packages;',sum(len(c['images'])+c['html'].count('data:image/jpeg;base64,') for c in cases),'image placements')
if __name__=='__main__':main()
