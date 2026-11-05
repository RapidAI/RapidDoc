"""Verify installed PPTX public API against frozen outputs; no Python parsers."""
import base64,gzip,hashlib,json,sys,tempfile
from io import BytesIO
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/migration'))
def main():
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in{'openpyxl','mammoth','docx','olefile','pptx'}or fullname.startswith('rapid_doc.'):
                raise AssertionError('Python Office parser imported: '+fullname)
    sys.meta_path.insert(0,Guard())
    from rapid_doc_rust import RapidDoc
    from rapid_doc_rust._native import parse_office_bytes,parse_docx_bytes
    from export_office_path_pixels import normalized_svg
    from PIL import Image
    from loguru import logger
    logger.remove();engine=RapidDoc()
    def forbidden(*a,**kw):raise AssertionError('Office must not load recognition models')
    engine._get_models=forbidden
    base=ROOT/'tests/fixtures/office_rust_reference_068';sources=json.loads(gzip.decompress((base/'pptx_context.json.gz').read_bytes()))['cases'];refs=json.loads(gzip.decompress((base/'pptx_output.json.gz').read_bytes()))['cases']
    checked=0;errors=0
    def verify(result,expected):
        assert len(result.images)==len(expected['assets']);aliases={};used=set()
        for name,data in result.images.items():
            if name.endswith('.svg'):summary=dict(svg=normalized_svg(data))
            else:
                with Image.open(BytesIO(data))as image:summary=dict(size=list(image.size),sha256=hashlib.sha256(image.convert('RGBA').tobytes()).hexdigest())
            other=next(k for k,v in expected['assets'].items()if k not in used and v==summary);used.add(other);aliases[name]=other
        def rename(v):
            if isinstance(v,str):
                for a,b in aliases.items():v=v.replace(a,b)
                return v
            if isinstance(v,list):return [rename(x)for x in v]
            if isinstance(v,dict):return {k:rename(x)for k,x in v.items()}
            return v
        if rename(result.middle_json)!=expected['middle']:
            (ROOT/'build/pptx-installed-diff.json').write_text(json.dumps(dict(name=expected['name'],actual=rename(result.middle_json),expected=expected['middle']),ensure_ascii=False,indent=2),encoding='utf-8')
            raise AssertionError('Installed middle differs: '+expected['name'])
        assert rename(result.markdown)==expected['exports']['mm_markdown']
        assert rename(result.content_list_json)==expected['exports']['content_list']
    for expected in refs:
        source=sources[expected['source_index']];data=base64.b64decode(source['package_b64'])
        if 'error'in expected:
            try:engine(data)
            except Exception:errors+=1
            else:raise AssertionError('Expected original PPTX error: '+expected['name'])
        else:
            verify(engine(data),expected);checked+=1
            if source.get('reference_normalized_input'):verify(engine(ROOT/source['name']),expected)
    source=ROOT/'demo/pptx/powerpoint_sample.pptx';raw=source.read_bytes();expected=engine(source)
    class NonSeekable:
        def read(self):return raw
    for value in [raw,bytearray(raw),BytesIO(raw),NonSeekable()]:
        actual=engine(value);assert actual.markdown==expected.markdown and actual.middle_json==expected.middle_json and actual.images==expected.images
    assert len(engine([source,raw]))==2
    uri=engine(source,image_output_mode='data_uri');assert 'data:image/'in uri.markdown
    with tempfile.TemporaryDirectory(prefix='rapid-doc-pptx-')as folder:
        engine(source,output_dir=folder);out=Path(folder)/source.stem/'office';assert (out/(source.stem+'.md')).read_text(encoding='utf-8')==expected.markdown
        assert all((out/'images'/k).read_bytes()==v for k,v in expected.images.items())
    class Writer:
        def __init__(self):self.data={}
        def write(self,name,data):self.data[name]=data
    iw,mw=Writer(),Writer();engine(source,image_writer=iw,md_writer=mw);assert iw.data==expected.images and mw.data[source.stem+'.md'].decode()==expected.markdown
    for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']:
        result=RapidDoc(make_md_mode=mode)(source);metadata,_=parse_office_bytes(raw,make_mode=mode);assert result.markdown==json.loads(metadata)['markdown']
    docx=(ROOT/'demo/docx/equations.docx').read_bytes();a,assets=parse_docx_bytes(docx);b,images=parse_office_bytes(docx);assert json.loads(a)==json.loads(b)and assets==images
    assert engine(ROOT/'demo/xlsx/xlsx_01.xlsx').markdown
    assert engine(ROOT/'demo/ms_office_docs/xlsx_01.xls').markdown
    for name in ['handmade-strict.pptx','handmade-altpath.pptx','handmade-order.pptx']:assert engine(ROOT/'tests/fixtures/pptx'/name).markdown
    assert engine._models is None
    receipt=dict(frozen_successes=checked,original_errors=errors,io_variants=4,batch=True,data_uri=True,disk_output=True,writers=True,export_modes=4,docx_xlsx_xls_compatibility=True,original_packages=True,no_python_office_imports=True,no_recognition_models=True)
    (ROOT/'build/pptx-installed-checks.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
