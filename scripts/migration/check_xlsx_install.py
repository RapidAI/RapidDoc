"""Installed native XLSX public API checks, forbidding Python Office parsers."""
import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[2]

def main():
    class Guard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in{'openpyxl','mammoth','docx','olefile'}or fullname.startswith('rapid_doc.'):
                raise AssertionError('Python Office import: '+fullname)
    sys.meta_path.insert(0,Guard())
    from rapid_doc_rust import RapidDoc
    from rapid_doc_rust._native import parse_docx_bytes,parse_office_bytes
    from loguru import logger
    logger.remove()
    engine=RapidDoc()
    def forbidden(*a,**kw):raise AssertionError('Office must not load recognition models')
    engine._get_models=forbidden
    paths=sorted((ROOT/'demo/xlsx').glob('*.xlsx'))+[ROOT/'demo/ms_office_docs/xlsx_01.xlsx']
    reports=[]
    for path in paths:
        result=engine(path)
        assert result.middle_json['_backend']=='office'
        assert result.markdown is not None
        assert engine._models is None
        assert len(result.middle_json['pdf_info'])>=1
        reports.append(dict(source=str(path.relative_to(ROOT)),sheets=len(result.middle_json['pdf_info']),images=len(result.images),markdown_chars=len(result.markdown)))
    source=paths[0];raw=source.read_bytes();expected=engine(source)
    class NonSeekable:
        def __init__(self):self.value=raw
        def read(self):return self.value
    for source_value in [raw,bytearray(raw),BytesIO(raw),NonSeekable()]:
        got=engine(source_value)
        assert got.markdown==expected.markdown and got.middle_json==expected.middle_json and got.images==expected.images
    for got in engine([raw,source]):assert got.markdown==expected.markdown
    uri=engine(source,image_output_mode='data_uri');assert 'data:image/'in uri.markdown
    with tempfile.TemporaryDirectory(prefix='rapid-doc-xlsx-')as directory:
        engine(source,output_dir=directory)
        folder=Path(directory)/source.stem/'office'
        assert (folder/(source.stem+'.md')).read_text(encoding='utf-8')==expected.markdown
        assert json.loads((folder/(source.stem+'_middle.json')).read_text(encoding='utf-8'))==expected.middle_json
        assert all((folder/'images'/name).read_bytes()==value for name,value in expected.images.items())
        empty=Path(directory)/'empty.xlsx';empty.write_bytes(b'');assert engine(empty).markdown==''
    class Writer:
        def __init__(self):self.data={}
        def write(self,name,data):self.data[name]=data
    iw,mw=Writer(),Writer();engine(source,image_writer=iw,md_writer=mw)
    assert iw.data==expected.images and mw.data[source.stem+'.md'].decode()==expected.markdown
    for mode in ['mm_markdown','nlp_markdown','content_list','content_list_v2']:
        result=RapidDoc(make_md_mode=mode)(source)
        assert result.markdown is not None
    # DOCX continues through the same public Office dispatcher and old native API.
    docx=(ROOT/'demo/docx/equations.docx').read_bytes()
    old,old_assets=parse_docx_bytes(docx);new,new_assets=parse_office_bytes(docx)
    assert json.loads(old)==json.loads(new) and old_assets==new_assets
    assert engine(docx).markdown
    receipt=dict(local_files=reports,io_variants=4,batch=True,data_uri=True,disk_output=True,writers=True,empty_xlsx=True,export_modes=4,docx_compatibility=True,no_python_office_imports=True,no_recognition_models=True)
    (ROOT/'build/xlsx-installed-checks.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
