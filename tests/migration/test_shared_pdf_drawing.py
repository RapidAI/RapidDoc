"""Compare shared Rust overlays against the original Python drawing rules."""
import json
from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader,PdfWriter
from rapid_doc.utils import draw_bbox as reference
from rapid_doc_rust._native import draw_pdf_bytes
from rapid_doc_rust import render_pdf_page
from drawing_fixtures import source,facts


def pixels(data):
    page=render_pdf_page(data,dpi=108)
    return np.asarray(Image.open(BytesIO(page.png)).convert('RGB')).astype(np.int16)


@pytest.mark.parametrize('rotation',[0,90,180,270])
@pytest.mark.parametrize('crop',[False,True])
@pytest.mark.parametrize('kind',['layout','span','line_sort'])
def test_vector_overlay_matches_python(rotation,crop,kind,tmp_path):
    data=source(rotation,crop);pages=[facts()]
    getattr(reference,'draw_'+kind+'_bbox')(pages,data,str(tmp_path),'reference.pdf')
    expected=pixels((tmp_path/'reference.pdf').read_bytes())
    actual=draw_pdf_bytes(data,json.dumps(pages),kind)
    rendered=pixels(actual)
    assert rendered.shape==expected.shape
    delta=np.abs(rendered-expected)
    assert delta.mean()<.12,(rotation,crop,kind,float(delta.mean()),float((delta>10).mean()))
    assert (delta>10).mean()<.001
    reader=PdfReader(BytesIO(actual))
    assert len(reader.pages)==1
    assert 'Original vector content 123' in reader.pages[0].extract_text()
    assert reader.pages[0].get('/Rotate',0)==rotation
    assert '/RD_Overlay0' in reader.pages[0]['/Resources']['/XObject']


def test_selected_pages_inherited_resources_and_rotation(tmp_path):
    writer=PdfWriter()
    for rotation in [0,90,270]:writer.add_page(PdfReader(BytesIO(source(rotation))).pages[0])
    # Inherit from Pages without mutating sibling resources.
    page=writer.pages[1];parent=page['/Parent'];parent.update({__import__('pypdf').generic.NameObject('/Resources'):page['/Resources']})
    del page['/Resources']
    data=BytesIO();writer.write(data)
    out=draw_pdf_bytes(data.getvalue(),json.dumps([facts()]),'layout',1,2)
    reader=PdfReader(BytesIO(out));assert len(reader.pages)==1
    assert reader.pages[0]['/Rotate']==90
    assert 'Original vector content 123' in reader.pages[0].extract_text()
    selected=PdfWriter();selected.add_page(PdfReader(BytesIO(data.getvalue())).pages[1]);s=BytesIO();selected.write(s)
    reference.draw_layout_bbox([facts()],s.getvalue(),str(tmp_path),'selected.pdf')
    assert np.abs(pixels(out)-pixels((tmp_path/'selected.pdf').read_bytes())).mean()<.12


def test_invalid_range_and_coordinates_are_explicit():
    data=source();page=facts()
    with pytest.raises(Exception,match='selected PDF range'):draw_pdf_bytes(data,'[]','layout')
    page['para_blocks'][0]['bbox']=[1,2,3]
    with pytest.raises(Exception,match='drawing coordinates'):draw_pdf_bytes(data,json.dumps([page]),'layout')


def test_production_drawing_without_python_pdf_libraries(tmp_path):
    import subprocess,sys
    code='''import sys,importlib.abc
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('pypdf','reportlab'):raise AssertionError('Python PDF library used: '+fullname)
sys.meta_path.insert(0,Block())
from rapid_doc_rust import RapidDoc
engine=RapidDoc(layout_config={'model_type':'docvortex_rule'},formula_enable=False,table_enable=False)
source=Path('tests/migration/inputs/pdf_native.pdf')
baseline=engine(source)
result=engine(source,output_dir=sys.argv[1],f_draw_layout_bbox=True,f_draw_span_bbox=True)
assert result.markdown==baseline.markdown and result.middle_json==baseline.middle_json
assert result.images==baseline.images and result.content_list_json==baseline.content_list_json
for suffix in ('layout','span'):
 path=Path(sys.argv[1])/'pdf_native'/'auto'/('pdf_native_'+suffix+'.pdf')
 assert path.read_bytes().startswith(b'%PDF-')
class Writer:
 def __init__(self):self.files={}
 def write(self,path,data):self.files[path]=data
writer=Writer()
engine(source,md_writer=writer,f_draw_layout_bbox=True,f_draw_span_bbox=True)
assert writer.files['pdf_native_layout.pdf'].startswith(b'%PDF-')
assert writer.files['pdf_native_span.pdf'].startswith(b'%PDF-')
assert 'rapid_doc_rust._drawing' not in sys.modules
assert 'rapid_doc_rust._preview' not in sys.modules
assert not any(name.split('.')[0] in ('pypdf','reportlab') for name in sys.modules)
'''
    subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,check=True)


def test_line_sort_uses_each_pages_own_lines(tmp_path):
    writer=PdfWriter()
    for _ in range(2):writer.add_page(PdfReader(BytesIO(source())).pages[0])
    data=BytesIO();writer.write(data)
    pages=[facts(),facts()]
    pages[1]['preproc_blocks']=[dict(type='text',lines=[dict(bbox=[200,220,240,240],index=9)])]
    actual=draw_pdf_bytes(data.getvalue(),json.dumps(pages),'line_sort')
    for i,page in enumerate(pages):
        reference.draw_line_sort_bbox([page],source(),str(tmp_path),'one.pdf')
        expected=pixels((tmp_path/'one.pdf').read_bytes())
        rendered=np.asarray(Image.open(BytesIO(render_pdf_page(actual,page_index=i,dpi=108).png)).convert('RGB')).astype(np.int16)
        assert np.abs(rendered-expected).mean()<.12


@pytest.mark.parametrize('encrypted',[False,True])
def test_annotations_and_empty_password_pdf_are_preserved(encrypted,tmp_path):
    from pypdf.generic import DictionaryObject,NameObject,TextStringObject,RectangleObject,ArrayObject,NumberObject
    writer=PdfWriter();page=writer.add_page(PdfReader(BytesIO(source())).pages[0])
    annotation=DictionaryObject({NameObject('/Type'):NameObject('/Annot'),NameObject('/Subtype'):NameObject('/Link'),NameObject('/Rect'):RectangleObject([20,20,50,50]),NameObject('/Border'):ArrayObject([NumberObject(0)]*3),NameObject('/A'):DictionaryObject({NameObject('/S'):NameObject('/URI'),NameObject('/URI'):TextStringObject('https://example.test/')})})
    page[NameObject('/Annots')]=ArrayObject([writer._add_object(annotation)])
    if encrypted:writer.encrypt(user_password='')
    buffer=BytesIO();writer.write(buffer);data=buffer.getvalue()
    actual=draw_pdf_bytes(data,json.dumps([facts()]),'layout')
    reader=PdfReader(BytesIO(actual))
    assert not reader.is_encrypted
    assert reader.pages[0]['/Annots'][0].get_object()['/A']['/URI']=='https://example.test/'
    reference.draw_layout_bbox([facts()],data,str(tmp_path),'annotated.pdf')
    assert np.abs(pixels(actual)-pixels((tmp_path/'annotated.pdf').read_bytes())).mean()<.12
