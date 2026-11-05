"""Memory-only page facades for the original auto classifier; PDFium stays in Rust."""
import json
from contextlib import nullcontext
from io import BytesIO
from types import SimpleNamespace
from contextvars import ContextVar

CLASSIFY_RANGE=ContextVar('rapid_doc_rust_classify_range',default=None)

pdfium_guard=nullcontext
def close_pdfium_child(value):pass
def close_pdfium_document(value):
    if value is not None:value.pages.clear()
def open_pdfium_document(opener,*args,**kwargs):return opener(*args,**kwargs)
def char_font_name(page,index):return page.chars[index][3]

class TextPage:
    def __init__(self,page):self.page=page;self.chars=page.facts['chars'];self.raw=self
    def count_chars(self):return len(self.chars)
    def get_text_bounded(self):return self.page.facts['text']
    def get_charbox(self,index):return self.chars[index][4]

class PageObject:
    def __init__(self,value):self.kind,self.box=value;self.raw=self
    def get_pos(self):return self.box

class Page:
    def __init__(self,doc,index):
        from ._native import native_classification_page_json
        self.doc,self.index=doc,index
        self.facts=json.loads(native_classification_page_json(doc.library,doc.data,index))
    def get_size(self):return self.facts['size']
    def get_bbox(self):return self.facts['bbox']
    def get_rotation(self):return self.facts['rotation']
    def get_textpage(self):return TextPage(self)
    def get_objects(self,filter=None,max_depth=3):
        return (PageObject(v)for v in self.facts['objects']if filter is None or v[0]in filter)

class Document:
    def __init__(self,data):
        from . import _prepare_pdf,parse_pdf_text
        self.library,self.data=_prepare_pdf(data,None,256*1024*1024)
        count=parse_pdf_text(self.data,start_page=0,end_page=0,allow_partial=True).document['source_page_count']
        self.first,self.last=CLASSIFY_RANGE.get()or(0,count)
        self.count=max(0,min(count,self.last)-self.first)
        self.pages={}
    def __len__(self):return self.count
    def __getitem__(self,index):
        if index not in self.pages:self.pages[index]=Page(self,self.first+index)
        return self.pages[index]

pdfium=SimpleNamespace(PdfDocument=Document)
pdfium_c=SimpleNamespace(
    FPDF_PAGEOBJ_IMAGE=3,FPDF_PAGEOBJ_TEXT=1,
    FPDFText_GetUnicode=lambda p,i:p.chars[i][0],
    FPDFText_IsGenerated=lambda p,i:int(p.chars[i][1]),
    FPDFText_HasUnicodeMapError=lambda p,i:int(p.chars[i][2]),
    FPDFPageObj_GetType=lambda o:o.kind,
)

def difference_without_text(page,scale):
    from PIL import Image
    from ._native import native_classification_difference_bytes
    png=native_classification_difference_bytes(page.doc.library,page.doc.data,page.index,scale)
    with Image.open(BytesIO(png))as image:return image.copy()

def font_reader(data):
    from pypdf import PdfReader
    reader=PdfReader(data if callable(getattr(data,'read',None))else BytesIO(data));selection=CLASSIFY_RANGE.get()
    return SimpleNamespace(reader=reader,pages=reader.pages[selection[0]:selection[1]])if selection else reader
