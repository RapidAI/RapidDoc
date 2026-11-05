"""Actual original file collection and full native analysis; no stage substitutions."""
import copy,json
from dataclasses import asdict
from io import BytesIO
from common import ROOT,write_json,digest
from export_native_late_image_text import project
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-pdf-document.json';folder=ROOT/'build/migration/native-pdf-document';folder.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    from reportlab.pdfgen.canvas import Canvas
    from PIL import Image
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,NumberObject,ArrayObject,FloatObject,DecodedStreamObject
    buf=BytesIO();c=Canvas(buf,pagesize=(400,600),pageCompression=0,invariant=1)
    for rotation in [0,90,180,270]:
        c.setFont('Helvetica',12);c.drawString(30,550,'Original signature widgets and linked text');c.linkURL('https://example.test/original',(30,545,260,565),relative=0);c.showPage()
    c.save();writer=PdfWriter();reader=PdfReader(BytesIO(buf.getvalue()));fields=ArrayObject()
    for index,page in enumerate(reader.pages):
        writer.add_page(page);page=writer.pages[-1];page[NameObject('/Rotate')]=NumberObject(index*90);annots=page.get('/Annots',ArrayObject())
        for n,mode in enumerate(['visible','hidden','no-appearance','text-field','inherited','clipped']):
            b=[30.+n*55,180.,75.+n*55,220.];b=[-20.,580.,30.,630.]if mode=='clipped'else b
            widget=DictionaryObject({NameObject('/Type'):NameObject('/Annot'),NameObject('/Subtype'):NameObject('/Widget'),NameObject('/Rect'):ArrayObject([FloatObject(v)for v in b]),NameObject('/FT'):NameObject('/Tx'if mode=='text-field'else'/Sig'),NameObject('/T'):__import__('pypdf').generic.TextStringObject(f'field-{index}-{n}'),NameObject('/F'):NumberObject(2 if mode=='hidden'else 0)})
            if mode!='no-appearance':
                stream=DecodedStreamObject();stream.set_data(b'0 0 1 rg 0 0 45 40 re f');stream.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Form'),NameObject('/BBox'):ArrayObject([NumberObject(v)for v in [0,0,45,40]])});widget[NameObject('/AP')]=DictionaryObject({NameObject('/N'):writer._add_object(stream)})
            if mode=='inherited':
                parent=DictionaryObject({NameObject('/FT'):NameObject('/Sig'),NameObject('/T'):__import__('pypdf').generic.TextStringObject(f'parent-{index}'),NameObject('/Kids'):ArrayObject()});parent_ref=writer._add_object(parent);widget.pop('/FT');widget[NameObject('/Parent')]=parent_ref
            ref=writer._add_object(widget);annots.append(ref)
            if mode=='inherited':parent['/Kids'].append(ref);fields.append(parent_ref)
            else:fields.append(ref)
        page[NameObject('/Annots')]=annots
    writer._root_object[NameObject('/AcroForm')]=writer._add_object(DictionaryObject({NameObject('/Fields'):fields,NameObject('/SigFlags'):NumberObject(3)}));path=folder/'signatures.pdf'
    if path.exists():raise FileExistsError(path)
    with path.open('wb')as f:writer.write(f)
    textured=Image.new('RGB',(32,32));textured.putdata([((x*43+y*17)%256,(x*31+y*67)%256,(x*7+y*101)%256)for y in range(32)for x in range(32)]);img=BytesIO();textured.save(img,format='PNG');textured.close()
    from reportlab.lib.utils import ImageReader
    stream=BytesIO();canvas=Canvas(stream,pagesize=(400,600),pageCompression=0,invariant=1)
    for i in range(3):
        canvas.setFont('Helvetica',10)
        for j in range(5):canvas.drawString(30,560-j*18,'Original ordinary body words, page '+str(i)+' row '+str(j))
        canvas.drawImage(ImageReader(BytesIO(img.getvalue())),220,80,width=150,height=180);canvas.setStrokeColorRGB(.1,.1,.1);canvas.line(30,400,370,400);canvas.showPage()
    canvas.save();raster=folder/'raster.pdf'
    if raster.exists():raise FileExistsError(raster)
    raster.write_bytes(stream.getvalue())
    pdfs=[ROOT/'build/migration/native-reference-runtime.pdf',ROOT/'build/migration/form-runtime.pdf',path,raster]
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _collect_document_sources,_analyze_native_document
    cases=[]
    for path in pdfs:
        data=path.read_bytes()
        with PDFDocument(data)as doc:
            sources=_collect_document_sources(doc);ps=[]
            for i,(s,g)in enumerate(zip(sources.page_sources,sources.page_text_geometries)):
                source=dict(page_size=s.page_size,page_index=s.page_index,lines=[project(l)for l in s.lines],chars=s.chars,drawing_lines=[asdict(r)for r in s.drawing_lines],signature_bboxes=s.signature_bboxes,form_bboxes=s.form_bboxes,path_infos=[asdict(p)for p in s.path_infos],form_infos=[asdict(p)for p in s.form_infos],image_bboxes=s.image_bboxes)
                geometry=dict(chars=g.chars,tight_bboxes=[dict(char_idx=k,bbox=v)for k,v in sorted(g.tight_bboxes.items())],loose_bboxes=[dict(char_idx=k,bbox=v)for k,v in sorted(g.loose_bboxes.items())],origins=[dict(char_idx=k,origin=v)for k,v in sorted(g.origins.items())])
                ps.append(dict(source=source,geometry=geometry,style_lines=[asdict(l)for l in sources.page_style_lines[i]],link_lines=[asdict(l)for l in sources.page_link_lines[i]],spacing_lines=[asdict(l)for l in sources.page_spacing_lines[i]]))
            collected=clean(dict(pages=ps,source_page_count=len(ps),page_sizes=sources.page_sizes))
        with PDFDocument(data)as doc:pages=_analyze_native_document(doc)
        cases.append(dict(label=path.stem,pdf=path.relative_to(ROOT).as_posix(),pdf_sha256=digest(data),collected=collected,expected=clean(dict(pages=pages,source_page_count=len(pages),page_sizes=sources.page_sizes))))
    write_json(target,dict(cases=cases));print(len(cases),'complete original PDF file cases;',sum(c['collected']['source_page_count']for c in cases),'pages;',sum(len(p['source']['signature_bboxes'])for c in cases for p in c['collected']['pages']),'visible signatures')
if __name__=='__main__':main()
