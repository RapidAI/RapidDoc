"""字符原语补充真实PDF：ToUnicode多码值、代理对、隐藏/透明文字及旋转。"""
from contextlib import closing
from io import BytesIO
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/pdf-character-edges.json';pdf=ROOT/'build/migration/pdf-character-edges.pdf'
    if target.exists()or pdf.exists():raise FileExistsError(target)
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject,NumberObject
    from reportlab.pdfgen.canvas import Canvas
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.text.extract import get_chars
    from rapid_doc._vendor.docvortex_core.document.pdf.native_text_geometry import _restore_pdfium_surrogate_pairs
    buffer=BytesIO();c=Canvas(buffer,pagesize=(320.5,240.25),invariant=1)
    for mode in (0,3,1,2):
        t=c.beginText(20,200);t.setTextRenderMode(mode);t.textOut('Repeated text');c.drawText(t)
    c.setFillAlpha(0);c.drawString(20,170,'Invisible alpha');c.setFillAlpha(1);c.drawString(20,140,'Visible alpha');c.showPage()
    for angle in (0,1,45,90,180,270):
        c.saveState();c.translate(160,120);c.rotate(angle);c.setFont('Helvetica-Oblique',11);c.drawString(0,0,'Angles');c.restoreState()
    c.showPage();c.beginForm('text',0,0,120,60);c.drawString(0,20,'Form letters');c.endForm();c.translate(10,50);c.scale(1.3,.7);c.doForm('text');c.translate(10,70);c.skew(12,8);c.doForm('text');c.save()
    writer=PdfWriter()
    for page in PdfReader(BytesIO(buffer.getvalue())).pages:writer.add_page(page)
    page=writer.add_blank_page(width=320.5,height=240.25)
    cmap=DecodedStreamObject();cmap.set_data(b'/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /Edge def /CMapType 2 def 1 begincodespacerange <00> <FF> endcodespacerange 7 beginbfchar <41> <D83DDE00> <42> <006600690066> <43> <D840DC00> <44> <D800> <45> <DC00> <46> <4E00> <47> <2F00> endbfchar endcmap CMapName currentdict /CMap defineresource pop end end')
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica'),NameObject('/Encoding'):NameObject('/WinAnsiEncoding'),NameObject('/ToUnicode'):writer._add_object(cmap)})
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
    stream=DecodedStreamObject();stream.set_data(b'BT /F1 14 Tf 20 200 Td (ABCDEFG) Tj ET BT /F1 14 Tf 20 160 Td (DD EE) Tj ET')
    page[NameObject('/Contents')]=writer._add_object(stream)
    with pdf.open('wb')as out:writer.write(out)
    cases=[]
    with PDFDocument(pdf.read_bytes())as doc:
        for index in range(len(doc)):
            with doc._open_page(index)as page:
                with closing(page.get_textpage())as textpage:
                    for extended in (False,True):
                        chars=get_chars(textpage,list(page.get_bbox()),page.get_rotation(),include_geometry=extended);raw=encode(chars)
                        restored=_restore_pdfium_surrogate_pairs(chars,textpage,raw_codes={x['char_idx']:x['raw_code']for x in chars})
                        cases.append(dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=digest(pdf.read_bytes()),page_index=index,extended=extended,expected=dict(page_size=list(page.get_size()),page_bbox=list(page.get_bbox()),rotation=page.get_rotation(),raw_char_count=textpage.count_chars(),chars=raw,restored_chars=encode(restored))))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/text/extract.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/_contracts.py','rapid_doc/_vendor/docvortex_core/document/pdf/native_text_geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'actual edge cases,',sum(len(c['expected']['chars'])for c in cases),'chars;',[(hex(c['raw_code']),c['char'])for c in cases[-1]['expected']['chars']])
if __name__=='__main__':main()
