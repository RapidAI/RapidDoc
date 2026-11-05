"""实际PDF重复绘制/隐藏副本/同字形汉字映射补充，保留旧输出。"""
from io import BytesIO
from contextlib import closing
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/pdf-native-dedup-extra.json';pdf=ROOT/'build/migration/pdf-native-dedup-extra.pdf'
    if target.exists()or pdf.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from pypdf import PdfReader,PdfWriter
    from pypdf.generic import DictionaryObject,NameObject,DecodedStreamObject
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.text.extract import get_chars
    from rapid_doc._vendor.docvortex_core.document.pdf.native_text_geometry import _extract_page_text_geometry,_restore_pdfium_surrogate_pairs
    buffer=BytesIO();c=Canvas(buffer,pagesize=(320,240),invariant=1)
    c.setFont('Courier',12)
    for x,y in [(20,190),(21,191),(22,192)]:c.drawString(x,y,'Offset copies ABCDEFGH')
    c.showPage();c.setFont('Courier',12);c.drawString(20,190,'Hidden copies ABCDEFGH')
    t=c.beginText(20,190);t.setFont('Courier-Bold',12);t.setTextRenderMode(3);t.textOut('Hidden copies ABCDEFGH');c.drawText(t);c.save()
    writer=PdfWriter()
    for page in PdfReader(BytesIO(buffer.getvalue())).pages:writer.add_page(page)
    page=writer.add_blank_page(width=320,height=240)
    cmap=DecodedStreamObject();cmap.set_data(b'/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def /CMapName /D def /CMapType 2 def 1 begincodespacerange <00> <FF> endcodespacerange 2 beginbfchar <41> <4E002F00> <42> <00660069> endbfchar endcmap CMapName currentdict /CMap defineresource pop end end')
    font=DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica'),NameObject('/ToUnicode'):writer._add_object(cmap)})
    page[NameObject('/Resources')]=DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
    stream=DecodedStreamObject();stream.set_data(b'BT /F1 12 Tf 20 190 Td (AB) Tj ET');page[NameObject('/Contents')]=writer._add_object(stream)
    with pdf.open('wb')as out:writer.write(out)
    cases=[];raw_cases=[]
    with PDFDocument(pdf.read_bytes())as doc:
        for index in range(len(doc)):
            with doc._open_page(index)as page:
                for extended in (False,True):
                    with closing(page.get_textpage())as textpage:
                        raw=get_chars(textpage,list(page.get_bbox()),page.get_rotation(),include_geometry=extended)
                        restored=_restore_pdfium_surrogate_pairs(raw,textpage,raw_codes={x['char_idx']:x['raw_code']for x in raw})
                        count=textpage.count_chars()
                    g=_extract_page_text_geometry(page,include_extended_geometry=extended)
                    cases.append(dict(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=digest(pdf.read_bytes()),page_index=index,extended=extended,expected=dict(page_size=list(page.get_size()),page_bbox=list(page.get_bbox()),rotation=page.get_rotation(),raw_char_count=count,chars=encode(g.chars),tight_bboxes=encode(g.tight_bboxes),loose_bboxes=encode(g.loose_bboxes),origins=encode(g.origins))))
                    raw_cases.append(dict(label=f'actual:{index}:{extended}',chars=encode(restored),expected=encode(g.chars)))
    refs=['rapid_doc/_vendor/docvortex_core/document/pdf/native_text_geometry.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/extract.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/dedup.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,raw_cases=raw_cases));print('Actual retained:',[(c['page_index'],c['extended'],c['expected']['raw_char_count'],len(c['expected']['chars']))for c in cases])
if __name__=='__main__':main()
