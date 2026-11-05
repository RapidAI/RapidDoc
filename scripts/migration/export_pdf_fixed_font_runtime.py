"""Actual unembedded CJK PDF geometry and pixels under the old fixed-font guard."""
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/pdf-fixed-font-runtime.json';pdf=ROOT/'build/migration/fixed-cjk-runtime.pdf'
    if target.exists()or pdf.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    fonts=[('STSong-Light','中文表格测试'),('MSung-Light','繁體中文表格'),('HeiseiMin-W3','日本語表の試験'),('HYSMyeongJo-Medium','한글 표 시험')]
    c=Canvas(str(pdf),pagesize=(400,300),pageCompression=0);labels=[]
    for font,text in fonts:
        pdfmetrics.registerFont(UnicodeCIDFont(font))
        for rotate in [0,90]:
            c.setFont(font,12)
            if rotate:c.translate(100,100);c.rotate(90)
            c.drawString(40,200 if rotate==0 else 0,text);c.setFont('Helvetica',10);c.drawString(40,180 if rotate==0 else -20,'Native CJK font 123');c.showPage();labels.append(font+'-'+str(rotate))
    c.save();data=pdf.read_bytes();cases=[]
    from rapid_doc._vendor.docvortex_core.document.pdf import PDFDocument
    from rapid_doc._vendor.docvortex_core.document.pdf.pdfium import pdfium_guard
    from rapid_doc._vendor.docvortex_core.document.pdf.text.extract import get_chars
    from rapid_doc._vendor.docvortex_core.document.pdf.native_text_geometry import _restore_pdfium_surrogate_pairs
    from contextlib import closing
    with PDFDocument(data)as doc:
        for i,label in enumerate(labels):
            with doc._open_page(i)as page:
                with closing(page.get_textpage())as text:
                    chars=get_chars(text,list(page.get_bbox()),page.get_rotation(),include_geometry=True)
                    restored=_restore_pdfium_surrogate_pairs(chars,text,raw_codes={x['char_idx']:x['raw_code']for x in chars})
                    expected=dict(page_size=list(page.get_size()),page_bbox=list(page.get_bbox()),rotation=page.get_rotation(),raw_char_count=text.count_chars(),chars=encode(chars),restored_chars=encode(restored))
                renders=[]
                with pdfium_guard():
                    for dpi in [72.,144.]:
                        bitmap=page.render(scale=dpi/72.)
                        try:image=bitmap.to_pil().convert('RGBA');renders.append(dict(dpi=dpi,width=image.width,height=image.height,rgba_sha256=digest(image.tobytes())))
                        finally:bitmap.close()
            cases.append(dict(pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(data),page_index=i,label=label,expected=expected,renders=renders))
    refs=['rapid_doc/_vendor/docvortex_core/'+p for p in ['document/pdf/font_runtime.py','document/pdf/pdfium.py','document/pdf/text/extract.py','resources/fonts/DroidSansFallbackFull.ttf']]
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='8 actual unembedded CJK CID pages, four language families and two orientations; old guarded raw geometry and 16 RGBA pixel references.',cases=cases));print('Frozen CJK actual pages',len(cases),'pixel references',sum(len(c['renders'])for c in cases))
if __name__=='__main__':main()
