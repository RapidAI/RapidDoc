"""Freeze original page_to_image long-edge scaling and RGB pixels; never overwrite."""
from pathlib import Path
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/api-render-caps.json'
    if target.exists():raise FileExistsError(target)
    import pypdfium2 as pdfium
    from rapid_doc.utils.pdf_reader import page_to_image
    cases=[]
    for filename,page_index in [('small_ocr.pdf',0),('small_ocr.pdf',1),('demo1.pdf',0),('示例1-论文模板.pdf',0)]:
        source=ROOT/'demo/pdfs'/filename
        with pdfium.PdfDocument(source)as document:
            page=document[page_index]
            try:image,scale=page_to_image(page,dpi=200)
            finally:page.close()
        with image:
            rgb=image.convert('RGB')
            try:cases.append(dict(source=source.relative_to(ROOT).as_posix(),source_sha256=digest(source.read_bytes()),page_index=page_index,dpi=200,scale=scale,width=rgb.width,height=rgb.height,pixels_sha256=digest(rgb.tobytes())))
            finally:rgb.close()
    write_json(target,dict(sources={'rapid_doc/utils/pdf_reader.py':digest((ROOT/'rapid_doc/utils/pdf_reader.py').read_bytes())},cases=cases))
    print('Original render cap reference:',len(cases),'pages')

if __name__=='__main__':main()
