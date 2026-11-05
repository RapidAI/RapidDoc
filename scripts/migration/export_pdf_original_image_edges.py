"""构造提图损坏/资源边界，坏图用实际旧函数+get_pos别名对照，限额仅作者合同。"""
from common import ROOT,digest,write_json

def main():
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject,DecodedStreamObject,NameObject,NumberObject
    import pypdfium2 as pdfium
    from rapid_doc.utils.pdf_image_tools import get_ori_image
    folder=ROOT/'build/migration/pdf-original-image-edges';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'manifest.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for name,w,h,n,bad in [('broken-jpeg',17,13,1,True),('huge-image',100000,100000,1,False),('too-many-images',1,1,257,False)]:
        writer=PdfWriter();page=writer.add_blank_page(width=100,height=100)
        image=DecodedStreamObject();image.set_data(b'broken'if bad else b'\0\0\0')
        image.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Image'),NameObject('/Width'):NumberObject(w),NameObject('/Height'):NumberObject(h),NameObject('/ColorSpace'):NameObject('/DeviceRGB'),NameObject('/BitsPerComponent'):NumberObject(8)})
        if bad:image[NameObject('/Filter')]=NameObject('/DCTDecode')
        page[NameObject('/Resources')]=DictionaryObject({NameObject('/XObject'):DictionaryObject({NameObject('/I'):writer._add_object(image)})})
        content=DecodedStreamObject();content.set_data(('q 10 0 0 10 5 5 cm /I Do Q\n'*n).encode());page[NameObject('/Contents')]=writer._add_object(content)
        path=folder/f'{name}.pdf';writer.write(path)
        if bad:
            pdfium.PdfImage.get_pos=pdfium.PdfImage.get_bounds
            try:
                with pdfium.PdfDocument(path)as doc:
                    page=doc[0]
                    try:assert get_ori_image(page,render=False)==[]
                    finally:page.close()
            finally:del pdfium.PdfImage.get_pos
        cases.append(dict(name=name,source=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes()),render=not bad,code=None if bad else 'resource_limit'))
    write_json(target,dict(cases=cases))
    print('Created bad-image diagnostic and two pre-allocation resource guards')

if __name__=='__main__':main()
