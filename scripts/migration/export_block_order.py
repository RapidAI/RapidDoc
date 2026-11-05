"""冻结页面块序作者合同与独立 CPU OCR/PDF 事实；不导入 Rust 扩展。"""
from io import BytesIO
import math
import pypdfium2 as pdfium
from reportlab.pdfgen.canvas import Canvas
from common import ROOT,write_json,digest
from export_page_blocks import make_ocr
from pdf_corpus import extract_facts


def main():
    def block(i,b,label='text',kind='text'):
        x,y,r,bottom=b
        points=[[x+2,y+2],[max(x+3,r-2),y+2],[max(x+3,r-2),y+18],[x+2,y+18]]
        return dict(layout_index=100+i,kind=kind,region=dict(label=label,coordinate=b),
            lines=[dict(points=points,text=f'block {i}',confidence=None)]if kind=='text'else [],
            text=f'block {i}'if kind!='unsupported'else None)
    cases=[]
    def add(name,boxes,indices,reason='geometry',labels=None,kinds=None):
        blocks=[block(i,b,(labels or {}).get(i,'text'),(kinds or {}).get(i,'text'))for i,b in enumerate(boxes)]
        cases.append(dict(name=name,page=dict(width=600,height=500,coordinate_space='page_pixels',blocks=blocks),indices=indices,reason=reason))
    add('empty',[],[],'already_ordered')
    add('single',[[10,10,200,50]],[0],'already_ordered')
    add('single-column',[[10,150,200,190],[10,20,200,60],[10,80,200,120]],[1,2,0])
    add('diagonal',[[10,100,100,140],[200,10,300,50]],[1,0])
    add('parallel-columns',[[330,20,580,200],[10,10,280,150]],[1,0])
    header_footer=[[0,450,600,480],[330,200,580,250],[0,0,600,30],[20,60,280,120],[330,60,580,120],[20,200,280,250]]
    add('spanning-header-footer',header_footer,[2,3,5,4,1,0],labels={2:'doc_title'})
    add('opaque-table-image',header_footer,[2,3,5,4,1,0],labels={2:'doc_title'},kinds={1:'table',3:'unsupported'})
    add('three-columns',[[420,150,580,200],[20,20,180,70],[220,150,380,200],[420,20,580,70],[20,150,180,200],[220,20,380,70]],
        [1,4,5,2,3,0])
    add('overlap',[[10,10,200,100],[20,50,240,120]],[0,1],'overlapping_regions')
    add('duplicate-region',[[10,10,200,100],[10,10,200,100]],[0,1],'overlapping_regions')
    add('touching',[[10,50,240,100],[10,0,240,50]],[0,1],'ambiguous_geometry')
    add('narrow-gap',[[10,21,240,41],[10,0,240,20]],[0,1],'ambiguous_geometry')
    add('zero-region',[[10,10,10,30]],[0],'invalid_region_geometry')
    add('outside-region',[[-1,10,200,40]],[0],'invalid_region_geometry')
    add('vertical-label',[[10,80,200,120],[10,10,200,50]],[0,1],'unsupported_direction',labels={0:'vertical_text'})
    add('middle-spanner',[[0,140,200,190],[300,0,500,50],[0,80,500,110],[0,0,200,50],[300,140,500,190]],
        [0,1,2,3,4],'ambiguous_geometry')

    output=ROOT/'build/migration/block-order';output.mkdir(parents=True,exist_ok=True)
    texts=['Reading order heading','Left upper paragraph','Left lower paragraph','Right upper text','Right lower text','Reading order footer']
    boxes=[[30,20,570,65],[30,90,280,150],[30,190,280,250],[325,90,570,150],[325,190,570,250],[30,430,570,480]]
    buf=BytesIO();canvas=Canvas(buf,pagesize=(600,500),invariant=1);canvas.setFont('Helvetica',24)
    for x,y,text in [(40,455,texts[0]),(40,380,texts[1]),(40,280,texts[2]),(335,380,texts[3]),(335,280,texts[4]),(40,40,texts[5])]:
        canvas.drawString(x,y,text)
    canvas.save();data=buf.getvalue();pdf=output/'columns.pdf';pdf.write_bytes(data)
    with pdfium.PdfDocument(data)as doc:
        p=doc[0];bitmap=p.render(scale=1);image=bitmap.to_pil().convert('RGB');bitmap.close();p.close()
    png=output/'columns.png';image.save(png);ocr=make_ocr();regions=[]
    for i,b in enumerate(boxes):
        bounds=[math.floor(b[0]/5)*5,math.floor(b[1]/5)*5,math.ceil(b[2]/5)*5,math.ceil(b[3]/5)*5]
        lines=ocr(image.crop(bounds));assert len(lines)==1,(i,lines)
        regions.append(dict(bounds=b,lines=lines,text=texts[i],label='doc_title'if i==0 else 'text'))
    permutation=[5,4,0,1,3,2]
    write_json(ROOT/'build/migration/block-order.json',dict(cases=cases,runtime=dict(
        pdf_input=str(pdf),pdf_sha256=digest(data),png_input=str(png),png_sha256=digest(png.read_bytes()),
        width=600,height=500,facts=extract_facts(data),regions=regions,permutation=permutation,indices=[2,3,5,4,1,0])))
    print(f'Exported {len(cases)} block-order contracts and independent 6-region OCR/native PDF fixture')


if __name__=='__main__':main()
