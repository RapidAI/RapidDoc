"""冻结有原生网格/合并格/扫描表格/缺边/空格/旋转/图片混合的 PDF 事实与裁图。"""
from io import BytesIO
from PIL import Image
from pypdf import PdfReader,PdfWriter
import pypdfium2 as pdfium
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader
from common import ROOT,write_json,digest
from pdf_corpus import extract_facts

def main():
    output=ROOT/'build/migration/native-tables';output.mkdir(parents=True,exist_ok=True)
    data=BytesIO();canvas=Canvas(data,pagesize=(600,500),invariant=1);pages=[]
    def grid(bounds,texts,size=20,missing=False,empty=False):
        left,top,right,bottom=bounds;middle=top+(bottom-top)*.32;split=left+(right-left)*.34
        canvas.setLineWidth(1)
        for y in [top,middle,bottom]:canvas.line(left,500-y,right,500-y)
        canvas.line(left,500-top,left,500-bottom)
        if not missing:canvas.line(right,500-top,right,500-bottom)
        canvas.line(split,500-middle,split,500-bottom)
        if not empty:
            canvas.setFont('Helvetica',size)
            for x,y,text in [(left+12,top+(middle-top)*.65,texts[0]),(left+12,middle+(bottom-middle)*.5,texts[1]),(split+12,middle+(bottom-middle)*.5,texts[2])]:
                canvas.drawString(x,500-y,text)
        return dict(bounds=bounds,x_tracks=[left,split,right],y_tracks=[top,middle,bottom],
            cells=[dict(row=0,column=0,row_span=1,column_span=2,text=texts[0]),
                   dict(row=1,column=0,row_span=1,column_span=1,text=texts[1]),
                   dict(row=1,column=1,row_span=1,column_span=1,text=texts[2])])
    bounds=[60,140,540,360];texts=['Quarter <&> 123','Left 456','Right 789']
    canvas.setFont('Helvetica',24);canvas.drawString(60,440,'Native report')
    first=grid(bounds,texts);pages.append(dict(name='native',tables=[first],reason='closed_native_grid'));canvas.showPage()
    scan=Image.open(ROOT/'build/migration/table-images/wired.png').convert('RGB')
    canvas.drawImage(ImageReader(scan),60,120,width=480,height=240)
    pages.append(dict(name='scan',tables=[dict(bounds=[60,140,540,380])],reason='raster_overlap'));canvas.showPage()
    grid(bounds,texts,missing=True);pages.append(dict(name='missing',tables=[first],reason='incomplete_grid'));canvas.showPage()
    grid(bounds,['','',''],empty=True);pages.append(dict(name='empty',tables=[first],reason='no_native_text'));canvas.showPage()
    grid(bounds,texts);pages.append(dict(name='rotated',tables=[dict(bounds=[140,60,360,540])],reason='rotated_page'));canvas.showPage()
    grid(bounds,texts);canvas.drawImage(ImageReader(Image.new('RGB',(10,10),'red')),400,180,width=40,height=40)
    pages.append(dict(name='mixed-image',tables=[first],reason='raster_overlap'));canvas.showPage()
    two=[grid([40,100,280,300],['T1 header','L1','R1'],14),grid([320,100,560,300],['T2 header','L2','R2'],14)]
    pages.append(dict(name='multiple',tables=two,reason='closed_native_grid'));canvas.showPage()
    left=grid([40,100,280,300],['T3 header','L3','R3'],14)
    canvas.drawImage(ImageReader(scan),320,280,width=240,height=120)
    pages.append(dict(name='native-and-scan',tables=[left,dict(bounds=[320,100,560,220],raster=True)],
        reason='mixed_routes',reasons=['closed_native_grid','raster_overlap']));canvas.showPage();canvas.save()
    reader=PdfReader(BytesIO(data.getvalue()));writer=PdfWriter()
    for index,page in enumerate(reader.pages):
        if index==4:page.rotate(90)
        writer.add_page(page)
    path=output/'tables.pdf';writer.write(path)
    cases=[]
    with pdfium.PdfDocument(path)as document:
        for index,dpi in [(i,72)for i in range(len(pages))]+[(0,137),(6,137)]:
            page=document[index];bitmap=page.render(scale=dpi/72)
            image=bitmap.to_pil().convert('RGB');bitmap.close();page.close()
            target=output/f'page-{index}-{dpi}.png';image.save(target)
            scale=[image.width/(500 if index==4 else 600),image.height/(600 if index==4 else 500)]
            proposals=[]
            for i,table in enumerate(pages[index]['tables']):
                b=table['bounds'];inset=0 if index==1 or table.get('raster')else 2
                proposals.append(dict(cls_id=21,label='table',score=.99,order=i+1,
                    coordinate=[(b[0]+inset)*scale[0],(b[1]+inset)*scale[1],(b[2]-inset)*scale[0],(b[3]-inset)*scale[1]]))
            cases.append(dict(page_index=index,dpi=dpi,width=image.width,height=image.height,input=str(target),layout=proposals,
                rgb_sha256=digest(image.tobytes()),**pages[index]))
            if index==1:
                crop=output/'scan-crop.png';image.crop([60,140,540,380]).save(crop)
            if index==7:
                mixed_crop=output/'mixed-scan-crop.png';image.crop([320,100,560,220]).save(mixed_crop)
    write_json(ROOT/'build/migration/native-tables.json',dict(pdf_input=str(path),facts=extract_facts(path.read_bytes()),cases=cases,scan_crop=str(crop),mixed_scan_crop=str(mixed_crop)))
    print('Exported 8 PDF pages / 10 page-DPI contracts, native table topology, independent facts and raster crops')
if __name__=='__main__':main()
