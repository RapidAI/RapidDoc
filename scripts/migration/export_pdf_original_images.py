"""冻结实际旧提图/资产函数；明确记录 get_pos→get_bounds 兼容别名和随机UUID投影。"""
import copy,json,math
from io import BytesIO
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/pdf-original-images.json'
    if target.exists():raise FileExistsError(target)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader,PdfWriter
    from rapid_doc.utils.pdf_image_tools import get_ori_image
    from rapid_doc.backend.pipeline.model_json_to_middle_json import page_model_info_to_page_info
    folder=ROOT/'build/migration/pdf-original-images';folder.mkdir(parents=True,exist_ok=True)
    stream=BytesIO();c=Canvas(stream,pagesize=(320,360),invariant=1)
    def make(mode,w,h):
        channels=len(mode)if mode!='L'else 1
        data=bytes((x*17+y*31+k*67)%256 for y in range(h)for x in range(w)for k in range(channels))
        return Image.frombytes(mode,(w,h),data)
    rgb=make('RGB',17,13);gray=make('L',11,19);rgba=make('RGBA',19,11)
    # Plain/gray/soft mask/color key, same-bbox overlap and tiny filter boundary.
    for i,img in enumerate([rgb,gray,rgba]):
        c.drawImage(ImageReader(img),10+i*75,280,60,50,mask='auto')
    c.drawImage(ImageReader(rgb),10,220,60,40,mask=[0,90,0,255,0,255])
    for size in [4.999,5.,5.001]:c.drawImage(ImageReader(rgb),80,220-size*10,size,size)
    c.drawImage(ImageReader(rgb),180,220,30,20);c.drawImage(ImageReader(rgba),180,220,30,20,mask='auto')
    c.showPage()
    for j,(angle,sx,sy)in enumerate([(0,1,1),(30,1,1),(90,1,1),(180,1,1),(270,1,1),(0,-1,1),(0,1,-1),(45,2,.5)]):
        c.saveState();c.translate(50+(j%4)*70,230-(j//4)*100);c.rotate(angle);c.scale(sx,sy);c.drawImage(ImageReader(rgba if j%2 else rgb),0,0,35,25,mask='auto');c.restoreState()
    c.showPage()
    c.beginForm('inner',0,0,100,100);c.drawImage(ImageReader(rgba),10,15,40,30,mask='auto');c.endForm()
    c.beginForm('outer',0,0,150,150);c.saveState();c.translate(20,30);c.scale(1.5,.75);c.doForm('inner');c.restoreState();c.endForm()
    c.saveState();c.translate(50,60);c.scale(2,2);c.doForm('outer');c.restoreState()
    c.saveState();c.translate(100,100);c.transform(1,.2,.3,1,0,0);c.drawImage(ImageReader(rgb),0,0,50,30);c.restoreState();c.showPage();c.save()
    paths=[]
    for rotation in [0,90,180,270]:
        writer=PdfWriter()
        for p in PdfReader(BytesIO(stream.getvalue())).pages:
            page=writer.add_page(p);page.cropbox.lower_left=(5.25,10.5);page.cropbox.upper_right=(315.75,350.25)
            if rotation:page.rotate(rotation)
        path=folder/f'images-{rotation}.pdf';writer.write(path);paths.append(path)
    paths+=list((ROOT/'tests/migration/inputs').glob('pdf_*.pdf'))
    cases=[];pages=[];errors=[]
    class Writer:
        _parent_dir='images'
        def __init__(self):self.calls=[]
        def write(self,name,data):
            with Image.open(BytesIO(data))as im:
                im.load();self.calls.append(dict(name=name,mode=im.mode,width=im.width,height=im.height,pixels_sha256=digest(im.tobytes())))
    has_pos=hasattr(pdfium.PdfImage,'get_pos')
    assert not has_pos,'Update baseline profile: get_pos is already present'
    for path in paths:
        with pdfium.PdfDocument(path)as doc:
            for index in range(len(doc)):
                page=doc[index]
                try:
                    # Preserve evidence of the installed old API failure, do not call it parity.
                    before=get_ori_image(page)
                    assert not before
                    for depth,render,scale in [(15,True,True),(15,True,False),(15,False,True),(1,True,True),(2,True,True),(0,True,True)]:
                        pdfium.PdfImage.get_pos=pdfium.PdfImage.get_bounds
                        try:images=get_ori_image(page,max_depth=depth,render=render,scale_to_original=scale)
                        finally:del pdfium.PdfImage.get_pos
                        expected=[]
                        for i,img in enumerate(images):
                            im=img['pil_image'];expected.append(dict(uuid=f'pdf-image-{index}-{i}',raw_uuid=img['uuid'],bbox=img['bbox'],mode=im.mode,width=im.width,height=im.height,pixels_sha256=digest(im.tobytes())))
                        cases.append(dict(source=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes()),page_index=index,max_depth=depth,render=render,scale_to_original=scale,old_unadapted_count=0,expected=expected))
                        if depth==15 and render and scale:
                            bitmap=page.render(scale=1.)
                            try:pil=bitmap.to_pil().convert('RGB')
                            finally:bitmap.close()
                            dets=[]
                            for i,img in enumerate(images[:12]):
                                x,y,r,b=img['bbox'];dets.append(dict(category_id=3,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=i))
                            # Use actual legacy writer and model pipeline with original replacement enabled.
                            model=dict(layout_dets=dets);w=Writer();size=page.get_size()
                            expected_page=page_model_info_to_page_info(copy.deepcopy(model),dict(scale=1.,img_pil=pil),dict(size=size,ori_image_list=images),w,index,ocr_enable=True,
                                image_config=dict(extract_original_image=True,extract_original_image_iou_thresh=.9))
                            pages.append(dict(source=path.relative_to(ROOT).as_posix(),page_index=index,model=model,expected=expected_page,calls=w.calls))
                            pil.close()
                        for img in images:img['pil_image'].close()
                finally:page.close()
    sources=['rapid_doc/utils/pdf_image_tools.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py']
    write_json(target,dict(cases=cases,pages=pages,sources={p:digest((ROOT/p).read_bytes())for p in sources},
        profile='实际旧函数；get_pos兼容别名到同版本get_bounds后冻结。未修改其遍历/矩阵/像素/匹配规则。未适配旧函数全为空另记；随机uuid保留raw_uuid并投影稳定ID，非完全无差异声明。'))
    print(f'Frozen {len(cases)} image cases/{sum(len(c["expected"])for c in cases)} images, {len(pages)} old complete pages')

if __name__=='__main__':main()
