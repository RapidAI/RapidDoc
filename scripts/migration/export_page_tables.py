"""冻结整页矩形表格布局与旧 5px 网格裁图，不依赖新 Rust 模块生成参考。"""
import json
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess
from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType,PP_DOCLAYOUTV2_layout_merge_bboxes_mode
from rapid_doc.utils.bbox_utils import normalize_to_int_bbox
from rapid_doc.utils.pdf_image_tools import get_crop_np_img
from common import ROOT,write_json,digest

def plan(image,regions,grid=5):
    values=[]
    for index,region in enumerate(regions):
        if region['label']!='table':continue
        b=normalize_to_int_bbox([v/grid for v in region['coordinate']])
        crop,_=get_crop_np_img(b,np.asarray(image),scale=grid,return_list=True)
        bounds=[max(0,b[0]*grid),max(0,b[1]*grid),min(image.width,b[2]*grid),min(image.height,b[3]*grid)]
        values.append(dict(layout_index=index,bounds=bounds,width=crop.shape[1],height=crop.shape[0],rgb_sha256=digest(crop.tobytes())))
    return values
def main():
    output=ROOT/'build/migration/page-tables';output.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    pages=[]
    for name,tables in [('single',[('wired',60,170)]),('multiple',[('wired',60,150),('wired',60,500)]),('blank',[])]:
        image=Image.new('RGB',(600,900),'white')
        if tables:
            draw=ImageDraw.Draw(image);draw.text((55,45),'Quarterly results',font=font,fill='black')
            draw.text((55,90),'Revenue and expense summary',font=font,fill='black')
        for kind,x,y in tables:image.paste(Image.open(ROOT/f'build/migration/table-images/{kind}.png').convert('RGB'),(x,y))
        path=output/f'{name}.png';image.save(path);pages.append((path,image))
    model=ROOT/'rapid_doc/model/layout/rapid_layout_self/models/pp_doclayoutv3.onnx'
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    labels=session.get_modelmeta().custom_metadata_map['character'].splitlines()
    post=PPPostProcess(labels,.5,layout_merge_bboxes_mode=PP_DOCLAYOUTV2_layout_merge_bboxes_mode,layout_unclip_ratio=[1.,1.],scale_size=(800,800))
    cases=[]
    for path,image in pages:
        tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(np.asarray(image))
        raw=session.run(None,dict(image=tensor,im_shape=np.array([[800,800]],np.float32),scale_factor=np.array([[800/image.height,800/image.width]],np.float32)))
        result=post(raw[0][:int(raw[1][0])].copy(),[image.width,image.height],layout_shape_mode='rect')
        regions=result if isinstance(result,list)else result.tolist()
        crops=plan(image,regions)
        for i,crop in enumerate(crops):
            target=output/f'{path.stem}-crop-{i}.png';image.crop(crop['bounds']).save(target);crop['input']=str(target)
        cases.append(dict(input=str(path),width=image.width,height=image.height,rgb_sha256=digest(image.tobytes()),layout=regions,tables=crops))
        print(path.stem,[(r['label'],r['coordinate'])for r in regions],flush=True)
    # 彩色随机图的裁边、网格与顺序合同，与原 normalize/get_crop_np_img 对照。
    rng=np.random.default_rng(134);image=Image.fromarray(rng.integers(0,256,(37,53,3),dtype=np.uint8));path=output/'colors.png';image.save(path)
    crop_cases=[]
    for grid in [1,5,8]:
        for index,bounds in enumerate([[.3,2.7,20.1,30.2],[0,0,53,37],[13,17,25,35],[50.1,30.3,52.9,36.9]]):
            region=dict(cls_id=0,label='table',score=.9,coordinate=bounds,order=2)
            crop_cases.append(dict(id=f'{grid}-{index}',grid=grid,regions=[region],tables=plan(image,[region],grid)))
    mixed=[dict(cls_id=0,label='text',score=.9,coordinate=[0,0,10,10],order=1),
           dict(cls_id=0,label='table',score=.9,coordinate=[13,17,25,35],order=9),
           dict(cls_id=0,label='table',score=.9,coordinate=[.3,2.7,20.1,30.2],order=2)]
    crop_cases.append(dict(id='mixed-preserve-order',grid=5,regions=mixed,tables=plan(image,mixed)))
    assert any(c['tables']for c in cases),'real layout did not detect any table'
    pdf=output/'pages.pdf';document=canvas.Canvas(str(pdf),pagesize=(600,900),invariant=1)
    for index in [2,0]:
        document.drawImage(ImageReader(pages[index][1]),0,0,width=600,height=900);document.showPage()
    document.save()
    write_json(ROOT/'build/migration/page-tables.json',dict(cases=cases,crop_input=str(path),crop_cases=crop_cases,pdf_input=str(pdf),model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes())))
    print(f'Exported {len(cases)} real page layouts and {len(crop_cases)} crop contracts')
if __name__=='__main__':main()
