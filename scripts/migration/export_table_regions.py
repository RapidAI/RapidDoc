"""UNET 角度/旋转灰度像素、逆变换区域与源索引逻辑格合同。"""
import cv2,json,numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.table.rapid_table_self.table_structure.unet.main import TSRUnetStructurer
from rapid_doc.model.table.rapid_table_self.table_structure.unet.table_recover import TableRecover
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_line_rec import draw_lines
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import sorted_ocr_boxes,box_4_2_poly_to_box_4_1
from common import ROOT,write_json,digest
from export_unet_lines import capture as capture_lines

def capture(pre,lines,width,height,rotated_fix):
    image=draw_lines(np.zeros((height,width),np.uint8),np.asarray(lines,np.float32).reshape(-1,4),255,2)
    angle=pre.cal_rotate_angle(image)
    if rotated_fix and abs(angle)>.3:
        rotated=pre.cal_region_boxes(pre.rotate_image(image,angle));polygons=pre.unrotate_polygons(rotated,angle,image.shape)
    else:rotated=pre.cal_region_boxes(image);polygons=rotated.copy()
    polygons=polygons.reshape(-1,4,2);rotated=rotated.reshape(-1,4,2)
    polygons[:,[1,3]]=polygons[:,[3,1]];rotated[:,[1,3]]=rotated[:,[3,1]]
    _,indices=sorted_ocr_boxes([box_4_2_poly_to_box_4_1(p)for p in rotated],threhold=.4)
    polygons=polygons[indices];rotated=rotated[indices]
    result=dict(rotation_degrees=angle,polygons=polygons.tolist(),rotated_polygons=rotated.tolist(),logic_points=None)
    if len(rotated):
        recover=TableRecover();points=rotated.astype(np.float64)
        rows=recover.get_rows(points,10);_,widths,_=recover.get_benchmark_cols(rows,points,15);heights,_=recover.get_benchmark_rows(rows,points)
        _,logic=recover(points);result['logic_points']=logic.tolist()
        if min(widths)<=0 or min(heights)<=0:result['expected_error']='invalid_input'
        if any(r0>r1 or c0>c1 for r0,r1,c0,c1 in logic):result['expected_error']='invalid_input'
    return result

def main():
    pre=TSRUnetStructurer.__new__(TSRUnetStructurer);directory=ROOT/'build/migration/table-regions';directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(553);rotations=[]
    for width,height in [(7,5),(127,95),(128,96),(480,240)]:
        gray=rng.integers(0,256,(height,width),np.uint8);source=directory/f'{width}x{height}.gray';source.write_bytes(gray.tobytes())
        for angle in [0.,.3,-.3,1.,-3.,15.,44.,89.,180.]:
            expected=pre.rotate_image(gray,angle);path=directory/f'{width}x{height}-{angle}.gray';path.write_bytes(expected.tobytes())
            rotations.append(dict(width=width,height=height,input=str(source),angle=angle,expected=str(path),sha256=digest(expected.tobytes())))
    drawings=json.loads((ROOT/'build/migration/table-draw.json').read_text())
    angles=[]
    for case in drawings['cases']:
        image=np.frombuffer(open(case['expected'],'rb').read(),np.uint8).reshape(case['height'],case['width'])
        angles.append(dict(id=case['id'],input=case['expected'],width=case['width'],height=case['height'],angle=pre.cal_rotate_angle(image)))
    cases=[];base=[]
    for y in [20,90,160,230]:base.append([20,y,460,y])
    for x in [20,160,310,460]:base.append([x,20,x,230])
    for angle in [0.,1.,-3.,10.,-10.,30.]:
        matrix=cv2.getRotationMatrix2D((240,128),angle,1)
        lines=cv2.transform(np.asarray(base,np.float32).reshape(-1,2,2),matrix).reshape(-1,4).tolist()
        for fix in [False,True]:cases.append(dict(id=f'grid-{angle}-{fix}',lines=lines,width=480,height=256,rotated_fix=fix,**capture(pre,lines,480,256,fix)))
    fixture=json.loads((ROOT/'build/migration/unet-lines.json').read_text());models=[]
    for case in fixture['cases']:
        if not case['morph_close'] or not case['id'].endswith('png-True'):continue
        lines=case['rows']+case['columns'];result=capture(pre,lines,case['source_width'],case['source_height'],True)
        models.append(dict(input=case['id'][:-5],**result))
    unet=json.loads((ROOT/'build/migration/unet-baseline.json').read_text())
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/unet['model']),sess_options=options,providers=['CPUExecutionProvider'])
    pre.inp_height=pre.inp_width=1024;pre.mean=np.array([123.675,116.28,103.53],np.float32);pre.std=np.array([58.395,57.12,57.375],np.float32)
    wired=Image.open(unet['png_cases'][0]['input']).convert('RGB')
    for angle in [-3,3]:
        image=wired.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor='white')
        path=directory/f'wired-rotated-{angle}.png';image.save(path)
        bgr=np.asarray(image)[:,:,::-1].copy();mask=session.run(None,{'input':pre.preprocess(bgr)['img']})[0][0,0].astype(np.uint8)
        lines=capture_lines(str(path),mask,image.width,image.height)[-1]
        models.append(dict(input=str(path),**capture(pre,lines['rows']+lines['columns'],image.width,image.height,True)))
    write_json(ROOT/'build/migration/table-regions.json',dict(rotations=rotations,angles=angles,cases=cases,png_cases=models,model=unet['model'],sha256=unet['sha256']))
    print(f'Exported {len(rotations)} nearest rotation rasters, {len(cases)} grid region contracts and {len(models)} UNET PNG cases')

if __name__=='__main__':main()
