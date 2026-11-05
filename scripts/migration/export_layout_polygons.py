"""Original V3 mask postprocessing: freeze raw masks and exact polygon outputs."""
import json
from io import BytesIO
import numpy as np
from pathlib import Path
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/layout-polygons.json'
    if target.exists():raise FileExistsError(target)
    import cv2
    import onnxruntime as ort
    from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess
    from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType,PP_DOCLAYOUTV2_layout_merge_bboxes_mode
    model=ROOT/json.loads((ROOT/'build/migration/legacy-model.json').read_text(encoding='utf-8'))['model']
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    labels=session.get_modelmeta().custom_metadata_map['character'].splitlines()
    folder=ROOT/'build/migration/layout-polygon-inputs';folder.mkdir(exist_ok=True)
    cases=[];geometries=[]
    def normal(value):
        if isinstance(value,np.ndarray):return normal(value.tolist())
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {k:normal(v)for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [normal(v)for v in value]
        return value
    def add(name,rows,masks,width,height):
        rows=np.asarray(rows,np.float32).reshape(-1,7);masks=np.asarray(masks,dtype='<f4')
        path=folder/(name+'.f32');assert not path.exists();path.write_bytes(masks.tobytes())
        for mode in ['auto','poly','quad','rect']:
            post=PPPostProcess(labels,.3,.5,layout_merge_bboxes_mode=PP_DOCLAYOUTV2_layout_merge_bboxes_mode,layout_unclip_ratio=[1.,1.],scale_size=(800,800))
            result=post(rows.copy(),[width,height],masks.copy(),mode)
            cases.append(dict(name=name+'-'+mode,rows=rows.tolist(),width=width,height=height,threshold=.3,mode=mode,masks=path.relative_to(ROOT).as_posix(),mask_shape=list(masks.shape),mask_sha256=digest(path.read_bytes()),expected=normal(result)))
    # Full source masks are 200x200; vary topology, sharp corners and clipping.
    polys=[[(10,10),(190,10),(190,190),(10,190)],[(10,10),(190,30),(160,180),(20,160)],[(10,10),(190,10),(190,80),(90,80),(90,190),(10,190)],[(100,10),(190,100),(100,190),(10,100)],[(10,10),(190,10),(190,190),(100,90),(10,190)],[(10,10),(190,190),(10,190),(190,10)]]
    for i,points in enumerate(polys):
        mask=np.zeros((200,200),np.uint8);cv2.fillPoly(mask,[np.array(points,np.int32)],1)
        add('synthetic-'+str(i),[[labels.index('text'),.95,0,0,800,800,0]],[mask],800,800)
    for name in ['text','rotated','blank']:
        image=np.asarray(Image.open(ROOT/f'build/migration/detection-images/{name}.png').convert('RGB'))
        tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(image)
        out=session.run(None,dict(image=tensor,im_shape=np.array([[800,800]],np.float32),scale_factor=np.array([[800/image.shape[0],800/image.shape[1]]],np.float32)))
        count=int(out[1][0]);add('model-'+name,out[0][:count],out[2][:count],image.shape[1],image.shape[0])
    from rapid_doc.utils.pdf_reader import page_to_image
    import pypdfium2 as pdfium
    for name in ['demo1','49－长表格-密集型','small_ocr','03','示例1-论文模板']:
        with pdfium.PdfDocument(ROOT/'demo/pdfs'/f'{name}.pdf')as doc:
            page=doc[0]
            try:image,_=page_to_image(page)
            finally:page.close()
        try:rgb=np.asarray(image.convert('RGB'))
        finally:image.close()
        tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(rgb)
        out=session.run(None,dict(image=tensor,im_shape=np.array([[800,800]],np.float32),scale_factor=np.array([[800/rgb.shape[0],800/rgb.shape[1]]],np.float32)))
        count=int(out[1][0]);add('pdf-'+name,out[0][:count],out[2][:count],rgb.shape[1],rgb.shape[0])
    source='rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/post_process.py'
    write_json(target,dict(source=source,source_sha256=digest((ROOT/source).read_bytes()),labels=labels,cases=cases))
    print('Original polygon layout references:',len(cases),'cases')
if __name__=='__main__':main()
