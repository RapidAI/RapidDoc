"""实际旧 native 页面→真实 CPU 后置 OCR→编号→段落→输出；同时覆盖普通资产。"""
import copy,importlib,json,types
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-native-text-chain.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');sp=importlib.import_module('rapid_doc.utils.span_pre_proc')
    modelmod=importlib.import_module('rapid_doc.model.ocr.rapid_ocr');render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    model=ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx';options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']);rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    decoder=CTCLabelDecode(character=session.get_modelmeta().custom_metadata_map['character'].splitlines())
    worker=modelmod.RapidOcrModel.__new__(modelmod.RapidOcrModel)
    worker.text_recognizer=types.SimpleNamespace(rec_batch_num=6,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=lambda tensor:session.run(None,{session.get_inputs()[0].name:tensor})[0],postprocess_op=decoder)
    class Manager:
        def get_atom_model(self,**kwargs):return worker
    original_manager=old.AtomModelSingleton;original_contrast=sp.calculate_contrast;cases=[]
    base=json.loads((ROOT/'build/migration/legacy-native-text.json').read_text());selected=[copy.deepcopy(base['pages'][i])for i in [0,1,2,3,6,9,18,27,28,29,30,31]]
    def det(kind,bbox,order=0,**extra):
        x,y,r,b=bbox;return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,**extra)
    for kind in [3,5,14]:
        c=copy.deepcopy(base['pages'][0]);c['name']='combined-asset/'+str(kind)
        c['request']['model']['layout_dets'].append(det(kind,[5,40,50,80],1,html='<table><td>cell</td></table>'if kind==5 else '',latex='x+y'))
        selected.append(c)
    for c in selected:
        r=c['request'];pd=copy.deepcopy(r['native_page'])
        for b in pd['blocks']:
            for line in b['lines']:line['bbox']=SimpleNamespace(bbox=line['bbox'])
        pd.update(size=[r['page_width'],r['page_height']],ori_image_list=[])
        accepted=[];calls=[]
        def capture(img,img_mode):
            v=original_contrast(img,img_mode)
            if v>=.17:accepted.append(img)
            return v
        class Writer:
            _parent_dir='images'
            def write(self,name,data):
                with Image.open(BytesIO(data))as image:image.load();calls.append(dict(name=name,mode=image.mode,width=image.width,height=image.height,pixels_sha256=digest(image.tobytes())))
        def normalize(v):
            if isinstance(v,np.ndarray):return 'ocr-7-'+str(next(i for i,a in enumerate(accepted)if a is v))
            if isinstance(v,dict):return {k:normalize(a)for k,a in v.items()}
            if isinstance(v,(list,tuple)):return [normalize(a)for a in v]
            return v
        try:
            sp.calculate_contrast=capture
            with Image.open(ROOT/c['source'])as image:page=old.page_model_info_to_page_info(copy.deepcopy(r['model']),dict(scale=r['scale'],img_pil=image),pd,Writer(),7,ocr_enable=False)
            before=normalize(page)
            pages=[page]if page is not None else [old.make_page_info_dict([],7,int(r['page_width']),int(r['page_height']),[])]
            middle=dict(pdf_info=pages);old.AtomModelSingleton=Manager;old._post_process_ocr(middle,'ch',{})
            after=copy.deepcopy(pages);old._optimize_formula_number_blocks(pages);old.para_split(pages)
        finally:old.AtomModelSingleton=original_manager;sp.calculate_contrast=original_contrast
        cases.append(dict(name=c['name'],request=r,source=c['source'],before_ocr=before,crops=[dict(name='ocr-7-'+str(i),width=a.shape[1],height=a.shape[0],rgb_sha256=digest(a[:,:,::-1].copy().tobytes()))for i,a in enumerate(accepted)],assets=calls,
            after_ocr=after,paragraphs=pages,outputs={mode:render.union_make(copy.deepcopy(pages),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}))
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/pdf_text_tool.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/model/ocr/rapid_ocr.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,model=model.relative_to(ROOT).as_posix(),model_sha256=digest(model.read_bytes()),inputs={c['source']:digest((ROOT/c['source']).read_bytes())for c in cases},sources={p:digest((ROOT/p).read_bytes())for p in refs},profile='实际旧函数及实际 CPU OCR，不注入固定识别响应；只将 bbox 转等坐标 JSON、np_img 换图片引用并另存像素哈希'))
    print(f'Frozen {len(cases)} native text→real CPU OCR→paragraph chains and {len(cases)*3} outputs')

if __name__=='__main__':main()
