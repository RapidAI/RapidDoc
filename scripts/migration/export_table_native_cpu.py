"""实际旧PDF→DB检测→方向→原生文字/CTC回退链；结构模型仅捕获输入，不执行。"""
import copy,json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import onnxruntime as ort
from common import ROOT,digest,write_json
from export_table_native import adapt
from export_table_text import norm

def main():
    target=ROOT/'build/migration/table-native-cpu.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc.backend.pipeline.model_list import AtomicModel
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
    from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
    from rapid_doc.model.orientation.rapid_orientation import RapidOrientation
    from rapid_doc.model.orientation.rapid_orientation.utils import ResizeImage,CropImage,NormalizeImage,ToCHWImage
    prior=json.loads((ROOT/'build/migration/table-text.json').read_text(encoding='utf-8'));refs=json.loads((ROOT/'build/migration/table-native.json').read_text(encoding='utf-8'))
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    def session(path):return ort.InferenceSession(str(ROOT/path),sess_options=opts,providers=['CPUExecutionProvider'])
    ds=session(prior['det_model']);rs=session(prior['rec_model']);op='rapid_doc/resources/rapid_orientation.onnx';os=session(op)
    det=TextDetector.__new__(TextDetector);det.limit_type='max';det.limit_side_len=960;det.mean=[.485,.456,.406];det.std=[.229,.224,.225]
    det.session=lambda t:ds.run(None,{ds.get_inputs()[0].name:t})[0];det.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)
    worker=RapidOcrModel.__new__(RapidOcrModel);worker.text_detector=det;worker.is_seal=False;worker.enable_merge_det_boxes=False
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    worker.text_recognizer=SimpleNamespace(rec_batch_num=6,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=lambda t:rs.run(None,{rs.get_inputs()[0].name:t})[0],postprocess_op=CTCLabelDecode(character=rs.get_modelmeta().custom_metadata_map['character'].splitlines()))
    engine=RapidOrientation.__new__(RapidOrientation);engine.labels=os.get_modelmeta().custom_metadata_map['character'].splitlines();engine.session=lambda t:os.run(None,{os.get_inputs()[0].name:t});engine.preprocess_ops=[ResizeImage(resize_short=256),CropImage(224),NormalizeImage(),ToCHWImage()]
    orientation=RapidOrientationModel.__new__(RapidOrientationModel);orientation.orientation_engine=engine
    saved=old._run_table_ocr;cases=[]
    for c in refs['real'][::2]:
        request=c['request'];rgb=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
        for force in [False,True]:
            captured=[];fallback=[]
            tm=SimpleNamespace(predict=lambda image,ocr_result,*a,**k:(captured.append(norm(ocr_result))or None))
            def run_ocr(*a,**k):fallback.append(True);return saved(*a,**k)
            old._run_table_ocr=run_ocr
            manager=SimpleNamespace(get_atom_model=lambda **k:worker if k['atom_model_name']==AtomicModel.OCR else orientation if k['atom_model_name']==AtomicModel.ImgOrientationCls else tm)
            task=dict(table_res=copy.deepcopy(request['table']),table_img=rgb.copy(),useful_list=copy.deepcopy(request['useful_list']),lang=None,single_page_mfdetrec_res=[],checkbox_res=[],ocr_enable=False)
            try:old._process_single_table(task,adapt(request['native_page']),request['scale'],manager,dict(use_rule_table=False,force_ocr=force,table_formula_enable=False,table_image_enable=False),{})
            finally:old._run_table_ocr=saved
            cases.append(dict(source=c['source'],source_sha256=c['source_sha256'],pdf=c['pdf'],request=dict(native_page=request['native_page'],table=request['table'],useful_list=request['useful_list'],scale=request['scale'],force_ocr=force,ocr_enable=False),expected=captured[0],text_source='ocr'if fallback else'native'if captured[0]else'none'))
    paths=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/ocr/rapid_ocr.py','rapid_doc/utils/span_pre_proc.py']
    write_json(target,dict(cases=cases,det_model=prior['det_model'],det_sha256=prior['det_sha256'],rec_model=prior['rec_model'],rec_sha256=prior['rec_sha256'],orientation_model=op,orientation_sha256=digest((ROOT/op).read_bytes()),sources={p:digest((ROOT/p).read_bytes())for p in paths},scope='Actual legacy PDF render/facts → real DB → native direction/orientation → native text or real CTC. Table structure model only captures OCR result; it is not executed.'))
    print(f'Frozen {len(cases)} actual PDF/CPU table-text chains; native={sum(c["text_source"]=="native"for c in cases)}, OCR={sum(c["text_source"]=="ocr"for c in cases)}')
if __name__=='__main__':main()
