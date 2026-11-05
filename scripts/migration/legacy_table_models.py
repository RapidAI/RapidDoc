"""仅供冻结旧基线：显式CPU会话，不自动下载模型。"""
from types import SimpleNamespace
import onnxruntime as ort
from common import ROOT

def create_models(metadata):
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapidocr.ch_ppocr_det.main import TextDetector
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
    from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
    from rapid_doc.model.orientation.rapid_orientation import RapidOrientation
    from rapid_doc.model.orientation.rapid_orientation.utils import ResizeImage,CropImage,NormalizeImage,ToCHWImage
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    def session(path):return ort.InferenceSession(str(ROOT/path),sess_options=opts,providers=['CPUExecutionProvider'])
    ds=session(metadata['det_model']);rs=session(metadata['rec_model']);os=session(metadata['orientation_model'])
    det=TextDetector.__new__(TextDetector);det.limit_type='max';det.limit_side_len=960;det.mean=[.485,.456,.406];det.std=[.229,.224,.225]
    det.session=lambda t:ds.run(None,{ds.get_inputs()[0].name:t})[0];det.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.6,use_dilation=True)
    worker=RapidOcrModel.__new__(RapidOcrModel);worker.text_detector=det;worker.is_seal=False;worker.enable_merge_det_boxes=False
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    worker.text_recognizer=SimpleNamespace(rec_batch_num=6,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=lambda t:rs.run(None,{rs.get_inputs()[0].name:t})[0],postprocess_op=CTCLabelDecode(character=rs.get_modelmeta().custom_metadata_map['character'].splitlines()))
    engine=RapidOrientation.__new__(RapidOrientation);engine.labels=os.get_modelmeta().custom_metadata_map['character'].splitlines();engine.session=lambda t:os.run(None,{os.get_inputs()[0].name:t});engine.preprocess_ops=[ResizeImage(resize_short=256),CropImage(224),NormalizeImage(),ToCHWImage()]
    orientation=RapidOrientationModel.__new__(RapidOrientationModel);orientation.orientation_engine=engine
    return worker,orientation

def create_full_models(metadata):
    """直接det+rec分支必须使用真实TextRecognizer.__call__，不能把旧行级rec适配器当成可调用模型。"""
    worker,orientation=create_models(metadata)
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    rec=TextRecognizer.__new__(TextRecognizer)
    rec.__dict__.update(worker.text_recognizer.__dict__)
    rec.cfg=SimpleNamespace(lang_type='ch',font_path=None)
    rec.RTL_LANGS={'arabic','fa','ur'}
    worker.text_recognizer=rec;worker.drop_score=.5
    return worker,orientation
