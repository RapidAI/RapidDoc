"""用未修改旧 text_recognizer_call 与真实 CPU 模型冻结共同 padding、还原顺序和输出。"""
import hashlib,importlib,importlib.metadata,types
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rapidocr.ch_ppocr_rec.main import TextRecognizer
from rapidocr.ch_ppocr_rec.typings import TextRecInput
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/recognition-batch.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.model.ocr.rapid_ocr');model=ROOT/'rapid_doc/resources/PP-OCRv6_rec_small.onnx'
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    rec=TextRecognizer.__new__(TextRecognizer);rec.rec_image_shape=[3,48,320]
    decoder=CTCLabelDecode(character=session.get_modelmeta().custom_metadata_map['character'].splitlines())
    folder=ROOT/'build/migration/recognition-batch-images';folder.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,(text,width,height)in enumerate([('A',80,48),('Batch 22',160,48),('识别测试',240,48),('Wide text line 123456789',500,48),('extremely wide line 12345678901234567890',800,48)]):
        image=Image.new('RGB',(width,height),'white');font=ImageFont.truetype(str(ROOT/'rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf'),28)
        ImageDraw.Draw(image).text((2,4),text,font=font,fill='black');path=folder/f'{i}.png';image.save(path);image.close();paths.append(path.relative_to(ROOT).as_posix())
    cases=[]
    def add(name,sources,batch_size,unsupported=False):
        images=[]
        for path in sources:
            with Image.open(ROOT/path)as image:images.append(np.asarray(image.convert('RGB'))[:,:,::-1].copy())
        indices=np.argsort(np.array([im.shape[1]/im.shape[0]for im in images]));calls=[];expected={}
        def run(tensor):
            offset=sum(c['shape'][0]for c in calls);calls.append(dict(shape=list(tensor.shape),sha256=digest(tensor.tobytes())))
            for j in range(len(tensor)):expected[int(indices[offset+j])]=dict(shape=list(tensor.shape[1:]),sha256=digest(tensor[j].tobytes()))
            return session.run(None,{session.get_inputs()[0].name:tensor})[0]
        worker=old.RapidOcrModel.__new__(old.RapidOcrModel)
        worker.text_recognizer=types.SimpleNamespace(rec_batch_num=batch_size,rec_image_shape=[3,48,320],resize_norm_img=rec.resize_norm_img,session=run,postprocess_op=decoder)
        result=worker.text_recognizer_call(TextRecInput(img=images))
        cases.append(dict(name=name,sources=sources,batch_size=batch_size,indices=indices.tolist(),calls=calls,
            tensors=[expected[i]for i in range(len(images))],results=list(zip(result.txts,[float(v)for v in result.scores])),unsupported=unsupported))
    for batch in [1,2,3,6]:add('mixed/'+str(batch),[paths[i]for i in [3,0,4,1,2]],batch)
    for batch in [1,2,6]:add('ties-equal-padding/'+str(batch),[paths[i]for i in [2,0,0,1,2,1,0]],batch)
    base=[f'build/migration/recognition-lines/line-{i}.png'for i in range(4)]
    for batch in [1,3,4,6]:add('previous-lines/'+str(batch),base,batch)
    add('ties-cross-changing-padding',[paths[0]]*13+[paths[4]],6,True)
    refs=['rapid_doc/model/ocr/rapid_ocr.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py']
    write_json(target,dict(cases=cases,model=model.relative_to(ROOT).as_posix(),model_sha256=digest(model.read_bytes()),
        inputs={p:digest((ROOT/p).read_bytes())for c in cases for p in c['sources']},sources={p:digest((ROOT/p).read_bytes())for p in refs},
        rapidocr=importlib.metadata.version('rapidocr'),numpy=np.__version__,profile='实际旧批调用＋真实 CPU ONNX；每个原始图片的共同 padding 张量 FP32 位模式和结果顺序；同分批内置换仅按原始图片对照，同分跨批改变 padding 的案例明确 unsupported，不归一化得分'))
    print(f'Frozen {len(cases)} old CPU recognition batches')

if __name__=='__main__':main()
