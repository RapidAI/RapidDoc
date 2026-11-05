"""实际旧表格空OCR全图det+rec与模型输入回退；结构模型仅捕获输入。"""
import copy,json,math
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json
from export_table_text import norm,pixel

def main():
    target=ROOT/'build/migration/table-full-ocr.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils
    from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import ModelType
    from rapidocr.ch_ppocr_rec.main import TextRecognizer
    from rapid_doc.model.table.utils import normalize_table_ocr_text
    source='build/migration/table-text/pixels.png';rgb=np.asarray(Image.open(ROOT/source).convert('RGB'));bgr=rgb[:,:,::-1].copy()
    def quad(x,y,w,h):return [[float(x),float(y)],[float(x+w),float(y)],[float(x+w),float(y+h)],[float(x),float(y+h)]]
    boxes=[quad(1,1,1,12),quad(2,25,35,10),[[2.,45.],[39.,47.],[38.,57.],[1.,55.]]]
    masks=[[],[dict(bbox=[8,20,14,40],latex='x')],[dict(bbox=[-100,-100,1000,1000],latex='all')]]
    scores=[-1.,0.,.49,math.nextafter(.5,0),.5,math.nextafter(.5,1),.50001,1.,1.00001]
    cases=[];applies=[]
    for text in [None,'','香','哦樂','1號','10號','&<>"\'',' α ','\x1c A \u3000']:
        for score in scores:
            for ms in masks:
                crops=[];worker=RapidOcrModel.__new__(RapidOcrModel);worker.is_seal=False;worker.enable_merge_det_boxes=False;worker.drop_score=.5
                worker.text_detector=lambda im:SimpleNamespace(boxes=np.asarray(boxes,np.float32),elapse=0)
                class FakeRecognizer:
                    def __call__(self,args):
                        if not args.img:
                            empty=TextRecognizer.__new__(TextRecognizer);empty.rec_batch_num=6;empty.rec_image_shape=[3,48,320]
                            return empty(args) # actual old empty-list failure, not a fabricated successful result
                        crops.extend(pixel(im[:,:,::-1].copy())for im in args.img)
                        return SimpleNamespace(txts=[text]*len(args.img),scores=[score]*len(args.img),elapse=0)
                worker.text_recognizer=FakeRecognizer();error=False
                try:
                    result=worker.ocr(bgr.copy(),mfd_res=copy.deepcopy(ms))[0]
                    expected=[list(x)for x in zip(*[[r[0],normalize_table_ocr_text(r[1][0]),r[1][1]]for r in result])]if result else[]
                except ValueError:error=True;expected=None
                cases.append(dict(source=source,boxes=boxes,masks=ms,text=text,score=score,expected=norm(expected),raises=error,crops=crops))
            applies.append(dict(request=dict(boxes=[boxes[0]],results=[[text,score]]),expected=[[boxes[0]],[normalize_table_ocr_text(text)],[score]]if score>=.5 else[]))
    # Real CPU session/recognizer, including raw empty/no-32-alignment cases (no white padding).
    from legacy_table_models import create_full_models
    prior=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'))
    worker,orientation=create_full_models(prior);real=[];resolved=[]
    paths=['build/migration/detection-images/text.png','build/migration/detection-images/rotated.png','build/migration/detection-images/blank.png','build/migration/table-text/thin.png']
    for path in paths:
        arr=np.asarray(Image.open(ROOT/path).convert('RGB'));h,w=arr.shape[:2]
        for ms in [[],[dict(bbox=[0,0,w//3,h],latex='x')]]:
            raw=worker.ocr(arr[:,:,::-1].copy(),mfd_res=copy.deepcopy(ms))[0]
            rows=[list(x)for x in zip(*[[r[0],normalize_table_ocr_text(r[1][0]),r[1][1]]for r in raw])]if raw else[]
            real.append(dict(source=path,source_sha256=digest((ROOT/path).read_bytes()),masks=ms,expected=norm(dict(boxes=rows[0]if rows else[],ocr_result=rows))))
        for ocr in [None,[],[[],[],[]],[[quad(1,1,10,10)],['prior&amp;'],[.1]]]:
            captured=[];calls=[];saved=worker.ocr
            def spy(*a,**k):calls.append(True);return saved(*a,**k)
            worker.ocr=spy
            def capture(images,rows):captured.append(dict(image=pixel(images[0][:,:,::-1].copy()),ocr_result=norm(rows[0])));return SimpleNamespace(pred_htmls=[None])
            model=RapidTableModel.__new__(RapidTableModel);model.model_type=ModelType.SLANETPLUS;model.table_model=capture;model.ocr_engine=worker
            ims=[dict(ocr_bbox=quad(0,0,min(w-1,30),min(h-1,20)),uuid='image')];ms=[dict(bbox=[1,1,4,4],latex='x<&>')]
            try:model.predict(arr.copy(),ocr_result=copy.deepcopy(ocr),fill_image_res=copy.deepcopy(ims),mfd_res=copy.deepcopy(ms),skip_table_orientation=True)
            finally:worker.ocr=saved
            resolved.append(dict(source=path,source_sha256=digest((ROOT/path).read_bytes()),request=dict(ocr_result=ocr,fill_images=ims,formulas=ms),expected=dict(ocr_result=captured[0]['ocr_result']if captured else None,needs_ocr=False,can_infer=bool(captured),full_ocr_used=bool(calls)),image=captured[0]['image']if captured else pixel(arr)))
    refs=['rapid_doc/model/ocr/rapid_ocr.py','rapid_doc/model/ocr/ocr_patch.py','rapid_doc/utils/ocr_utils.py','rapid_doc/model/table/rapid_table.py','rapid_doc/model/table/utils.py']
    models={k:prior[k]for k in ['det_model','det_sha256','rec_model','rec_sha256']}
    write_json(target,dict(cases=cases,applies=applies,real=real,resolved=resolved,sources={p:digest((ROOT/p).read_bytes())for p in refs},**models,scope='Actual old RapidOcrModel.__call__ full OCR at default 0.5 and no merge/white padding/pre-mask; fixed rec isolates score/crop policy. Real direct TextRecognizer CPU cases separately. Already selected orientation / skip_table_orientation=True only. Structure network captures input and is not run. Default ch language, no RTL/seal/word boxes or old batch-retry contract.'))
    print(f'Frozen {len(cases)} full-OCR policies/{len(applies)} score policies/{len(real)} real CPU full-OCR/{len(resolved)} real model-input resolutions')
if __name__=='__main__':main()
