"""Opt-in local parity capture; runs original Python once per selected document."""
import argparse,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from rapid_doc.main import RapidDoc
from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
import importlib
middle=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--only',nargs='+',required=True);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
context={};rec_original=RapidOcrModel.text_recognizer_call;det_original=RapidOcrModel.det_batch_predict;decode_original=CTCLabelDecode.decode;page_original=middle.page_model_info_to_page_info

def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()if x.size<512 else {'array_shape':list(x.shape)}
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def dump(name,v):
 (context['folder']/name).write_text(json.dumps(v,ensure_ascii=False,default=encode),encoding='utf-8')
def rec(self,inp,*a,**kw):
 i=context['rec'];context['rec']+=1
 images=[inp.img]if isinstance(inp.img,np.ndarray)else list(inp.img)
 np.savez(context['folder']/f'rec-{i:03}.npz',**{str(k):v for k,v in enumerate(images)})
 out=rec_original(self,inp,*a,**kw)
 dump(f'rec-{i:03}.json',dict(texts=out.txts,scores=out.scores,sizes=[list(v.shape)for v in images]))
 return out
def det(self,images,*a,**kw):
 i=context['det'];context['det']+=1
 out=det_original(self,images,*a,**kw)
 np.savez(context['folder']/f'det-{i:03}.npz',**{str(k):v for k,v in enumerate(images)})
 dump(f'det-{i:03}.json',[None if boxes is None else boxes.tolist() for boxes,_ in out])
 return out
def decode(self,text_index,text_prob=None,*a,**kw):
 out=decode_original(self,text_index,text_prob,*a,**kw)
 if text_prob is not None:
  records=[]
  for i,ids in enumerate(text_index):
   sel=np.ones(len(ids),dtype=bool)
   if kw.get('remove_duplicate',False):sel[1:]=ids[1:]!=ids[:-1]
   for blank in self.get_ignored_tokens():sel &= ids!=blank
   records.append(dict(text=out[0][i][0],score=out[0][i][1],probabilities=text_prob[i][sel].tolist()))
  index=context['ctc'];context['ctc']+=1;dump(f'ctc-{index:03}.json',records)
 return out
def page(page_model_info,image_dict,page_dict,image_writer,page_index,*a,**kw):
 dump(f'page-{page_index:03}-model.json',page_model_info)
 return page_original(page_model_info,image_dict,page_dict,image_writer,page_index,*a,**kw)
RapidOcrModel.text_recognizer_call=rec;RapidOcrModel.det_batch_predict=det;CTCLabelDecode.decode=decode;middle.page_model_info_to_page_info=page
for stem in args.only:
 folder=args.output/stem;folder.mkdir();context.update(folder=folder,rec=0,det=0,ctc=0)
 engine=RapidDoc(formula_config={'model_dir_or_path':str(ROOT/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx')},output_dir=folder/'exports')
 result=engine(ROOT/'demo/pdfs'/f'{stem}.pdf');dump('completed.json',dict(rec_calls=context['rec'],ctc_calls=context['ctc']))
 print('CAPTURED',stem,flush=True)
