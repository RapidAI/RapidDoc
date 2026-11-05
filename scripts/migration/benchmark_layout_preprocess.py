"""Same-pixel production Python layout preprocessing, outside model execution."""
import argparse,hashlib,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--native',type=Path)
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 import numpy as np
 import cv2
 from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
 from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType
 preprocess=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3);rows=[]
 for width,height in [(1600,2200),(800,800),(400,600)]:
  index=np.arange(width*height*3,dtype=np.uint64);pixels=((index*37+index//13)%256).astype(np.uint8).reshape(height,width,3)
  seconds=[]
  for i in range(13):
   start=time.perf_counter();tensor=preprocess(pixels);elapsed=time.perf_counter()-start
   if i>=3:seconds.append(elapsed)
  data=np.ascontiguousarray(tensor,dtype='<f4').tobytes();name=f'{width}x{height}.f32';(args.output/name).write_bytes(data)
  row=dict(source=[width,height],seconds=seconds,median_seconds=statistics.median(seconds),sha256=hashlib.sha256(data).hexdigest())
  if args.native:
   expected=np.frombuffer((args.native/name).read_bytes(),dtype='<f4');actual=np.frombuffer(data,dtype='<f4')
   row['different_tensor_values']=int(np.count_nonzero(expected.view('<u4')!=actual.view('<u4')))
  rows.append(row);print(row,flush=True)
 (args.output/'report.json').write_text(json.dumps(dict(opencv=cv2.__version__,cases=rows),indent=2))
if __name__=='__main__':main()
