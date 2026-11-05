"""冻结当前 Paddle 表格分类前处理、解码与真实 CPU 模型输出。"""
import cv2
import numpy as np
import onnxruntime as ort
from rapid_doc.model.table.rapid_table_self.table_cls.main import PaddleCls,QanythingCls,TableCls
from rapid_doc.model.table.rapid_table_self.inference_engine.onnxruntime.optimization import classifier_output_type
from common import ROOT,write_json,digest

def main():
    processor=PaddleCls.__new__(PaddleCls)
    processor.inp_h=processor.inp_w=224;processor.resize_short=256
    processor.mean=np.array([.485,.456,.406],np.float32)
    processor.std=np.array([.229,.224,.225],np.float32)
    output=ROOT/'build/migration/table-classification';output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(7842);cases=[]
    for height,width in [(1,1),(1,10),(3,5),(4,8),(43,77),(77,43),(224,224),(256,256),(256,320),(320,480),(1024,768),(32,1000),(319,479)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        tensor=processor.batch_preprocess([image])
        name=f'{width}x{height}';source=output/f'{name}.bgr';expected=output/f'{name}.f32'
        source.write_bytes(image.tobytes());expected.write_bytes(tensor.astype('<f4',copy=False).tobytes())
        cases.append(dict(id=name,width=width,height=height,bgr_file=source.relative_to(ROOT).as_posix(),tensor_file=expected.relative_to(ROOT).as_posix(),sha256=digest(expected.read_bytes())))
    model=ROOT/'rapid_doc/model/table/rapid_table_self/models/paddle_cls.onnx'
    output_type=classifier_output_type(model)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    processor.session=lambda x:session.run(None,{session.get_inputs()[0].name:x})
    processor.output_type=output_type;processor.cls={0:'wired',1:'wireless'}
    images=[]
    for name in ['wired','wireless','blank']:
        path=ROOT/f'build/migration/table-images/{name}.png';image=cv2.imread(str(path))
        tensor=processor.batch_preprocess([image]);labels,scores=processor.predict_with_scores(tensor)
        images.append(dict(input=str(path),label=labels[0],score=scores[0]))
    qprocessor=QanythingCls.__new__(QanythingCls)
    qprocessor.inp_h=qprocessor.inp_w=224;qprocessor.mean=processor.mean;qprocessor.std=processor.std;qprocessor.cls=processor.cls
    qcases=[]
    for case in cases:
        image=np.frombuffer((ROOT/case['bgr_file']).read_bytes(),np.uint8).reshape(case['height'],case['width'],3)
        tensor=qprocessor.batch_preprocess([image]);path=output/f"q-{case['id']}.f32";path.write_bytes(tensor.astype('<f4',copy=False).tobytes())
        qcases.append(dict(case,tensor_file=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes())))
    qmodel=ROOT/'rapid_doc/model/table/rapid_table_self/models/q_cls.onnx'
    qsession=ort.InferenceSession(str(qmodel),sess_options=options,providers=['CPUExecutionProvider'])
    qprocessor.session=lambda x:qsession.run(None,{qsession.get_inputs()[0].name:x})
    qimages=[]
    for image in images:
        labels,scores=qprocessor.predict_with_scores(qprocessor.batch_preprocess([cv2.imread(image['input'])]))
        qimages.append(dict(input=image['input'],label=labels[0],score=scores[0]))
    # 执行旧 TableCls 的四种标签组合，冻结其共识策略。
    class Fixed:
        def __init__(self,labels,scores):self.labels=labels;self.scores=scores
        def batch_preprocess(self,x):return x
        def predict_with_scores(self,x):return self.labels,self.scores
    p_labels=['wired','wired','wireless','wireless'];q_labels=['wired','wireless','wired','wireless']
    p_scores=[.9,.8,.7,.6];q_scores=[.8,.9,.6,.7]
    ensemble=TableCls.__new__(TableCls);ensemble.table_engine=Fixed(p_labels,p_scores);ensemble.table_engine2=Fixed(q_labels,q_scores);ensemble._load_imgs=lambda x:x
    labels,scores,_=ensemble([None]*4,return_scores=True)
    consensus=[dict(primary=dict(label=p,score=ps),secondary=dict(label=q,score=qs),label=l,score=s)for p,ps,q,qs,l,s in zip(p_labels,p_scores,q_labels,q_scores,labels,scores)]
    decode=[]
    for kind,values in [('probabilities',[.9,.1]),('probabilities',[.5,.5]),('probabilities',[0.,1.]),('logits',[2000.,1999.]),('logits',[-1000.,1000.]),('logits',[-1.,-1.])]:
        processor.output_type=kind;processor.session=lambda x,v=values:[np.array([v],np.float32)]
        labels,scores=processor.predict_with_scores(None)
        decode.append(dict(output_type=kind,values=values,label=labels[0],score=scores[0]))
    write_json(ROOT/'build/migration/table-classification.json',dict(cases=cases,qcases=qcases,images=images,qimages=qimages,consensus=consensus,decode=decode,model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes()),qmodel=qmodel.relative_to(ROOT).as_posix(),qsha256=digest(qmodel.read_bytes()),output_type=output_type,opencv=cv2.__version__))
    print(f'Exported {len(cases)} Paddle + {len(qcases)} Qanything tensors, {len(decode)} decoder cases, {len(images)+len(qimages)} CPU PNG classifications and 4 consensus cases')
if __name__=='__main__':main()
