"""冻结 PP-DocLayoutV3 rect 模式的框规则与真实 PNG 输出。"""
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess
from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType,PP_DOCLAYOUTV2_layout_merge_bboxes_mode
from common import ROOT,write_json,digest

def main():
    model=ROOT/"rapid_doc/model/layout/rapid_layout_self/models/pp_doclayoutv3.onnx"
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=opts,providers=["CPUExecutionProvider"])
    labels=session.get_modelmeta().custom_metadata_map["character"].splitlines()
    def row(label,score,bounds,order):return [labels.index(label),score,*bounds,order]
    def case(name,rows,width=640,height=256,threshold=.5):
        rows=np.array(rows,np.float32).reshape(-1,7)
        post=PPPostProcess(labels,threshold,layout_merge_bboxes_mode=PP_DOCLAYOUTV2_layout_merge_bboxes_mode,layout_unclip_ratio=[1.,1.],scale_size=(800,800))
        result=post(rows.copy(),[width,height],layout_shape_mode="rect")
        expected=result if isinstance(result,list)else result.tolist()
        return dict(id=name,rows=rows.tolist(),width=width,height=height,threshold=threshold,expected=expected)
    cases=[case("empty",[]),case("strict-threshold",[row("text",.5,[10,10,50,30],0),row("text",.6,[60,10,100,30],1)]),
        case("same-class-nms",[row("text",.9,[10,10,90,40],1),row("text",.8,[12,11,92,41],0)]),
        case("different-class-nms",[row("text",.9,[10,10,90,40],1),row("figure_title",.8,[10,10,90,40],0)]),
        case("different-class-partial",[row("text",.9,[10,10,90,40],1),row("figure_title",.8,[20,10,100,40],0)]),
        case("large-background-image",[row("image",.95,[0,0,640,256],0),row("text",.9,[10,10,90,40],1)]),
        case("only-background-image",[row("image",.95,[0,0,640,256],0)]),
        case("large-title-containment",[row("doc_title",.8,[10,10,200,100],0),row("text",.9,[30,30,90,40],1)]),
        case("clipping",[row("text",.9,[-10,-5,700,300],0)]),
        case("degenerate-order-gap",[row("text",.9,[20,20,20,20],0),row("text",.8,[30,30,60,60],1)]),
        case("float-coordinates",[row("text",.99,[.1,.2,17.9,20.7],2),row("text",.9,[200.3,100.7,250.8,200.1],1)])]
    rng=np.random.default_rng(883);rows=[]
    for index in range(50):
        x,y=rng.uniform(0,300),rng.uniform(0,150);w,h=rng.uniform(10,280),rng.uniform(10,100)
        rows.append(row(rng.choice(["text","doc_title","image","inline_formula"]),float(rng.uniform(.3,.999)),[x,y,x+w,y+h],index))
    cases.append(case("seeded-overlap",rows))
    pngs=[]
    for name in ["text","rotated","blank"]:
        path=ROOT/f"build/migration/detection-images/{name}.png";image=np.asarray(Image.open(path).convert("RGB"));tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(image)
        outputs=session.run(None,dict(image=tensor,im_shape=np.array([[800,800]],np.float32),scale_factor=np.array([[800/image.shape[0],800/image.shape[1]]],np.float32)))
        value=case(f"model-{name}",outputs[0][:int(outputs[1][0])],image.shape[1],image.shape[0]);cases.append(value)
        pngs.append(dict(input=str(path),expected=value["expected"]))
    write_json(ROOT/"build/migration/layout-post.json",dict(cases=cases,png_cases=pngs,model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes()),labels=labels,scope="PP-DocLayoutV3 rect, scalar threshold, default merge modes; no mask polygons"))
    print(f"Exported {len(cases)} layout rule cases and {len(pngs)} PNG layout outputs")
if __name__=="__main__":main()
