"""多输入和整型输出合同：合成模型覆盖整型精度，真实版面模型覆盖运行路径。"""
import json
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
import onnxruntime as ort
from PIL import Image
from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
from rapid_doc.model.layout.rapid_layout_self.model_handler.utils import ModelType
from common import ROOT,write_json,digest

def capture(model,inputs):
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=opts,providers=["CPUExecutionProvider"])
    results=session.run(None,inputs)
    outputs={v.name:dict(dtype=str(result.dtype),shape=list(result.shape),values=result.ravel().tolist())for v,result in zip(session.get_outputs(),results)}
    return dict(model=model.relative_to(ROOT).as_posix(),sha256=digest(model.read_bytes()),inputs={name:dict(shape=list(value.shape),values=value.ravel().tolist())for name,value in inputs.items()},outputs=outputs)

def main():
    model=ROOT/"build/migration/named-test.onnx"
    graph=helper.make_graph([
        helper.make_node("Add",["left","right"],["sum"]),helper.make_node("Cast",["sum"],["int32"],to=TensorProto.INT32),
        helper.make_node("Constant",[],["int64"],value=numpy_helper.from_array(np.array([9007199254740993,-9007199254740993],np.int64))),
    ],"named-contract",[helper.make_tensor_value_info(name,TensorProto.FLOAT,[2,2])for name in ["left","right"]],
       [helper.make_tensor_value_info("sum",TensorProto.FLOAT,[2,2]),helper.make_tensor_value_info("int32",TensorProto.INT32,[2,2]),helper.make_tensor_value_info("int64",TensorProto.INT64,[2])])
    value=helper.make_model(graph,opset_imports=[helper.make_opsetid("",13)],ir_version=9);onnx.checker.check_model(value);onnx.save(value,str(model))
    cases=[capture(model,dict(left=np.array([[1,-2],[3,4]],np.float32),right=np.array([[2,4],[5,-7]],np.float32)))]
    model=ROOT/"rapid_doc/model/layout/rapid_layout_self/models/pp_doclayoutv3.onnx"
    image=np.asarray(Image.open(ROOT/"build/migration/detection-images/text.png").convert("RGB"))
    tensor=PPPreProcess((800,800),ModelType.PP_DOCLAYOUTV3)(image)
    cases.append(capture(model,dict(image=tensor,im_shape=np.array([[800,800]],np.float32),scale_factor=np.array([[800/image.shape[0],800/image.shape[1]]],np.float32))))
    write_json(ROOT/"build/migration/named-baseline.json",dict(cases=cases,scope="Prepared multi-input tensors and typed outputs; Rust layout preprocessing/postprocessing not covered"))
    print(f"Exported {len(cases)} named model cases; layout rows {cases[-1]['outputs']['fetch_name_0']['shape']}")
if __name__=="__main__":main()
