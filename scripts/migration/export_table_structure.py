"""导出 SLANet 两变体的前处理、EOS/重复 token 和真实 PNG 结构合同。"""
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont
from rapid_doc.model.table.rapid_table_self.table_structure.pp_structure.pre_process import TablePreprocess
from rapid_doc.model.table.rapid_table_self.table_structure.pp_structure.post_process import TableLabelDecode
from rapid_doc.model.table.rapid_table_self.utils.typings import ModelType
from common import ROOT,write_json,digest

def main():
    pre=TablePreprocess();cases=[];rng=np.random.default_rng(518)
    directory=ROOT/"build/migration/table-inputs";directory.mkdir(parents=True,exist_ok=True)
    for height,width in [(3,5),(77,43),(96,640),(97,641),(1,10),(488,488),(1,1000)]:
        image=rng.integers(0,256,(height,width,3),dtype=np.uint8)
        value=dict(id=f"{width}x{height}",width=width,height=height,bgr=image.ravel().tolist())
        if int(height*488/max(width,height))==0 or int(width*488/max(width,height))==0:value["expected_error"]="invalid_input"
        else:
            tensors,shapes=pre([image]);tensor=tensors[0]
            value.update(shape=[1,*tensor.shape],shape_list=shapes[0].tolist(),fp32_le_sha256=digest(tensor.astype("<f4",copy=False).tobytes()))
        cases.append(value)
    pngs=[];output=ROOT/"build/migration/table-images";output.mkdir(parents=True,exist_ok=True)
    for name in ["wired","wireless","blank"]:
        image=Image.new("RGB",(480,240),"white");draw=ImageDraw.Draw(image)
        if name!="blank":
            font=ImageFont.truetype(str(ROOT/"rapid_doc/_vendor/docvortex_core/resources/fonts/DroidSansFallbackFull.ttf"),24)
            for y,text in [(24,"Header"),(96,"Alpha"),(168,"Beta")]:draw.text((32,y),text,font=font,fill="black")
            for y,text in [(96,"123"),(168,"456")]:draw.text((270,y),text,font=font,fill="black")
            if name=="wired":
                for y in [12,84,156,228]:draw.line((12,y,468,y),fill="black",width=2)
                for x in [12,468]:draw.line((x,12,x,228),fill="black",width=2)
                draw.line((240,84,240,228),fill="black",width=2)
        path=output/f"{name}.png";image.save(path);pngs.append(path)
    synthetic=[];models=[]
    for variant,path,kind,d in [
        ("slanet-plus","rapid_doc/model/table/rapid_table_self/models/slanet-plus.onnx",ModelType.SLANETPLUS,8),
        ("slanet-1m","rapid_doc/resources/slanet-1m.onnx",ModelType.SLANET1M,4),
    ]:
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
        session=ort.InferenceSession(str(ROOT/path),sess_options=opts,providers=["CPUExecutionProvider"])
        dictionary=session.get_modelmeta().custom_metadata_map["character"];decoder=TableLabelDecode(dictionary.splitlines(),dict(model_type=kind))
        inputs=[]
        for png in pngs:
            bgr=np.asarray(Image.open(png))[:,:,::-1].copy();tensors,shapes=pre([bgr]);pred=session.run(None,{session.get_inputs()[0].name:np.array(tensors)})
            structures,bboxes=decoder(pred[0],pred[1],shapes,[bgr])
            inputs.append(dict(input=str(png),tokens=structures[0][0],confidence=structures[0][1],cell_bboxes=bboxes[0].tolist()))
        models.append(dict(variant=variant,model=path,sha256=digest((ROOT/path).read_bytes()),png_cases=inputs))
        dictionary="<tbody>\n</tbody>\n<tr>\n</tr>\n<td>";decoder=TableLabelDecode(dictionary.splitlines(),dict(model_type=kind))
        labels=decoder.character;sequence=["eos","sos","<tbody>","<tr>","<td></td>","<td></td>","</tr>","</tbody>","eos","<td></td>"]
        prob=np.zeros((1,len(sequence),len(labels)),np.float32);bbox=np.zeros((1,len(sequence),d),np.float32)
        for index,token in enumerate(sequence):prob[0,index,labels.index(token)]=.9
        bbox[0,5]=np.array([.1,.2,.4,.5]if d==4 else[.1,.2,.4,.2,.4,.5,.1,.5],np.float32)
        structs,boxes=decoder(bbox.copy(),prob,np.array([[240,480,488/480,488/480,488,488]]),[np.zeros((240,480,3),np.uint8)])
        synthetic.append(dict(variant=variant,dictionary=dictionary,bbox_shape=list(bbox.shape),bbox_values=bbox.ravel().tolist(),prob_shape=list(prob.shape),prob_values=prob.ravel().tolist(),width=480,height=240,tokens=structs[0][0],confidence=structs[0][1],cell_bboxes=boxes[0].tolist()))
    write_json(ROOT/"build/migration/table-structure.json",dict(preprocess_cases=cases,models=models,decode_cases=synthetic))
    print(f"Exported {len(cases)} table preprocessing cases, 2 sequence contracts and 6 PNG structure baselines")
if __name__=="__main__":main()
