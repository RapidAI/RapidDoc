"""Original RapidDoc API, actual CPU default models/auto masks, reported PDF references."""
import base64,json,os
from io import BytesIO
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/api-document.json'
    if target.exists():raise FileExistsError(target)
    os.environ['MINERU_DEVICE_MODE']='cpu';os.environ['MINERU_DONOT_CLEAN_MEM']='1'
    from rapid_doc import RapidDoc
    from PIL import Image
    lm=json.loads((ROOT/'build/migration/legacy-model.json').read_text(encoding='utf-8'))
    fm=json.loads((ROOT/'build/migration/formula-decode.json').read_text(encoding='utf-8'))['model']
    threads=dict(intra_op_num_threads=1,inter_op_num_threads=1,use_cuda=False,use_dml=False)
    engine=RapidDoc(layout_config=dict(model_dir_or_path=str(ROOT/lm['model']),engine_cfg=threads),
        formula_config=dict(model_dir_or_path=str(ROOT/fm['path']),engine_cfg=threads),table_config=dict(engine_cfg=threads),
        ocr_config={'seal_enable':False,'EngineConfig.onnxruntime.intra_op_num_threads':1,'EngineConfig.onnxruntime.inter_op_num_threads':1})
    cases=[]
    for name,end in [('demo1.pdf',1),('49－长表格-密集型.pdf',0),('small_ocr.pdf',0),('03.pdf',0),('示例1-论文模板.pdf',0)]:
        path=ROOT/'demo/pdfs'/name;print('Original API:',name,flush=True)
        result=engine(path,end_page_id=end)
        assets=[]
        for key,blob in result.images.items():
            with Image.open(BytesIO(blob))as image:assets.append(dict(name=key,mode=image.mode,width=image.width,height=image.height,pixels_sha256=digest(image.tobytes())))
        cases.append(dict(source=path.relative_to(ROOT).as_posix(),sha256=digest(path.read_bytes()),end_page_id=end,
            expected=dict(markdown=result.markdown,middle_json=result.middle_json,content_list_json=result.content_list_json),assets=assets))
        print(name,len(result.markdown),'characters,',len(result.images),'assets',flush=True)
    write_json(target,dict(cases=cases,scope='Original RapidDoc actual CPU V3 auto polygon layout/OCRv6/formula M/dual table models. Seal disabled (step 4).'))
if __name__=='__main__':main()
