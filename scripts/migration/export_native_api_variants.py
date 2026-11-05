"""Freeze original public native-layout/window behavior in a separate process."""
import json,os
from io import BytesIO
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/native-api-variants.json'
    if target.exists():raise FileExistsError(target)
    os.environ['MINERU_DEVICE_MODE']='cpu';os.environ['MINERU_DONOT_CLEAN_MEM']='1'
    from rapid_doc import RapidDoc
    from PIL import Image
    threads=dict(intra_op_num_threads=1,inter_op_num_threads=1,use_cuda=False,use_dml=False)
    cases=[]
    variants=[('示例1-论文模板.pdf',0,dict(layout_config=dict(model_type='docvortex_rule'),formula_enable=False)),
              ('small_ocr.pdf',0,dict(layout_config=dict(model_type='docvortex_rule'),formula_enable=False)),
              ('demo1.pdf',1,dict(layout_config=dict(docvortex_mode='rule'),pdf_pages_batch=1))]
    for name,end,options in variants:
        print('Original native API:',name,options,flush=True)
        config=dict(options);config['layout_config']=dict(config['layout_config'],engine_cfg=threads)
        config['formula_config']=dict(engine_cfg=threads);config['table_config']=dict(engine_cfg=threads)
        config['ocr_config']={'seal_enable':False,'EngineConfig.onnxruntime.intra_op_num_threads':1,'EngineConfig.onnxruntime.inter_op_num_threads':1}
        source=ROOT/'demo/pdfs'/name;result=RapidDoc(**config)(source,end_page_id=end)
        assets=[]
        for key,blob in result.images.items():
            with Image.open(BytesIO(blob))as image:assets.append(dict(name=key,mode=image.mode,width=image.width,height=image.height,pixels_sha256=digest(image.tobytes())))
        cases.append(dict(source=source.relative_to(ROOT).as_posix(),sha256=digest(source.read_bytes()),options=options,end_page_id=end,
                          expected=dict(markdown=result.markdown,middle_json=result.middle_json,content_list_json=result.content_list_json),assets=assets))
    write_json(target,dict(cases=cases,scope='Original public native-only text/scan and hybrid rule with one-page batches. CPU, no seal/CUDA.'))
if __name__=='__main__':main()
