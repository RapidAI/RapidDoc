"""旧实际页面→真实 CPU 识别响应回填→编号→段落→输出参考。"""
import copy,importlib,json
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-post-ocr-chain.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent');original=old.AtomModelSingleton
    batch_path=ROOT/'build/migration/recognition-batch.json';batch=json.loads(batch_path.read_text());cases=[]
    for c in batch['cases']:
        if c['unsupported']:continue
        dets=[]
        for i in range(len(c['sources'])):
            x,y,r,b=10,10+i*25,170,30+i*25;poly=[x,y,r,y,r,b,x,b]
            dets.extend([dict(category_id=1,poly=poly,score=.95,original_order=i),dict(category_id=15,poly=poly,score=.95,original_order=i,text='pending')])
        with Image.new('RGB',(200,250),'white')as image:
            page=old.page_model_info_to_page_info(dict(layout_dets=dets),dict(scale=1.,img_pil=image),dict(size=[200,250],ori_image_list=[]),None,0,ocr_enable=True)
        count=0;images={}
        for block in page['preproc_blocks']:
            for line in block['lines']:
                for span in line['spans']:
                    key='image-'+str(count);span['np_img']=key;images[key]=c['sources'][count];count+=1
        assert count==len(c['sources'])
        class Model:
            def ocr(self,images,**kwargs):assert images==list(globals_images);return [c['results']]
        class Manager:
            def get_atom_model(self,**kwargs):return Model()
        globals_images=list(images);middle=dict(pdf_info=[copy.deepcopy(page)])
        try:old.AtomModelSingleton=Manager;old._post_process_ocr(middle,'ch',{})
        finally:old.AtomModelSingleton=original
        after=copy.deepcopy(middle['pdf_info']);old._optimize_formula_number_blocks(middle['pdf_info']);old.para_split(middle['pdf_info'])
        cases.append(dict(name=c['name'],pages=[page],images=images,batch_size=c['batch_size'],expected=after,paragraphs=middle['pdf_info'],outputs={mode:render.union_make(copy.deepcopy(middle['pdf_info']),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}))
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,batch_sha256=digest(batch_path.read_bytes()),sources={p:digest((ROOT/p).read_bytes())for p in sources},profile='实际旧页面及后置 OCR 函数，模型响应来自另存的实际旧 CPU 批推理；该处仅隔离已冻结响应，不是新模型质量证明'))
    print(f'Frozen {len(cases)} real CPU response→post OCR→paragraph chains and {len(cases)*3} outputs')

if __name__=='__main__':main()
