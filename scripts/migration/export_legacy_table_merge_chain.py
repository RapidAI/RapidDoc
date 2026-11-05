"""冻结实际旧单页→编号→段落→跨页表格→三种输出；不覆盖基线。"""
import copy,importlib
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-table-merge-chain.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    table=importlib.import_module('rapid_doc.utils.table_merge')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent');cases=[]
    def det(kind,bbox,order,**extra):
        x,y,r,b=bbox;return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,**extra)
    for length in [2,3]:
        for caption in [None,'续表','新表','下一节']:
            for notes in [0,1,2]:
                for numbers in [False,True]:
                    requests=[];pages=[]
                    for i in range(length):
                        html=f'<table><tr><th>ID</th><th>Value</th></tr><tr><td>{i+1}</td><td>value {i+1}</td></tr></table>'
                        dets=[det(5,[20,30,120,110],0,html=html)]
                        if i and caption is not None:
                            box=[20,120,120,130]if caption=='下一节'else [20,20,120,29]
                            dets.extend([det(6,box,1),det(15,box,1,text=caption)])
                        if i<length-1:
                            for n in range(notes):
                                box=[20,115+n*12,120,125+n*12];dets.extend([det(7,box,3+n),det(15,box,3+n,text='note '+str(n))])
                        request=dict(model=dict(layout_dets=dets),page_width=200,page_height=200,page_index=i,scale=1.,formula_enabled=True);requests.append(request)
                        with Image.new('RGB',(200,200),'white')as image:
                            page=old.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=1.,img_pil=image),dict(size=[200,200],ori_image_list=[]),None,i,ocr_enable=True)
                        pages.append(page)
                    if numbers:old._optimize_formula_number_blocks(pages)
                    old.para_split(pages);unmerged=copy.deepcopy(pages);table.merge_table(pages)
                    cases.append(dict(name=f'{length}/{caption}/{notes}/{numbers}',requests=requests,optimize_formula_numbers=numbers,unmerged=unmerged,expected=pages,outputs={mode:render.union_make(copy.deepcopy(pages),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}))
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/pipeline_magic_model.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/utils/table_merge.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in sources},profile='实际旧完整 stage 串联；writer=None、OCR 已识别、模型 original_order；未做模型推理/资产/PDF 原图提取'))
    print(f'Frozen {len(cases)} model→paragraph→table merge chains and {len(cases)*3} old outputs')

if __name__=='__main__':main()
