"""补充旧页面字典去重的 int/float/bool 等值语义及多边形十进制舍入。"""
import copy,importlib,json
from PIL import Image
from common import ROOT,digest,write_json


def main():
    target=ROOT/'build/migration/legacy-page-numbers.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');cases=[]
    def run(name,dets):
        r=dict(model=dict(layout_dets=dets),page_width=600,page_height=800,page_index=0,scale=1.,formula_enabled=True)
        image=Image.new('RGB',(600,800),'white')
        try:expected=old.page_model_info_to_page_info(copy.deepcopy(r['model']),dict(scale=1.,img_pil=image),dict(size=[600,800],ori_image_list=[]),None,0,ocr_enable=True)
        finally:image.close()
        cases.append(dict(name=name,request=r,expected=expected))
    def seal(value):return dict(category_id=3,poly=[10,10,300,10,300,200,10,200],original_label='seal',original_order=0,score=.95,text=value)
    for i,values in enumerate([(2**53,2**53+1),(float(2**53),2**53+1),(2**53+1,float(2**53)),(2**53+2,2**53,2**53+1),
                               (1,1.),(1.,1),(True,1),(False,0),(2**64-1,2**64-2)]):run(f'python-numeric-equality-{i}',[seal(v)for v in values])
    for value in [.005,.015,1.005,2.675,2.685,-.005,-2.675,10.145,100.555]:
        d=seal('印章');d['polygon_points']=[[value,value],[300.005,10.005],[300.015,200.005]];run('polygon-round-'+str(value),[d])
    paths=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/pipeline_magic_model.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in paths},profile='实际旧单页转换；大整数/float/bool 字典等值去重及 Python round(x,2)；writer=None'))
    print(f'Frozen {len(cases)} old numeric page references')


if __name__=='__main__':main()
