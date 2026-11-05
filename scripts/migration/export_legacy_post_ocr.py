"""冻结实际旧后置 OCR 的选取/删除/回填；模型仅使用显式固定响应。"""
import copy,importlib,math
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-post-ocr.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');original=old.AtomModelSingleton;cases=[]
    def block(kind,tag,source=True):
        span=dict(type='text',bbox=[1,2,30,10],content='before',score=.123,unknown=tag)
        if source:span['np_img']=tag
        return dict(type=kind,bbox=[1,2,30,10],lines=[dict(spans=[span])])
    def page():return dict(preproc_blocks=[],discarded_blocks=[],para_blocks=[block('text','stale-para')],page_idx=0,page_size=[100,100])
    def add(name,pages,score=.9,responses=None):
        calls=[];requests=[]
        class Model:
            def ocr(self,images,**kwargs):
                calls.append(dict(images=copy.deepcopy(images),kwargs=kwargs))
                return [responses if responses is not None else [(str(i)+':'+str(img),score)for i,img in enumerate(images)]]
        class Manager:
            def get_atom_model(self,**kwargs):requests.append(kwargs);return Model()
        value=dict(pdf_info=copy.deepcopy(pages),unknown='preserve')
        try:
            old.AtomModelSingleton=Manager;old._post_process_ocr(value,'ch',dict(example=True))
        finally:old.AtomModelSingleton=original
        results=responses if responses is not None else [(str(i)+':'+str(img),score)for i,img in enumerate(calls[0]['images'])]if calls else []
        cases.append(dict(name=name,pages=pages,results=results,expected=value['pdf_info'],targets=calls[0]['images']if calls else [],calls=calls,requests=requests))
    for kind in ['text','title','formula_number','table_caption','image_caption','table_footnote','image_footnote','interline_equation','list','table_body','image_body','code']:
        for discarded in [False,True]:
            p=page();p['discarded_blocks'if discarded else 'preproc_blocks'].append(block(kind,kind));add(kind+'/'+str(discarded),[p])
    for group in ['table','image']:
        p=page();p['preproc_blocks'].append(dict(type=group,blocks=[block(k,k)for k in ['table_caption','table_footnote','image_caption','image_footnote','table_body','image_body','text']]))
        add('nested/'+group,[p])
    p=page();p['preproc_blocks']=[block('text','first'),block('text','ignored',False),dict(type='table',blocks=[block('table_body','ignore-body'),block('table_caption','caption')])];p['discarded_blocks']=[block('discarded','discard')]
    add('two-pages-order',[copy.deepcopy(p),copy.deepcopy(p)])
    for score in [0.,.49999999999999994,.5,math.nextafter(.5,math.inf),.5005,.5015,.5025,.9995,1.,1.2345,-.1,123456.1235]:
        for delta in [-1,0,1]:
            s=math.nextafter(score, -math.inf if delta<0 else math.inf)if delta else score
            value=page();value['preproc_blocks']=[block('text','crop')];add('round/'+str(score)+'/'+str(delta),[value],s)
    for payload in [None,False,0,'',[],{},'repeated']:
        p=page();p['preproc_blocks']=[block('text',payload),block('text',payload)];add('payload/'+repr(payload),[p])
    p=page();p['preproc_blocks']=[block('text','x'),block('title','y')];add('empty-and-unicode',[p],responses=[('',.9),('中文\nＡβ',.8765)])
    add('zero-pages',[]);add('no-images',[page()])
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/utils/ocr_utils.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in sources},profile='真实未修改旧 _post_process_ocr；仅以固定模型响应隔离回填，保留调用参数/图片顺序；np_img 用 JSON 标记代替像素，本文件不是模型识别质量证明'))
    print(f'Frozen {len(cases)} actual old post-OCR selection/rounding references')

if __name__=='__main__':main()
