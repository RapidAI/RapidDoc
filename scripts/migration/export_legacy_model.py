"""冻结实际旧RapidLayout/BatchAnalyze转换、过滤和任务裁图；固定响应与实际模型证据分开。"""
import copy,json,math,random
from io import BytesIO
from types import SimpleNamespace
from PIL import Image
import numpy as np
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-model.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel,get_cls_dicts
    from rapid_doc.model.layout.rapid_layout_self import ModelType
    from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
    from rapid_doc.backend.utils.utils import filter_overlap_boxes
    from rapid_doc.utils.model_utils import crop_img
    defaults=['number','footnote','header','header_image','footer','footer_image','aside_text']
    def normalized(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:normalized(x)for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [normalized(x)for x in v]
        return v
    def old_layout(request):
        regions=request['regions'];ignored=request['markdown_ignore_labels'];ignored=defaults if ignored is None else ignored
        wrapper=RapidLayoutModel.__new__(RapidLayoutModel);wrapper.docvortex_mode='off';wrapper.model_type=ModelType.PP_DOCLAYOUTV3
        wrapper.pp_doclayoutv2_cls_dict=get_cls_dicts(ignored)[2]
        result=SimpleNamespace(boxes=[r['coordinate']for r in regions],scores=[r['score']for r in regions],class_names=[r['label']for r in regions],orders=[r['order']for r in regions],polygon_points=None)
        wrapper.model=lambda **kwargs:[result]
        batch=BatchAnalyze.__new__(BatchAnalyze);batch.model=SimpleNamespace(layout_model=wrapper);batch.layout_base_batch_size=1
        batch.formula_enable=request['formula_enabled'];batch.formula_level=request['formula_level'];batch.seal_enable=False;batch.use_custom_ocr=request['use_custom_ocr'];batch.table_enable=True;batch.use_det_mode='auto'
        pages=[{}]if request['filter_overlaps']else[dict(_docvortex_layout=True)]
        return dict(layout_dets=normalized(batch._run_layout_detection([np.zeros((100,100,3),np.uint8)],pages,[1.])[0]))
    labels=list(get_cls_dicts([])[2]);cases=[];filters=[]
    def region(label='text',bbox=(10.,10.,50.,40.),score=.95,order=1):return dict(label=label,coordinate=list(bbox),score=score,order=order,cls_id=labels.index(label))
    def add(regions,ignored=None,formulas=True,level=0,custom=False,filter_=True):
        request=dict(regions=regions,layout_shape_mode='rect',markdown_ignore_labels=ignored,formula_enabled=formulas,formula_level=level,use_custom_ocr=custom,filter_overlaps=filter_)
        cases.append(dict(request=request,expected=old_layout(request)))
    for label in labels:
        for ignored in [None,[],[label]]:add([region(label)],ignored)
    for label in ['inline_formula','display_formula']:
        for formulas in [False,True]:
            for level in [0,1,2]:add([region(label)],formulas=formulas,level=level)
    for score in [.5005,.5015,.0005,.9995,math.nextafter(.5005,0),math.nextafter(.5005,1)]:add([region(score=score)])
    for threshold in [.5,.7]:
        for ratio in [math.nextafter(threshold,0),threshold,math.nextafter(threshold,1)]:
            for a,b in [('text','text'),('inline_formula','text'),('image','text'),('table','text'),('seal','chart')]:
                for custom in [False,True]:add([region(a,(0.,0.,100.,100.)),region(b,(100*(1-ratio),0.,200-100*ratio,100.),order=2)],custom=custom)
    for size in [5.999999999999999,6.,6.000000000000001]:add([region(bbox=(0.,0.,size,10.))])
    add([]);add([region('reference'),region('reference_content',order=2)])
    r=random.Random(2815)
    for _ in range(100):
        regions=[]
        for i in range(r.randrange(1,25)):
            x,y=r.randrange(100),r.randrange(100);w,h=r.randrange(1,80),r.randrange(1,80)
            regions.append(region(r.choice(labels),(float(x),float(y),float(x+w),float(y+h)),r.random(),i%5))
        add(regions,custom=r.choice([True,False]),filter_=r.choice([True,False]))
    for c in cases[::3]:
        dets=old_layout(dict(c['request'],filter_overlaps=False))['layout_dets']
        for i,d in enumerate(dets):d['unknown']=dict(index=i,integer=2**53+1,flag=True)
        request=dict(layout_dets=dets,use_custom_ocr=c['request']['use_custom_ocr']);filters.append(dict(request=request,expected=filter_overlap_boxes(dets,request['use_custom_ocr'])))
    raw=json.loads((ROOT/'build/migration/layout-post.json').read_text());real=[]
    for c in raw['png_cases']:
        request=dict(regions=c['expected'],layout_shape_mode='rect',markdown_ignore_labels=None,formula_enabled=True,formula_level=0,use_custom_ocr=False,filter_overlaps=True)
        real.append(dict(source=c['input'],sha256=digest(__import__('pathlib').Path(c['input']).read_bytes()),request=request,expected=old_layout(request)))
    folder=ROOT/'build/migration/legacy-model-crops';folder.mkdir(parents=True,exist_ok=True)
    arr=np.fromfunction(lambda y,x,k:(x*17+y*31+k*67)%256,(128,160,3),dtype=int).astype(np.uint8);Image.fromarray(arr).save(folder/'page.png')
    crops=[]
    def det(kind,box,label='text',order=0,**extra):
        x,y,r,b=box;return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,original_label=label,polygon_points=None,**extra)
    def crop_case(dets,expansion=2):
        model=dict(layout_dets=copy.deepcopy(dets));dets=model['layout_dets'];batch=BatchAnalyze.__new__(BatchAnalyze);batch.checkbox_enable=False;batch.formula_bbox_expand_px=expansion
        ocr,tables,formulas=batch._collect_detection_regions([dets],[arr],[(None,1.,False,None,{})])
        images=[];raw_uuid={};index=3
        def emit(a):
            if isinstance(a,Image.Image):a=np.asarray(a)
            key=f'model-{index}-{len(images)}';images.append(dict(key=key,width=a.shape[1],height=a.shape[0],pixels_sha256=digest(a.tobytes())));return key
        def idx(d):return next(i for i,v in enumerate(dets)if v is d)
        # Native get_res_list created image crops before the other task images.
        for i,img in enumerate(dets):
            if img['category_id']!=3:continue
            for j,table in enumerate(dets):
                for entry in table.get('layout_image_list',[]):
                    if entry['poly'] is img['poly']:
                        raw_uuid[entry['uuid']]=f'table-image-{index}-{i}-{j}'
                        entry['uuid']=raw_uuid[entry['uuid']];entry['source']=emit(entry.pop('pil_image'))
        table_tasks=[dict(layout_index=idx(t['table_res']),source=emit(t['table_img']),useful_list=t['useful_list'])for t in tables]
        formula_tasks=[]
        for f in formulas:
            from rapid_doc.backend.utils.utils import _expand_formula_crop_res
            res=_expand_formula_crop_res(f['formula_res'],dets,arr.shape,expansion);_,useful=crop_img(res,arr)
            formula_tasks.append(dict(layout_index=idx(f['formula_res']),source=emit(f['formula_img']),useful_list=useful))
        text_tasks=[]
        for d in ocr[0]['ocr_res_list']:
            pixels,useful=crop_img(d,arr,crop_paste_x=50,crop_paste_y=50);text_tasks.append(dict(layout_index=idx(d),source=emit(pixels),useful_list=useful))
        expected=dict(model=model,ocr=text_tasks,tables=table_tasks,formulas=formula_tasks)
        crops.append(dict(source=(folder/'page.png').relative_to(ROOT).as_posix(),request=dict(model=dict(layout_dets=copy.deepcopy(original[0])),page_index=index,formula_bbox_expand_px=expansion),expected=normalized(expected),images=images,uuid_projection=raw_uuid))
    original=[None]
    def add_crop(dets,expansion=2):original[0]=dets;crop_case(dets,expansion)
    add_crop([])
    for expansion in [0,1,2,10]:
        add_crop([det(1,[5.9,10.1,40.9,20.9]),det(5,[50.1,30.2,149.9,121.9],'table'),det(14,[5.5,40.5,24.5,49.5],'display_formula'),det(3,[60.,40.,80.,60.],'image'),det(13,[35.5,40.5,49.5,49.5],'inline_formula')],expansion)
    for threshold in [.7999999999999999,.8,.8000000000000002,1.]:
        add_crop([det(3,[0.,0.,20.,20.],'image'),det(5,[20*(1-threshold),0.,100.,100.],'table')])
    add_crop([det(5,[0.,0.,160.,128.],'table',html='<table><td>existing</td></table>'),det(3,[2.,2.,20.,20.],'image')])
    for _ in range(50):
        dets=[]
        for i in range(r.randrange(1,15)):
            x,y=r.uniform(1,110),r.uniform(1,85);w,h=r.uniform(7,30),r.uniform(7,30)
            kind=r.choice([0,1,2,3,5,8,13,14]);dets.append(det(kind,[x,y,x+w,y+h],order=i))
        add_crop(dets,r.randrange(6))
    refs=['rapid_doc/model/layout/rapid_layout.py','rapid_doc/backend/pipeline/batch_analyze.py','rapid_doc/backend/utils/utils.py','rapid_doc/utils/model_utils.py','rapid_doc/utils/bbox_utils.py']
    write_json(target,dict(cases=cases,filters=filters,real=real,crops=crops,model=raw['model'],model_sha256=raw['sha256'],sources={p:digest((ROOT/p).read_bytes())for p in refs},profile='固定regions隔离实际旧wrapper/BatchAnalyze；3真实ONNX模型后处理结果另列；裁图运行实际_collect_detection_regions，PIL/ndarray换ID+独立原RGB哈希，随机UUID原值保留并投影稳定ID，rect限定，不含完整OCR/表格/公式执行器。'))
    print(f'Frozen {len(cases)} layout conversions/{len(filters)} filters, {len(real)} real model layouts, {len(crops)} crop plans/{sum(len(c["images"])for c in crops)} RGB crops')

if __name__=='__main__':main()
