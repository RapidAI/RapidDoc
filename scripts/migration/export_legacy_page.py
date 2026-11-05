"""运行未修改的旧单页模型转换函数，冻结 OCR 已完成、writer=None 的完整 JSON。"""
import argparse,copy,importlib,json,random,sys
from PIL import Image
from common import ROOT,digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='build/migration/legacy-page.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Frozen baseline exists; use a new --output')
    module=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/pipeline_magic_model.py',
             'rapid_doc/utils/magic_model_utils.py','rapid_doc/utils/block_pre_proc.py','rapid_doc/utils/span_pre_proc.py',
             'rapid_doc/utils/span_block_fix.py','rapid_doc/utils/block_sort.py','rapid_doc/utils/boxbase.py','rapid_doc/utils/ocr_utils.py']
    files={str(ROOT/p)for p in sources};coverage={};cases=[]
    def trace(frame,event,arg):
        if frame.f_code.co_filename.replace('\\','/') not in {f.replace('\\','/')for f in files}:return None
        if event=='line':coverage.setdefault(frame.f_code.co_name,set()).add(frame.f_lineno)
        return trace
    def det(cat,box,order=0,**extra):
        x,y,r,b=box
        return dict(category_id=cat,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,**extra)
    def add(name,dets,scale=1.,width=600,height=800,formula=True,index=0):
        request=dict(model=dict(layout_dets=dets),page_width=width,page_height=height,page_index=index,scale=scale,formula_enabled=formula)
        image=Image.new('RGB',(int(width*scale),int(height*scale)),color='white')
        sys.settrace(trace)
        try:
            expected=module.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=scale,img_pil=image),
                dict(size=[width,height],ori_image_list=[]),None,index,ocr_enable=True,formula_enabled=formula)
        finally:sys.settrace(None);image.close()
        cases.append(dict(name=name,request=request,expected=expected))
    add('empty',[])
    add('discarded-only',[det(2,[10,10,200,30]),det(15,[10,10,200,30],text='discarded')])
    add('outside-ocr-only',[det(15,[10,10,200,30],text='no owner')])
    add('text-title-and-checkbox',[det(0,[10,10,200,35],0),det(1,[10,50,300,160],1),det(15,[10,10,200,30],0,text='Title'),
        det(15,[10,50,150,65],1,text='Body'),det(200,[160,50,175,65],checkbox='☑'),det(15,[10,90,250,105],1,text='second line')])
    for cat,caption,footnote in [(3,4,101),(5,6,7)]:
        body=dict(html='<table><tr><td>cell</td></tr></table>')if cat==5 else {}
        for count in [0,1,2]:
            ds=[det(cat,[100,100,400,300],0,polygon_points=[[100,100],[400,100],[400,300],[100,300]],**body)]
            for i in range(count):ds.extend([det(caption,[100,310+i*40,400,330+i*40],2+i),det(15,[105,311+i*40,395,328+i*40],2+i,text=f'Caption {i}')])
            if count:ds.extend([det(footnote,[100,70,400,90],1),det(15,[105,72,395,88],1,text='Footnote')])
            add(f'group-{cat}-{count}',ds)
    for original in ['seal','image',None]:
        add('native-image-foreground/'+str(original),[det(3,[50,50,500,500],0,original_label=original,text='印章'),
            det(1,[100,100,200,130],1,native_text_region=True),det(15,[50,50,400,400],1,text='native foreground',docvortex_native_text=True)])
    for formula in [True,False]:
        add(f'formula-{formula}',[det(1,[0,0,500,90],0),det(15,[10,10,90,30],0,text='before'),det(13,[100,10,120,30],latex='x^2'),
            det(8,[0,100,300,140],1),det(14,[0,100,300,140],1,latex='a+b=c'),det(9,[310,100,350,140],2),det(15,[310,100,350,140],2,text='(1)')],formula=formula)
        add(f'layout-formula-only-{formula}',[det(8,[10,10,200,80],0)],formula=formula)
    # 竖排比率 2、块内竖排占比 0.8 与同行重叠 0.6 的严格边界。
    for vertical in [3,4,5]:
        ds=[det(1,[10,10,200,350],0)]
        for i in range(5):
            box=[150-i*20,10,160-i*20,60]if i<vertical else [20,100+i*30,180,115+i*30]
            ds.append(det(15,box,0,text=f'span {i}'))
        add(f'vertical-{vertical}',ds)
    for delta in [3.999999999,4.,4.000000001]:
        add(f'line-threshold-{delta}',[det(1,[0,0,300,60],0),det(15,[0,0,100,10],0,text='left'),det(15,[120,delta,240,delta+10],0,text='right')])
    for score in [.049999999,.05,.050000001,.8,.95]:
        add(f'confidence-{score}',[det(1,[0,0,300,100],0),dict(det(15,[10,10,200,30],0,text='confidence'),score=score)])
    for shift in [0,1,2,5,10]:
        ds=[det(1,[10,10,300,200],0),det(0,[10+shift,10+shift,300,200],1),
            det(15,[20,30,150,45],0,text='winner'),dict(det(15,[20+shift,30,150,45],1,text='other'),score=.8)]
        add(f'dedup-{shift}',ds)
    for scale in [.5,1.,1.25,1.333333333,2.]:
        ds=[det(1,[10.005,10.675,400.555,180.145],0,polygon_points=[[10.005,10.675],[400.555,10.675],[400.555,180.145]]),
            det(15,[20.005,30.675,200.555,50.145],0,text='scaled')]
        add(f'scale-{scale}',ds,scale=scale,index=7)
    rng=random.Random(24831)
    for i in range(240):
        ds=[]
        for j in range(rng.randint(1,8)):
            cat=rng.choice([0,1,1,1,2,3,4,5,6,7,8,9,101]);x=rng.choice([0,10,150,310]);y=rng.randrange(0,660);w=rng.choice([80,160,240]);h=rng.choice([20,60,100])
            ds.append(det(cat,[x,y,x+w,y+h],j,original_label=rng.choice([None,'text','seal','algorithm']),html='<table><tr><td>x</td></tr></table>'if cat==5 else ''))
            for k in range(rng.randint(0,3)):
                sy=y+k*15;ds.append(det(rng.choice([15,15,13,200]),[x+2,sy+2,x+w-2,sy+14],j,text=rng.choice(['body','中文','1. item','tail.']),latex='x^2',checkbox='☐'))
        add(f'random-{i}',ds,formula=rng.choice([True,False]))
    old=json.loads((ROOT/'build/migration/legacy-paragraph-boundaries.json').read_text())
    for c in old['cases']:
        if not c['name'].startswith('cpu-ocr-'):continue
        p=c['pages'][0];width,height=p['page_size'];ds=[det(1,[0,0,width,height],0)]
        for line in p['preproc_blocks'][0]['lines']:
            ds.append(det(15,line['bbox'],0,text=line['spans'][0]['content']))
        add('real/'+c['name'],ds,width=width,height=height)
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in sources},cases=cases,
        executed_lines={k:sorted(v)for k,v in coverage.items()},
        profile='实际旧 page_model_info_to_page_info，ocr_enable=True、image_writer=None、无 VL/原生字符填充/资产；模型 original_order；未执行跨页表格与公式编号后处理'))
    print(f'Frozen {len(cases)} complete old page outputs; source functions: {len(coverage)}')


if __name__=='__main__':main()
