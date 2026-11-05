"""冻结旧 block_sort 模型顺序分支完整输出；拒绝覆盖参考文件。"""
import argparse,copy,importlib,json,random,sys
from common import ROOT,digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='build/migration/legacy-block-sort.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Baseline exists; use a new --output')
    module=importlib.import_module('rapid_doc.utils.block_sort');cases=[];coverage={}
    filename=module.sort_blocks_by_bbox.__code__.co_filename
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=filename:return None
        if event=='line':coverage.setdefault(frame.f_code.co_name,set()).add(frame.f_lineno)
        return trace
    def block(kind='text',order=0,bbox=None,lines=None,**extra):
        box=bbox or [0,0,200,60]
        return dict(type=kind,original_order=order,bbox=box,
                    lines=lines if lines is not None else [dict(bbox=[box[0],box[1],box[2],box[1]+10],spans=[dict(type='text',content='body',custom=[None,False])],custom='line')],
                    custom=dict(unknown='保留'),**extra)
    def add(name,blocks,width=600,height=800,footnotes=None):
        request=dict(blocks=blocks,page_width=width,page_height=height,footnote_blocks=footnotes or [])
        copied=copy.deepcopy(blocks)
        sys.settrace(trace)
        try:expected=module.sort_blocks_by_bbox(copied,width,height,request['footnote_blocks'],None)
        finally:sys.settrace(None)
        cases.append(dict(name=name,request=request,expected=expected))
    add('missing-sentinel-is-block-count',[block(order=100),block(order=None),block(order=0),block(order=-1)])
    add('stable-ties',[block(order=2),block(order=2.),block(order=True),block(order=1),block(order=-0.),block(order=0)])
    add('span-min-and-block-priority',[block(order=-1,lines=[dict(bbox=[0,0,100,12],spans=[dict(original_order=9),dict(original_order=2.),dict(original_order=2),dict(original_order=-1)])]),block(order=12,lines=[dict(bbox=[0,0,100,12],spans=[dict(original_order=0)])])])
    add('zero-clamps-type',[block(bbox=[-2.,-0.,200.,60.]),block(bbox=[0.,0,100,20])])
    for kind in ['title','text','image_body','table_body','interline_equation','image_caption','table_footnote','unknown']:
        for width in [100,150,150.00001,240,240.00001,299,300,301,500]:
            for height in [20,20.00001,60,200,200.00001,220]:
                for empty in [True,False]:
                    kwargs=dict(group_id='g')if kind.startswith(('image_','table_'))else {}
                    add(f'virtual-{kind}-{width}-{height}-{empty}',[block(kind,bbox=[0,0,width,height],lines=[]if empty else None,**kwargs)],footnotes=[[1,2,10,12,'ignored']])
    add('groups-median-ties-first-body',[block('table_caption',4,group_id=2),block('image_caption',4,group_id='x'),block('text',4),block('image_body',2,group_id='x',polygon_points=[[0,0],[1,1]]),block('image_body',100,group_id='x'),block('table_footnote',4,group_id=2)])
    add('groups-python-equal-keys',[block('image_caption',True,group_id=True),block('image_body',1.,group_id=1.),block('table_caption',0,group_id=None),block('image_footnote',False,group_id=1)])
    add('preexisting-real-lines',[block('title',lines=[],real_lines=[dict(bbox=[0,0,100,10],spans=[dict(original_order=7)])])])
    rng=random.Random(71024)
    kinds=['text','title','image_body','image_caption','image_footnote','table_body','table_caption','table_footnote','interline_equation','unknown']
    for i in range(300):
        blocks=[]
        for j in range(rng.randint(1,15)):
            kind=rng.choice(kinds);x=rng.choice([-12.,0,10.25]);y=rng.choice([-0.,0,80.5]);w=rng.choice([100,150,240,300,500]);h=rng.choice([20.,50.,250.333333333])
            b=block(kind,order=rng.choice([None,-1,0,1,2,20.,True]),bbox=[x,y,x+w,y+h],lines=[]if rng.random()<.3 else None)
            if kind.startswith(('image_','table_')):b['group_id']=rng.choice([None,1,2,'x'])
            if rng.random()<.3:b['polygon_points']=rng.choice([[],None,[[1,2],[3,4]]])
            blocks.append(b)
        blocks[0]['original_order']=rng.choice([0,2,100])
        add(f'random-{i}',blocks)
    # 复用上轮真实 CPU OCR/PDF 输入的旧格式适配，而非只比较作者几何样本。
    previous=json.loads((ROOT/'build/migration/legacy-paragraphs.json').read_text())
    for c in previous['cases']:
        if c['name'].startswith('audit/'):
            for pi,p in enumerate(c['pages']):
                blocks=copy.deepcopy(p['preproc_blocks'])
                if not blocks:continue
                for bi,b in enumerate(blocks):b['original_order']=bi;b.setdefault('lines',[])
                add('paragraph-input/'+c['name']+'/'+str(pi),blocks,p['page_size'][0],p['page_size'][1])
    source='rapid_doc/utils/block_sort.py'
    write_json(target,dict(source=source,source_sha256=digest((ROOT/source).read_bytes()),cases=cases,
                           executed_lines={k:sorted(v)for k,v in coverage.items()},profile='模型顺序分支；正宽高 bbox；JSON 副本接口；无模型顺序不进入参考集'))
    print(f'Frozen {len(cases)} complete legacy block_sort outputs; '+str({k:len(v)for k,v in coverage.items()}))


if __name__=='__main__':main()
