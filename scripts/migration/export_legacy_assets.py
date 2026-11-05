"""冻结实际旧单页 writer 的页面 JSON、写入顺序、文件名及解码后的 PNG 像素。"""
import argparse,copy,hashlib,importlib,json
from io import BytesIO
from PIL import Image
from common import ROOT,digest,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='build/migration/legacy-assets.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Frozen baseline exists; use a new --output')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');cases=[]
    folder=ROOT/'build/migration/legacy-asset-images';folder.mkdir(parents=True,exist_ok=True)
    def make(mode,width=48,height=40,seed=0):
        channels={'L':1,'LA':2,'RGB':3,'RGBA':4}[mode]
        data=bytes((x*17+y*37+c*61+seed*13)%256 for y in range(height)for x in range(width)for c in range(channels))
        image=Image.frombytes(mode,(width,height),data);p=folder/f'{mode}-{width}-{height}-{seed}.png'
        if not p.exists():image.save(p)
        image.close();return p.relative_to(ROOT).as_posix()
    def det(kind,bbox,order=0,scale=1.,**extra):
        x,y,r,b=[v*scale for v in bbox]
        return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,latex='x+y',html='<table><tr><td>cell</td></tr></table>'if kind==5 else '',**extra)
    class Writer:
        def __init__(self):self.calls=[]
        def write(self,name,data):
            with Image.open(BytesIO(data))as img:
                img.load();self.calls.append(dict(name=name,mode=img.mode,width=img.width,height=img.height,pixels_sha256=digest(img.tobytes())))
    def add(name,source,dets,scale=1.,width=None,height=None,originals=None,extract=False,threshold=.9):
        image=Image.open(ROOT/source);image.load();originals=originals or [];opened=[]
        try:
            origin=[]
            for entry in originals:
                img=Image.open(ROOT/entry['source']);img.load();opened.append(img);origin.append(dict(bbox=entry['bbox'],pil_image=img))
            request=dict(model=dict(layout_dets=dets),page_width=width if width is not None else image.width/scale,page_height=height if height is not None else image.height/scale,
                         page_index=7,scale=scale,formula_enabled=True)
            writer=Writer()
            expected=old.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=scale,img_pil=image),dict(size=[request['page_width'],request['page_height']],ori_image_list=origin),writer,7,
                ocr_enable=True,image_config=dict(extract_original_image=extract,extract_original_image_iou_thresh=threshold))
            cases.append(dict(name=name,source=source,source_sha256=digest((ROOT/source).read_bytes()),request=request,
                originals=originals,extract_original=extract,threshold=threshold,expected=expected,calls=writer.calls))
        finally:
            image.close()
            for img in opened:img.close()
    boxes=[[2,3,20,22],[-3,-4,20,22],[40,30,60,50],[-30,-30,-20,-20],[55,45,63,55],[.9,1.9,20.1,21.1],[1.2,2.3,14.9,12.8],[-.6,-.6,15.4,14.2],[0,0,48,40]]
    for mode in ['L','LA','RGB','RGBA']:
        source=make(mode)
        for kind in [3,5,14]:
            for scale in [.5,1.,2.]:
                for i,bbox in enumerate(boxes):add(f'crop/{mode}/{kind}/{scale}/{i}',source,[det(kind,bbox,scale=scale)],scale)
    source=make('RGB');bbox=[2,3,20,22]
    origins=[dict(bbox=bbox,source=make('RGB',9,6,1)),dict(bbox=bbox,source=make('RGBA',17,13,2))]
    for kind in [3,5,14]:add(f'original/last-match-{kind}',source,[det(kind,bbox)],originals=origins,extract=True)
    add('original/lower-iou-last-match',source,[det(3,bbox)],originals=[origins[0],dict(bbox=[2,3,21,22],source=make('LA',10,11,3))],extract=True,threshold=.9)
    add('original/zero-threshold',source,[det(3,bbox)],originals=[dict(bbox=[100,100,110,110],source=make('L',8,9,4))],extract=True,threshold=0.)
    add('original/not-matched',source,[det(3,bbox)],originals=[dict(bbox=[100,100,110,110],source=make('L',8,9,4))],extract=True)
    add('written-before-fill-even-if-unassigned',source,[det(3,bbox),det(2,bbox,1)])
    add('duplicate-name-retains-write-order',source,[det(14,[1.1,2.1,1.6,2.6],0,scale=10.),det(14,[1.4,2.4,1.9,2.9],1,scale=10.)],scale=10.)
    add('empty-page-writes-none',source,[])
    previous=json.loads((ROOT/'build/migration/legacy-page.json').read_text())
    for name in ['group-3-2','group-5-2','formula-True','native-image-foreground/seal']:
        c=next(c for c in previous['cases']if c['name']==name);r=c['request']
        add('full-page/'+name,make('RGB',600,800),copy.deepcopy(r['model']['layout_dets']),width=600,height=800)
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/utils/cut_image.py','rapid_doc/utils/pdf_image_tools.py','rapid_doc/utils/hash_utils.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in sources},
        profile='实际旧单页函数和 writer；8-bit L/LA/RGB/RGBA PNG；页面 JSON/文件名/写入顺序/解码像素精确对照，PNG 压缩字节不承诺相同；原图最后匹配优先，不含内联/表内图片与 PDF 原图提取'))
    print(f'Frozen {len(cases)} complete old asset-page references ({sum(len(c["calls"])for c in cases)} writer calls)')


if __name__=='__main__':main()
