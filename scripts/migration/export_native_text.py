"""冻结未修改旧 txt_spans_extract、完整页面及补 OCR 原始像素；不覆盖基线。"""
import argparse,copy,importlib,json,math
from io import BytesIO
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='build/migration/legacy-native-text.json');parser.add_argument('--narrow',action='store_true');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');sp=importlib.import_module('rapid_doc.utils.span_pre_proc');contrast=sp.calculate_contrast
    folder=ROOT/'build/migration/native-text-images';folder.mkdir(parents=True,exist_ok=True)
    def image(name,w=100,h=100,a=0,b=255):
        values=np.zeros((h,w,3),dtype=np.uint8)
        for y in range(h):
            for x in range(w):values[y,x]=a if (x+y)%2 else b
        path=folder/(name+'.png');Image.fromarray(values).save(path);return path.relative_to(ROOT).as_posix()
    high=image('high');low=image('low',a=200,b=200);boundary=image('boundary',a=100,b=140)
    def span(bbox=None,content='',**extra):return dict(type='text',bbox=list(bbox if bbox is not None else (10,10,30 if args.narrow else 70,20)),score=.95,content=content,**extra)
    def block(bbox=(0,0,100,100),kind='text'):return [*bbox,None,None,None,kind]
    def char(text,bbox=(12,11,18,19),index=0,**extra):return dict(char=text,bbox=list(bbox),char_idx=index,**extra)
    def native(chars=None,rotation=0,bbox=(0,0,100,100),text=None):
        chars=chars or [];return dict(blocks=[dict(lines=[dict(rotation=rotation,bbox=list(bbox),spans=[dict(text=text if text is not None else ''.join(c['char']for c in chars),chars=chars)])])])
    def adapt(page):
        page=copy.deepcopy(page)
        for b in page['blocks']:
            for line in b['lines']:line['bbox']=SimpleNamespace(bbox=line['bbox'])
        return page
    cases=[];pages=[]
    def run(name,request,source,full=False,parent='images'):
        accepted=[];assets=[]
        def capture(img,img_mode):
            value=contrast(img,img_mode)
            if value>=.17:accepted.append(img)
            return value
        def normalize(value):
            if isinstance(value,np.ndarray):
                return 'ocr-7-'+str(next(i for i,a in enumerate(accepted)if a is value))
            if isinstance(value,dict):return {k:normalize(v)for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [normalize(v)for v in value]
            return value
        class Writer:
            _parent_dir=parent
            def write(self,name,data):
                with Image.open(BytesIO(data))as img:img.load();assets.append(dict(name=name,mode=img.mode,width=img.width,height=img.height,pixels_sha256=digest(img.tobytes())))
        with Image.open(ROOT/source)as img:
            try:
                sp.calculate_contrast=capture
                if full:
                    pd=adapt(request['native_page']);pd.update(size=[request['page_width'],request['page_height']],ori_image_list=[])
                    expected=old.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=request['scale'],img_pil=img),pd,Writer(),7,ocr_enable=False)
                else:expected=sp.txt_spans_extract(adapt(request['native_page']),copy.deepcopy(request['spans']),img,request['scale'],copy.deepcopy(request['blocks']),copy.deepcopy(request['discarded_blocks']))
                expected=normalize(expected)
            finally:sp.calculate_contrast=contrast
        crops=[dict(name='ocr-7-'+str(i),width=a.shape[1],height=a.shape[0],rgb_sha256=digest(a[:,:,::-1].copy().tobytes()))for i,a in enumerate(accepted)]
        entry=dict(name=name,request=request,source=source,expected=expected,crops=crops,assets=assets)
        (pages if full else cases).append(entry)
    def add(name,spans,chars=None,blocks=None,discarded=None,source=high,scale=1.,page=None):
        request=dict(spans=spans,native_page=page or native(chars),blocks=[block()]if blocks is None else blocks,discarded_blocks=discarded or [],scale=scale,page_index=7)
        run(name,request,source)
    add('basic',[span()],[char('A'),char('B',(19,11,25,19),1)])
    for gap in [0,1,1.5,1.500000000000001,2,10]:
        for explicit in [False,True]:
            chars=[char('A'),char('B',(18+gap,11,24+gap,19),2)]
            if explicit:chars.insert(1,char('\u00a0',(18,11,18.5,19),1))
            add('gap/'+str(gap)+'/'+str(explicit),[span()],chars)
    for text in ['ﬁﬂﬀﬃﬄﬅﬆ','Ａβ中文','A\r\nB','A\u0002B','\t\u001cA\u3000','',')','(','.']:
        add('unicode/'+repr(text),[span()],[char(text)])
    for text in ['.',')','(', '[', 'x','"']:
        for bbox in [[25,11,45,19]if args.narrow else [65,11,85,19],[0,11,15,19],[12,20,18,28],[12,9.3,18,10.7],[12,12.3,18,13.7]]:
            add('boundary/'+str(len(cases)),[span()],[char(text,bbox)])
    for rotation in [None,0,math.pi/2,math.pi,3*math.pi/2,math.pi/4,math.radians(.1),math.radians(.099999999),math.radians(90.1),math.radians(-.1),'bad',False]:
        add('rotation/'+repr(rotation),[span()],[char('A')],page=native([char('A')],rotation))
    add('duplicate-character-dicts',[span()],[char('A'),char('A'),char('B',(40,11,46,19),2)])
    add('char-index-order',[span()],[char('B',(40,11,46,19),2),char('A')])
    add('first-span-wins',[span(),span((11,10,71,20))],[char('A')])
    add('native-bypass',[span(docvortex_native_text=True,content='native')],[char('A')])
    add('outside-keeps-dimensions',[span((110,10,150,20))],[char('A')])
    add('excluded-image-body',[span()],[char('A')],blocks=[block(kind='image_body')])
    add('discarded',[span()],[char('A')],blocks=[],discarded=[block(kind='discarded')])
    for bbox in [[-20,0,-1,20],[-10,0,20,20],[-.9,-.9,20.9,20.9],[90,90,120,120],[110,10,130,20],[0,0,100,100]]:
        for scale in [.5,1.,2.]:add('numpy-crop/'+str(len(cases)),[span(bbox)],blocks=[block(bbox)],scale=scale)
    for source in [high,low,boundary]:add('contrast/'+source,[span()],source=source)
    for text in ['', 'already known']:
        add('vertical/'+text,[span((10,10,15,90),content=text),span((30,10,80,20)),span((30,30,80,40))],page=native([],bbox=(10,10,15,40),text=text),source=low)
    add('empty',[],[])
    def det(kind,bbox,order=0,**extra):
        x,y,r,b=bbox;return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,**extra)
    for source in [high,low,boundary]:
        for chars in [[],[char('A')],[char('中'),char('文',(19,11,25,19),1)]]:
            for kind in [1,0,2]:
                request=dict(model=dict(layout_dets=[det(kind,[10,10,70,20]),det(15,[10,10,70,20],text='')]),native_page=native(chars),page_width=100,page_height=100,page_index=7,scale=1.,formula_enabled=True)
                run('page/'+str(len(pages)),request,source,True)
    # 使用真实 PDF 的旧字符事实，避免只验证手造 bbox。
    from pdf_corpus import generate_pdfs
    import pypdfium2 as pdfium
    from rapid_doc.utils.pdf_text_tool import get_page
    for name,data in generate_pdfs().items():
        with pdfium.PdfDocument(data)as doc:
            for i in range(len(doc)):
                p=doc[i]
                try:
                    facts=get_page(p);normalized=dict(blocks=[])
                    for b in facts['blocks']:
                        lines=[]
                        for line in b['lines']:
                            spans=[]
                            for s in line['spans']:
                                chars=[]
                                for c in s['chars']:
                                    c=dict(c);c['bbox']=list(getattr(c['bbox'],'bbox',c['bbox']));chars.append(c)
                                spans.append(dict(text=s['text'],chars=chars))
                            lines.append(dict(bbox=list(line['bbox'].bbox),rotation=line.get('rotation',0),spans=spans))
                        normalized['blocks'].append(dict(lines=lines))
                    bitmap=p.render(scale=1.)
                    try:pil=bitmap.to_pil().convert('RGB')
                    finally:bitmap.close()
                    path=folder/f'real-{name}-{i}.png';pil.save(path);pil.close()
                    boxes=[line['bbox']for b in normalized['blocks']for line in b['lines']][:20]
                    if not boxes:boxes=[[10,10,100,30]]
                    dets=[]
                    for j,box in enumerate(boxes):dets.extend([det(1,box,j),det(15,box,j,text='')])
                    request=dict(model=dict(layout_dets=dets),native_page=normalized,page_width=400,page_height=400,page_index=7,scale=1.,formula_enabled=True)
                    run('real-pdf/'+name+'/'+str(i),request,path.relative_to(ROOT).as_posix(),True)
                finally:p.close()
    if args.narrow:pages=[]
    refs=['rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/pdf_image_tools.py','rapid_doc/utils/pdf_text_tool.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py']
    write_json(target,dict(cases=cases,pages=pages,inputs={c['source']:digest((ROOT/c['source']).read_bytes())for c in cases+pages},sources={p:digest((ROOT/p).read_bytes())for p in refs},numpy=np.__version__,profile='未修改旧函数；仅 bbox 对象换成等坐标 JSON，np_img 像素负载按顺序映射为 ocr-7-N 并独立记录 RGB 哈希；不归一化内容/字段/裁图决策'))
    print(f'Frozen {len(cases)} native span fills and {len(pages)} complete old native-text pages')

if __name__=='__main__':main()
