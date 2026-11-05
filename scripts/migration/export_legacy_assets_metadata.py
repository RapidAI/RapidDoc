"""实际旧 writer 的 ICC/EXIF/透明色补充参考；保持旧 PNG 模式和页面哈希。"""
import copy,importlib
from io import BytesIO
from PIL import Image,ImageCms
from common import ROOT,digest,write_json


def main():
    target=ROOT/'build/migration/legacy-assets-metadata.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json');cases=[]
    folder=ROOT/'build/migration/legacy-asset-images';folder.mkdir(parents=True,exist_ok=True)
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    exif=Image.Exif();exif[274]=6;exif[305]='RapidDoc migration'
    def make(mode,info,label):
        channels={'L':1,'RGB':3,'RGBA':4}[mode]
        image=Image.frombytes(mode,(40,30),bytes((x*17+y*13+c*31)%256 for y in range(30)for x in range(40)for c in range(channels)))
        path=folder/f'metadata-{label}.png';image.save(path,**info);image.close();return path.relative_to(ROOT).as_posix()
    plain=make('RGB',{},'plain')
    class Writer:
        def __init__(self):self.calls=[]
        def write(self,name,data):
            with Image.open(BytesIO(data))as img:
                img.load();call=dict(name=name,mode=img.mode,width=img.width,height=img.height,pixels_sha256=digest(img.tobytes()))
                for k in ['icc_profile','exif']:
                    if img.info.get(k):call[k+'_sha256']=digest(img.info[k])
                if 'transparency'in img.info:call['transparency']=list(img.info['transparency'])if isinstance(img.info['transparency'],tuple)else img.info['transparency']
                self.calls.append(call)
    def add(name,source,bbox,original=None):
        image=Image.open(ROOT/source);image.load();opened=Image.open(ROOT/original)if original else None
        try:
            if opened:opened.load()
            x,y,r,b=bbox;request=dict(model=dict(layout_dets=[dict(category_id=3,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=0)]),
                page_width=40,page_height=30,page_index=7,scale=1.,formula_enabled=True)
            writer=Writer();originals=[dict(bbox=bbox,pil_image=opened)]if opened else []
            expected=old.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=1.,img_pil=image),dict(size=[40,30],ori_image_list=originals),writer,7,
                ocr_enable=True,image_config=dict(extract_original_image=bool(opened)))
            cases.append(dict(name=name,source=source,source_sha256=digest((ROOT/source).read_bytes()),request=request,
                originals=[dict(bbox=bbox,source=original)]if original else [],extract_original=bool(original),threshold=.9,expected=expected,calls=writer.calls))
        finally:
            image.close()
            if opened:opened.close()
    for mode in ['RGB','RGBA']:
        source=make(mode,dict(icc_profile=profile,exif=exif.tobytes()),mode+'-icc-exif')
        add(mode+'/crop-icc-exif',source,[-3,-4,20,22]);add(mode+'/original-icc-exif',plain,[2,3,20,22],source)
    for mode,colors in [('RGB',[(0,0,0),(17,31,62)]),('L',[0,17])]:
        for i,color in enumerate(colors):
            source=make(mode,dict(transparency=color),mode+'-transparent-'+str(i))
            add(mode+'/transparent/'+str(i),source,[-3,-4,20,22])
        add(mode+'/original-transparent',plain,[2,3,20,22],source)
    source=make('RGB',dict(icc_profile=profile,exif=exif.tobytes(),transparency=(0,0,0)),'combined')
    add('combined-metadata',source,[-3,-4,20,22])
    sources=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/utils/cut_image.py','rapid_doc/utils/pdf_image_tools.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in sources},profile='实际旧 writer，ICC profile/EXIF 字节哈希、RGB/L 透明色、PNG模式/像素、页面 MD5/SHA256名称；不比较 PNG 压缩字节'))
    print(f'Frozen {len(cases)} old ICC/EXIF/transparent-color asset references')


if __name__=='__main__':main()
