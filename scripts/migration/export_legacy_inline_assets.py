"""冻结未修改旧 HTML 内联图片工具及完整单页 writer 输出。"""
import argparse,base64,copy,hashlib,importlib,itertools,json
from io import BytesIO
from PIL import Image
from loguru import logger
from common import ROOT,digest,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='build/migration/legacy-inline-assets.json');args=p.parse_args()
    target=ROOT/args.output
    if target.exists():p.error('Frozen baseline exists; use a new --output')
    logger.disable('rapid_doc.backend.utils.html_image_utils')
    old=importlib.import_module('rapid_doc.backend.utils.html_image_utils')
    pipeline=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    class Slots:
        __slots__=('calls',)
        def __init__(self):self.calls=[]
        def write(self,name,data):self.calls.append(dict(name=name,bytes_hex=data.hex()))
    class Writer:
        def __init__(self):self.calls=[]
        def write(self,name,data):self.calls.append(dict(name=name,sha256=digest(data)))
    cases=[]
    def add(markup,prefix='images/'):
        sink=Slots();expected=old.replace_inline_base64_img_src(markup,sink,7,prefix)
        cases.append(dict(markup=markup,prefix=prefix,expected=expected,calls=sink.calls))
    for length in range(6):
        for chars in itertools.product('AB/=!\n',repeat=length):add('<img src="data:image/png;base64,'+''.join(chars)+'">')
    for length in range(65):
        b64=base64.b64encode(bytes((i*37)%256 for i in range(length))).decode()
        for fmt in ['PNG','jpeg','gif','svg+xml','webp']:
            add('前<img src="data:image/'+fmt+';base64,'+b64+'">后',prefix='目录/')
    uri='data:image/png;base64,YWJjZA=='
    for markup in [f'<img src="{uri}"><img src="{uri}">',f"<img src='{uri}'>",f'<img SRC="{uri}">',f'<img src ="{uri}">',f'<img src="{uri}',f'<x data-src="{uri}">',f'<img src="{uri}废">',f'<img src="data:image/png;foo;base64,YQ==">',f'<img src="data:image/png;base64,YQ==\nmore">','',uri]:add(markup,'')
    source='build/migration/legacy-asset-images/RGB-48-40-0.png'
    pages=[]
    def det(kind,html,score=.95):return dict(category_id=kind,poly=[2,3,20,3,20,22,2,22],score=score,original_order=0,html=html)
    for kind in [2,3,5,99]:
        for html in [f'<table><td><img src="{uri}"></td></table>',f'<img src="{uri}"><img src="{uri}">','<img src="data:image/png;base64,bad">']:
            model=dict(layout_dets=[det(kind,html)])
            if kind==99:model['layout_dets'].append(det(5,'<table><td>x</td></table>'))
            request=dict(model=model,page_width=48,page_height=40,page_index=7,scale=1.,formula_enabled=True)
            class PageWriter(Writer):
                def write(self,name,data):
                    if len(name)==68 and name.endswith('.png') and data.startswith(b'\x89PNG'):
                        with Image.open(BytesIO(data))as image:
                            self.calls.append(dict(name=name,pixels_sha256=digest(image.tobytes()),mode=image.mode,width=image.width,height=image.height))
                    else:super().write(name,data)
            writer=PageWriter()
            with Image.open(ROOT/source)as image:
                expected=pipeline.page_model_info_to_page_info(copy.deepcopy(model),dict(scale=1.,img_pil=image),dict(size=[48,40],ori_image_list=[]),writer,7,ocr_enable=True)
            pages.append(dict(request=request,expected=expected,calls=writer.calls))
    paths=['rapid_doc/backend/utils/html_image_utils.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/_vendor/docvortex_core/analyzers/native/office/image.py']
    write_json(target,dict(cases=cases,pages=pages,source=source,source_sha256=digest((ROOT/source).read_bytes()),sources={s:digest((ROOT/s).read_bytes())for s in paths}))
    print(f'Frozen {len(cases)} inline and {len(pages)} complete page references')

if __name__=='__main__':main()
