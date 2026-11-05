"""保存未修改旧表内 writer 原始结果，并单列稳定文件名投影；不宣称文件名等价。"""
import copy,hashlib,importlib,json
from io import BytesIO
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-table-assets.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    tools=importlib.import_module('rapid_doc.utils.pdf_image_tools')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    original=tools.str_sha256;cases=[]
    source='build/migration/legacy-asset-images/RGB-48-40-0.png'
    def det(html,bbox=(2,3,20,22),kind=5,score=.95,order=0,scale=1.):
        x,y,r,b=[v*scale for v in bbox]
        return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=score,original_order=order,html=html)
    def fill(mode='RGB',bbox=(1,2,5,6),uuid='UUID-A',path=None):
        return dict(uuid=uuid,bbox=list(bbox),source=path or f'build/migration/legacy-asset-images/{mode}-48-40-0.png')
    def add(name,dets,fills,scale=1.,parent='images'):
        seeds={};calls=[];renames={};opened=[]
        def capture(seed):
            value=original(seed);seeds[value+'.png']=seed;return value
        page=Image.open(ROOT/source);page.load();page_md5=hashlib.md5(page.tobytes()).hexdigest().upper()
        class Writer:
            _parent_dir=parent
            def write(self,name,data):
                call=dict(name=name)
                if data.startswith(b'\x89PNG'):
                    with Image.open(BytesIO(data))as image:
                        image.load();call.update(mode=image.mode,width=image.width,height=image.height,pixels_sha256=digest(image.tobytes()))
                        if image.info.get('icc_profile'):call['icc_profile_sha256']=digest(image.info['icc_profile'])
                        if 'transparency'in image.info:call['transparency']=list(image.info['transparency'])if isinstance(image.info['transparency'],tuple)else image.info['transparency']
                else:call['bytes_sha256']=digest(data)
                seed=seeds.get(name)
                if seed and seed.startswith('<function save_table_fill_image.'):
                    stable_seed='table_fill/'+page_md5+'_'+seed.rsplit('>_',1)[1]
                    stable_name=original(stable_seed)+'.png';renames[name]=stable_name
                    call.update(legacy_seed=seed,stable_seed=stable_seed)
                calls.append(call)
        try:
            descriptors=[]
            for f in fills:
                image=Image.open(ROOT/f['source']);image.load();opened.append(image)
                descriptors.append(dict(uuid=f['uuid'],bbox=f['bbox'],pil_image=image))
            request=dict(model=dict(layout_dets=dets),page_width=48/scale,page_height=40/scale,page_index=7,scale=scale,formula_enabled=True)
            tools.str_sha256=capture
            expected=old.page_model_info_to_page_info(copy.deepcopy(request['model']),dict(scale=scale,img_pil=page),dict(size=[48/scale,40/scale],ori_image_list=[],table_fill_image_list=descriptors),Writer(),7,ocr_enable=True)
            def project(value):
                if isinstance(value,str):
                    for before,after in renames.items():value=value.replace(before,after)
                    return value
                if isinstance(value,list):return [project(v)for v in value]
                if isinstance(value,dict):return {k:project(v)for k,v in value.items()}
                return value
            pages=[copy.deepcopy(expected)]if expected is not None else [old.make_page_info_dict([],7,int(48/scale),int(40/scale),[])]
            old._optimize_formula_number_blocks(pages);old.para_split(pages)
            cases.append(dict(name=name,request=request,fills=fills,parent=parent,legacy_expected=expected,expected_with_stable_names=project(expected),legacy_calls=calls,
                calls=[{**{k:v for k,v in c.items()if k not in ('legacy_seed','stable_seed','name')},'name':renames.get(c['name'],c['name'])}for c in calls],
                renames=renames,pages_with_stable_names=project(pages),outputs_with_stable_names={mode:project(render.union_make(copy.deepcopy(pages),mode,'images'))for mode in ['mm_markdown','nlp_markdown','content_list']}))
        finally:
            tools.str_sha256=original;page.close()
            for image in opened:image.close()
    for mode in ['L','LA','RGB','RGBA']:
        for scale in [.5,1.,2.]:
            for bbox in [[1,2,5,6],[-1.9,-2.9,5.9,6.9],[0,0,48,40]]:
                add(f'{mode}/{scale}/{bbox}',[det('<table><td>UUID-A</td></table>',scale=scale)],[fill(mode,bbox)],scale)
    html='<table><td>UUID-A UUID-A UUID-B</td></table>'
    add('multiple-and-repeated',[det(html)],[fill(),fill('RGBA',uuid='UUID-B')])
    add('same-in-two-tables',[det(html),det(html,(24,3,42,22),order=1)],[fill(),fill('L',uuid='UUID-B')])
    add('same-in-two-tables-same-bbox-different-pixels',[det(html)],[fill('RGB'),fill('RGBA',uuid='UUID-B')])
    add('duplicate-uuid-first-consumes-all',[det(html)],[fill(),fill('RGBA')])
    add('overlapping-uuids-sequential',[det('<td>UUID-AA UUID-A</td>')],[fill(),fill(uuid='UUID-AA')])
    add('non-matching',[det('<table><td>none</td></table>')],[fill()])
    add('non-table',[det('UUID-A',kind=3)],[fill()])
    add('filtered-low-score',[det('UUID-A',score=.01)],[fill()])
    add('filtered-invalid-bbox',[det('UUID-A',(2,3,2,22))],[fill()])
    add('overlap-lower-score',[det('UUID-A',score=.5),det('kept',score=.9)],[fill()])
    add('html-replacement-not-limited-to-tags',[det('UUID-A<img src="UUID-A"><td>UUID-A</td>')],[fill()])
    add('inline-before-uuid',[det('<img src="data:image/jpeg;base64,YWJjZA=="><td>UUID-A</td>')],[fill()])
    add('unicode-prefix',[det(html)],[fill()],parent='目录/图片')
    add('empty-prefix',[det(html)],[fill()],parent='')
    paths=json.loads((ROOT/'build/migration/legacy-assets-metadata.json').read_text())['cases']
    for c in paths:
        if c['originals']:continue
        add('metadata/'+c['name'],[det('<td>UUID-A</td>')],[fill(path=c['source'])])
    refs=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/pipeline_magic_model.py','rapid_doc/utils/pdf_image_tools.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    inputs={source}|{f['source']for c in cases for f in c['fills']}
    write_json(target,dict(cases=cases,source=source,inputs={p:digest((ROOT/p).read_bytes())for p in inputs},sources={p:digest((ROOT/p).read_bytes())for p in refs},
        profile='intentional_output_difference: 原始旧 JSON/调用/含地址哈希种子保留；expected_with_stable_names 仅投影已捕获资产名称，其他字段/像素/顺序保持原样；不是旧文件名精确等价'))
    print(f'Frozen {len(cases)} old table writer pages and {len(cases)*3} outputs, with explicit stable-name projection')

if __name__=='__main__':main()
