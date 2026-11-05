"""冻结实际旧表格原生文字提取/字符框及整表OCR回退策略，不覆盖基线。"""
import argparse,copy,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from pdftext.schema import Bbox
from common import ROOT,digest,write_json

def adapt(page):
    page=copy.deepcopy(page)
    for block in page['blocks']:
        for line in block['lines']:
            line['bbox']=Bbox(line['bbox'])
            for span in line['spans']:
                if 'bbox'in span:span['bbox']=Bbox(span['bbox'])
                for c in span['chars']:c['bbox']=Bbox(c['bbox'])
    return page
def quad(b):return [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--typed-detector',action='store_true');args=parser.parse_args()
    target=ROOT/('build/migration/table-native-typed-detector.json'if args.typed_detector else 'build/migration/table-native.json')
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc.utils.ocr_utils import get_ocr_result_list_table
    geometry=[];cases=[];real=[]
    prior=json.loads((ROOT/'build/migration/table-text.json').read_text(encoding='utf-8'))
    for c in prior['boxes']:
        for scale in [1,1.37]:
            req=dict(boxes=c['request']['boxes'],scale=scale,useful_list=[3,7,11,17,160,128,160,128])
            geometry.append(dict(request=req,expected=get_ocr_result_list_table(copy.deepcopy(req['boxes']),req['useful_list'],scale)))
    def run(source,boxes,page,scale=1,useful=None,table=None,real_info=None):
        with Image.open(ROOT/source)as im:rgb=np.asarray(im.convert('RGB'));w,h=im.size
        useful=useful or [0,0,0,0,w,h,w,h];table=table or dict(poly=[0,0,w,0,w,h,0,h])
        spans=get_ocr_result_list_table(copy.deepcopy(boxes),useful,scale)
        # 此接口对退化填充框显式报错；旧except返回[]，不冒充这些错误合同等价。
        if any(s['bbox'][2]<=s['bbox'][0]or s['bbox'][3]<=s['bbox'][1]for s in spans):return
        for words in [False,True]:
            req=dict(boxes=copy.deepcopy(boxes),native_page=copy.deepcopy(page),table=copy.deepcopy(table),useful_list=list(useful),scale=scale,word_boxes=words)
            task=dict(table_res=copy.deepcopy(table),table_img=rgb.copy(),useful_list=list(useful))
            result=old._extract_table_text_from_pdf(task,adapt(page),scale,copy.deepcopy(boxes),list(useful),words)
            entry=dict(source=source,source_sha256=digest((ROOT/source).read_bytes()),request=req,expected=result)
            if real_info:entry['pdf']=real_info;real.append(entry)
            else:cases.append(entry)
    fills=json.loads((ROOT/'build/migration/legacy-native-text.json').read_text(encoding='utf-8'))
    for c in fills['cases']:
        r=c['request'];s=r['scale'];boxes=[quad([x*s for x in v['bbox']])for v in r['spans']if v['type']=='text']
        run(c['source'],boxes,r['native_page'],s)
    source='build/migration/table-text/pixels.png';folder=ROOT/'build/migration/table-native';folder.mkdir(parents=True,exist_ok=True)
    blank=folder/'blank.png'
    if not blank.exists():Image.new('RGB',(160,128),'white').save(blank)
    def native(text='ABCD',offset=0,angle=0,rows=1):
        lines=[]
        for row in range(rows):
            y=10+row*25
            chars=[dict(char=c,char_idx=i,bbox=[10+i*5+offset,y,14+i*5+offset,y+10])for i,c in enumerate(text)]
            lines.append(dict(rotation=angle,bbox=[10+offset,y,40+offset,y+10],spans=[dict(bbox=[10+offset,y,40+offset,y+10],text=text,chars=chars)]))
        return dict(blocks=[dict(lines=lines)])
    boxes=[quad([10,10,40,20]),quad([10,35,40,45]),quad([10,60,40,70])]
    for text in ['ABCD','A B','香','哦樂','1號','10號','ﬁﬂﬃ','&<>"\'','A\x02B','\r\n','\x1c \u3000','']:
        for image in [source,blank.relative_to(ROOT).as_posix()]:run(image,boxes,native(text,rows=2))
    for scale in [.5,1,1.37,2.]:
        for useful in [[0,0,0,0,160,128,160,128],[5,7,25,37,185,165,160,128],[-7,-9,0,0,160,128,160,128]]:
            run(source,boxes,native(rows=3),scale,useful)
    # PDF事实由旧PDFium/pdftext实际生成，渲染图也来自旧PDFium。
    import pypdfium2 as pdfium
    facts=json.loads((ROOT/'build/migration/legacy-native-page.json').read_text(encoding='utf-8'))
    selected=[c for c in facts['cases']if c['quote_loosebox']][:5]
    for i,c in enumerate(selected):
        with pdfium.PdfDocument(ROOT/c['source'])as doc:
            page=doc[c['page_index']];im=page.render(scale=1.37,rev_byteorder=True).to_pil().convert('RGB');page.close()
        p=folder/f'pdf-{i}.png'
        if not p.exists():im.save(p)
        qs=[quad([x*1.37 for x in line['bbox']])for block in c['expected']['blocks']for line in block['lines']if line['bbox'][2]-line['bbox'][0]>=3 and line['bbox'][3]>line['bbox'][1]]
        run(p.relative_to(ROOT).as_posix(),qs,c['expected'],1.37,real_info={k:c[k]for k in ['source','sha256','page_index','quote_loosebox']})
    # Actual _process_single_table; fix detector/recognizer/structure only. Native extraction remains real.
    saved=(old._detect_table_text,old._run_table_ocr);workflows=[]
    for kind in ['dense','partial','empty','180','90','page180','no_boxes']:
        for force in [False,True]:
            for ocr_enable in [False,True]:
                pg=native(rows=3 if kind=='dense'else 1)
                if kind=='empty':pg=dict(blocks=[])
                if kind in ('180','90'):pg=native(angle=math.radians(int(kind)),rows=3)
                if kind=='page180':pg['rotate_label']='180'
                qs=[]if kind=='no_boxes'else copy.deepcopy(boxes)
                if args.typed_detector:qs=np.asarray(qs,np.float32).tolist()
                calls=[];captured=[]
                def fallback(worker,image,det,words):
                    calls.append(True);return [copy.deepcopy(det),['FALLBACK']*len(det),[.7]*len(det)]if det else []
                old._detect_table_text=lambda *a,**k:copy.deepcopy(qs);old._run_table_ocr=fallback
                tm=SimpleNamespace(predict=lambda image,ocr_result,*a,**k:(captured.append(copy.deepcopy(ocr_result))or None))
                model=SimpleNamespace(predict=lambda *a,**k:'0')
                from rapid_doc.backend.pipeline.model_list import AtomicModel
                manager=SimpleNamespace(get_atom_model=lambda **k:model if k['atom_model_name']==AtomicModel.ImgOrientationCls else tm)
                table=dict(poly=[0,0,160,0,160,128,0,128]);useful=[0,0,0,0,160,128,160,128]
                task=dict(table_res=copy.deepcopy(table),table_img=np.asarray(Image.open(ROOT/source).convert('RGB')).copy(),useful_list=useful,lang=None,single_page_mfdetrec_res=[],checkbox_res=[],ocr_enable=ocr_enable)
                try:old._process_single_table(task,adapt(pg),1,manager,dict(use_rule_table=False,force_ocr=force,table_formula_enable=False,table_image_enable=False),{})
                finally:old._detect_table_text,old._run_table_ocr=saved
                workflows.append(dict(kind=kind,source=source,page=pg,table=table,useful_list=useful,boxes=qs,force_ocr=force,ocr_enable=ocr_enable,expected=captured[0],fallback=bool(calls)))
    refs=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/ocr_utils.py','rapid_doc/model/table/utils.py','rapid_doc/utils/pdf_image_tools.py']
    write_json(target,dict(geometry=geometry,cases=cases,real=real,workflows=workflows,sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual legacy native table text and word boxes; fixed detector/recognizer/structure responses isolate fallback policy. Real PDF facts and renders listed separately; no structural table model inference.'))
    print(f'Frozen {len(geometry)} geometries/{len(cases)} native fills/{len(real)} actual PDF fills/{len(workflows)} legacy fallback workflows')
if __name__=='__main__':main()
