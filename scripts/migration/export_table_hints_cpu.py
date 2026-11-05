"""真实PDF/CPU文字链→旧公式图片填充→结构模型捕获点；不执行结构网络。"""
import copy,json
from types import SimpleNamespace
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json
from export_table_native import adapt
from export_table_text import norm,pixel
from legacy_table_models import create_models

def main():
    target=ROOT/'build/migration/table-hints-cpu.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils as old
    from rapid_doc.backend.pipeline.model_list import AtomicModel
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import ModelType
    prior=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'))
    worker,orientation=create_models(prior);cases=[]
    for index in [0,1,8,9]:
        c=prior['cases'][index];r=copy.deepcopy(c['request']);page=r['native_page'];page['ori_image_list']=[]
        table=r['table'];table['layout_image_list']=[dict(poly=[15.,60.,55.,60.,55.,80.,15.,80.],uuid='image-uuid',image_ref='external-png')]
        formulas=[dict(bbox=[25.,15.,50.,30.],latex='x<&>')];checkboxes=[dict(bbox=[90.,40.,105.,55.],checkbox='香')]
        captured=[]
        def capture(images,rows):captured.append(dict(ocr_result=norm(rows[0]),image=pixel(images[0][:,:,::-1].copy())));return SimpleNamespace(pred_htmls=[None])
        tm=RapidTableModel.__new__(RapidTableModel);tm.model_type=ModelType.SLANETPLUS;tm.table_model=capture;tm.ocr_engine=worker
        manager=SimpleNamespace(get_atom_model=lambda **k:worker if k['atom_model_name']==AtomicModel.OCR else orientation if k['atom_model_name']==AtomicModel.ImgOrientationCls else tm)
        rgb=np.asarray(Image.open(ROOT/c['source']).convert('RGB'))
        task=dict(table_res=copy.deepcopy(table),table_img=rgb.copy(),useful_list=r['useful_list'],lang=None,single_page_mfdetrec_res=copy.deepcopy(formulas),checkbox_res=copy.deepcopy(checkboxes),ocr_enable=False)
        old._process_single_table(task,adapt(page),r['scale'],manager,dict(use_rule_table=False,force_ocr=r['force_ocr']),{})
        if len(captured)!=1:raise AssertionError('expected one actual model preparation')
        cases.append(dict(source=c['source'],source_sha256=c['source_sha256'],pdf=c['pdf'],request=dict(page=page,table=table,useful_list=r['useful_list'],scale=r['scale'],force_ocr=r['force_ocr'],formulas=formulas,checkboxes=checkboxes),expected=captured[0]))
    refs=['rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/model/table/rapid_table.py','rapid_doc/utils/span_pre_proc.py','rapid_doc/utils/ocr_utils.py']
    models={k:prior[k]for k in ['det_model','det_sha256','rec_model','rec_sha256','orientation_model','orientation_sha256']}
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in refs},**models,scope='2 real PDF pages × force_ocr false/true: actual DB/direction/native text or CTC then formulas/images preparation; formula and layout-image metadata are supplied author inputs, not formula/image detection or extraction. Structure network is a capture stub only.'))
    print(f'Frozen {len(cases)} real PDF→CPU→formula/image model payloads; structure network not executed')
if __name__=='__main__':main()
