"""冻结真实旧UNET最终HTML：已有转义OCR，默认不挂空格识别器；不覆写旧参考。"""
import copy,json
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-unet.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import RapidTable,RapidTableInput,ModelType,EngineType
    from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import plot_html_table
    from rapid_doc.model.table.utils import normalize_table_html_cell_text
    from legacy_table_models import create_full_models
    prior=json.loads((ROOT/'build/migration/wired-pipeline.json').read_text(encoding='utf-8'))
    meta=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'))
    ocr,_=create_full_models(meta)
    model=RapidTableModel.__new__(RapidTableModel);model.model_type=ModelType.UNET;model.ocr_engine=ocr
    model.table_model=RapidTable(RapidTableInput(model_type=ModelType.UNET,engine_type=EngineType.ONNXRUNTIME,model_dir_or_path=str(ROOT/prior['unet_model']),use_ocr=False,engine_cfg={'intra_op_num_threads':1}))
    real=[];pipelines=[];html=[]
    for old in prior['cases']:
        path=__import__('pathlib').Path(old['input']);image=np.asarray(Image.open(path).convert('RGB'));h,w=image.shape[:2]
        rows=[[[[30,20],[130,20],[130,60],[30,60]],[[30,90],[140,90],[140,130],[30,130]],[[260,90],[350,90],[350,130],[260,130]],[[30,165],[140,165],[140,210],[30,210]],[[260,165],[350,165],[350,210],[260,210]]],['Header','中 文','&amp; &lt;A&gt;','<b>Beta</b>','&#x27;'],[1.]*5]
        pred=model.table_model([image[:,:,::-1].copy()],[copy.deepcopy(rows)])
        raw=pred.pred_htmls[0]
        real.append(dict(source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path.read_bytes()),ocr_result=rows,raw_html=raw,html=normalize_table_html_cell_text(raw)))
        final=model.predict(image,None,skip_table_orientation=True)
        pipelines.append(dict(source=path.relative_to(ROOT).as_posix(),html=final))
    for n in range(1,5):
        points=[[0,0,i,i]for i in range(n)];polys=[[[i*100.,0.],[i*100.,40.],[(i+1)*100.,40.],[(i+1)*100.,0.]]for i in range(n)]
        for text in ['', '中 文','&amp; &lt; &quot; &#x27;', '<b>A</b>', 'a<br>b', ' \n\t ', '\x1cA\r\nB\u3000']:
            texts=[[text,'second']if i%2 else[text]for i in range(n)]
            html.append(dict(points=points,polygons=polys,texts=texts,expected=plot_html_table(points,{i:t for i,t in enumerate(texts)},polys)))
    paths=['rapid_doc/model/table/rapid_table_self/table_structure/unet/unet_table_rec.py','rapid_doc/model/table/rapid_table_self/table_structure/unet/utils/utils_table_recover.py','rapid_doc/model/table/rapid_table.py']
    write_json(target,dict(real=real,pipelines=pipelines,html=html,model=prior['unet_model'],sha256=prior['unet_sha256'],models=meta,
        sources={p:digest((ROOT/p).read_bytes())for p in paths},scope='Actual old UNET default enable_blank_cell_rec=False, provided already-escaped OCR and full OCR fallback, raw HTML and final normalization; not optional blank-cell recognizer, dual-model classifier or arbitrary HTML parity.'))
    print(f'Frozen {len(html)} old unescaped HTML cases, {len(real)} real UNET inputs and {len(pipelines)} actual OCR→UNET outputs')
if __name__=='__main__':main()
