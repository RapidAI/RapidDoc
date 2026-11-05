"""Freeze actual original native layout -> CPU OCR/table -> Markdown and assets."""
import copy,importlib,json,os,sys
from io import BytesIO
from types import SimpleNamespace
from common import ROOT,digest,write_json
from legacy_table_models import create_full_models
def main(samples=False):
    target=ROOT/('build/migration/native-host-samples.json'if samples else'build/migration/native-host-document.json')
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove()
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import Image
    from rapid_doc.utils.pdf_text_tool import get_page
    from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel
    batch=importlib.import_module('rapid_doc.backend.pipeline.batch_analyze')
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import RapidTable,RapidTableInput,ModelType,EngineType
    from rapidocr.ch_ppocr_det.utils import DBPostProcess
    analyze=importlib.import_module('rapid_doc.backend.pipeline.analyze_utils')
    middle=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    metadata=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'))
    worker,orientation=create_full_models(metadata)
    body=copy.copy(worker);body.text_detector=copy.copy(worker.text_detector)
    body.text_detector.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.8,use_dilation=True);body.enable_merge_det_boxes=True
    table_ref=json.loads((ROOT/'build/migration/legacy-table-match.json').read_text(encoding='utf-8'))['real'][0]
    table=RapidTableModel.__new__(RapidTableModel);table.model_type=ModelType.SLANETPLUS;table.ocr_engine=worker
    table.table_model=RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS,model_dir_or_path=str(ROOT/table_ref['model']),engine_type=EngineType.ONNXRUNTIME,use_ocr=False,engine_cfg={'intra_op_num_threads':1}))
    source=ROOT/'build/migration/model-document/five-pages.pdf'
    formula=SimpleNamespace(batch_predict=lambda imgs,**kw:['x+y']*len(imgs))
    if samples:
        from rapid_doc.model.formula.rapid_formula_model import RapidFormulaModel
        formula=RapidFormulaModel(dict(model_dir_or_path=ROOT/'build/migration/models/pp_formulanet_plus_m.onnx',engine_cfg=dict(intra_op_num_threads=1,inter_op_num_threads=1,use_cuda=False,use_dml=False,use_cann=False)))
    layout=RapidLayoutModel(dict(model_type='docvortex_rule'))
    class Manager:
        def get_atom_model(self,**kw):
            kind=kw['atom_model_name'];return orientation if kind=='img_ori_cls'else table if kind=='table'else worker if kw.get('det_db_unclip_ratio')==1.6 else body
        def get_model(self,**kw):return SimpleNamespace(ocr_model=body,layout_model=layout,table_model=table,formula_model=formula,device='cpu')
    class Writer:
        _parent_dir='images'
        def __init__(self):self.assets=[]
        def write(self,name,data):
            with Image.open(BytesIO(data))as im:self.assets.append(dict(name='images/'+name,mode=im.mode,width=im.width,height=im.height,pixels_sha256=digest(im.tobytes())))
    originals=[(mod,mod.AtomModelSingleton)for mod in (batch,analyze,middle)];cases=[];os.environ['MINERU_DONOT_CLEAN_MEM']='1'
    try:
        for mod,_ in originals:mod.AtomModelSingleton=Manager
        jobs=[(ROOT/'demo/pdfs'/name,True)for name in ['示例1-论文模板.pdf','比亚迪财报.pdf']]if samples else[(source,True),(source,False)]
        scale=2. if samples else 1.
        for source,native in jobs:
            data=source.read_bytes();native_pages=analyze_docvortex_layout(data)
            images=[];facts=[]
            with pdfium.PdfDocument(data)as doc:
                for i,p in enumerate(doc):
                    facts.append(get_page(p));facts[-1].update(ori_image_list=[],_docvortex_layout=native_pages[i])
                    if not native:facts[-1]['_docvortex_ocr']=True
                    if native and facts[-1]['blocks']:facts[-1].update(_source_pdf_bytes=data,_source_page_index=i)
                    bitmap=p.render(scale=scale);images.append(bitmap.to_pil().convert('RGB'));bitmap.close();p.close()
            analyzer=batch.BatchAnalyze(Manager(),1,True,True,ocr_config=dict(seal_enable=False,use_det_mode='auto'if native else'ocr'),table_config=dict(use_rule_table=True,force_ocr=not native))
            analyzer.use_doc_orientation_classify=False
            layouts=analyzer([(im,scale,not native,None,pd)for im,pd in zip(images,facts)])
            writer=Writer();pages=middle.result_to_middle_json([dict(layout_dets=d)for d in layouts],[dict(img_pil=im,scale=scale)for im in images],facts,writer,ocr_enable=not native,formula_enabled=True)['pdf_info']
            cases.append(dict(source=source.relative_to(ROOT).as_posix(),source_sha256=digest(data),native_text=native,dpi=scale*72,expected=dict(pages=pages,markdown=render.union_make(copy.deepcopy(pages),'mm_markdown','images'),content_list=render.union_make(copy.deepcopy(pages),'content_list','images')),assets=writer.assets))
    finally:
        for mod,value in originals:mod.AtomModelSingleton=value
    write_json(target,dict(source=source.relative_to(ROOT).as_posix(),source_sha256=digest(data),models={k:metadata[k]for k in ['det_model','det_sha256','rec_model','rec_sha256','orientation_model','orientation_sha256']}|dict(table_model=table_ref['model'],table_sha256=table_ref['sha256']),cases=cases,scope='Original complete native PdfModel/layout adapter and BatchAnalyze, real CPU OCR/SLANet, actual PDF and image output. '+('Actual CPU formula model; two user samples at 144dpi.'if samples else'Formula recognizer fixed to x+y.')+' No actual layout model is used in native-only mode.'))
    print(len(cases),'original native host documents,',sum(len(c['expected']['pages'])for c in cases),'pages; samples=',samples,flush=True)
if __name__=='__main__':main(samples='--samples'in sys.argv)
