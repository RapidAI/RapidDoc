"""冻结实际旧BatchAnalyze→result_to_middle_json→输出；固定版面/公式，真实CPU OCR/SLANet。"""
import copy,importlib,json,os,sys
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from common import ROOT,digest,write_json
from legacy_table_models import create_full_models

def main():
    profile='native'if '--native-rule'in sys.argv else'dual'if '--dual-table'in sys.argv else'unet'if '--unet'in sys.argv else'default'if '--default-body'in sys.argv else''
    default_body=bool(profile)
    target=ROOT/f'build/migration/model-document{("-"+profile)if profile else""}.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.utils.pdf_text_tool import get_page
    from rapid_doc.model.layout.rapid_layout import RapidLayoutModel,get_cls_dicts
    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutType
    batch=importlib.import_module('rapid_doc.backend.pipeline.batch_analyze')
    from rapid_doc.model.table.rapid_table import RapidTableModel
    from rapid_doc.model.table.rapid_table_self import RapidTable,RapidTableInput,ModelType,EngineType
    analyze=importlib.import_module('rapid_doc.backend.pipeline.analyze_utils')
    middle=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    metadata=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'))
    worker,orientation=create_full_models(metadata)
    body=worker
    if default_body:
        from rapidocr.ch_ppocr_det.utils import DBPostProcess
        body=copy.copy(worker);body.text_detector=copy.copy(worker.text_detector)
        body.text_detector.postprocess_op=DBPostProcess(thresh=.3,box_thresh=.5,unclip_ratio=1.8,use_dilation=True)
        body.enable_merge_det_boxes=True
    table_ref=json.loads((ROOT/'build/migration/legacy-table-match.json').read_text(encoding='utf-8'))['real'][0]
    table=RapidTableModel.__new__(RapidTableModel);table.model_type=ModelType.SLANETPLUS;table.ocr_engine=worker
    table.table_model=RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS,model_dir_or_path=str(ROOT/table_ref['model']),engine_type=EngineType.ONNXRUNTIME,use_ocr=False,engine_cfg={'intra_op_num_threads':1}))
    if profile=='unet':
        table.model_type=ModelType.UNET
        table.table_model=RapidTable(RapidTableInput(model_type=ModelType.UNET,model_dir_or_path=str(ROOT/'rapid_doc/model/table/rapid_table_self/models/unet.onnx'),engine_type=EngineType.ONNXRUNTIME,use_ocr=False,engine_cfg={'intra_op_num_threads':1}))
    elif profile=='dual':
        cls=json.loads((ROOT/'build/migration/table-classification.json').read_text(encoding='utf-8'))
        table=RapidTableModel(worker,dict(model_type=ModelType.UNET_SLANET_PLUS,engine_type=EngineType.ONNXRUNTIME,engine_cfg={'intra_op_num_threads':1},
            **{'unet.model_dir_or_path':str(ROOT/'rapid_doc/model/table/rapid_table_self/models/unet.onnx'),
               'slanet_plus.model_dir_or_path':str(ROOT/table_ref['model']),
               'cls.model_dir_or_path':{'paddle_cls':str(ROOT/cls['model']),'q_cls':str(ROOT/cls['qmodel'])}}))
    folder=ROOT/'build/migration/model-document';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'five-pages.pdf';c=Canvas(str(source),pagesize=(240,200),invariant=1)
    c.setFont('Helvetica',12);c.drawString(20,170,'Native text first page.');c.drawString(20,145,'Second line of the document.');c.showPage()
    c.showPage()
    c.setFont('Helvetica',12)
    for x in [20,120,220]:c.line(x,40,x,160)
    for y in [40,80,120,160]:c.line(20,y,220,y)
    for y,a,b in [(140,'Name','Value'),(100,'Alpha','123'),(60,'Beta','456')]:c.drawString(30,y,a);c.drawString(130,y,b)
    c.showPage();c.drawImage(str(ROOT/'build/migration/detection-images/text.png'),20,80,width=80,height=60)
    c.setFont('Helvetica',14);c.drawString(140,145,'x + y');c.showPage();c.showPage();c.save()
    def region(label,b,order=0):return dict(label=label,coordinate=b,score=.95,order=order)
    regions=[[region('text',[15.,15.,225.,70.])],[],[region('table',[18.,38.,222.,162.])],
             [region('image',[20.,60.,100.,120.]),region('display_formula',[138.,40.,190.,60.],1)],[]]
    wrapper=RapidLayoutModel.__new__(RapidLayoutModel);wrapper.docvortex_mode='off';wrapper.model_type=LayoutType.PP_DOCLAYOUTV3
    wrapper.pp_doclayoutv2_cls_dict=get_cls_dicts(['number','footnote','header','header_image','footer','footer_image','aside_text'])[2]
    class Layout:
        def batch_predict(self,images,*a,**kw):
            def predict(**kwargs):
                return [SimpleNamespace(boxes=[x['coordinate']for x in r],scores=[x['score']for x in r],class_names=[x['label']for x in r],orders=[x['order']for x in r],polygon_points=None)for r in regions]
            wrapper.model=predict
            return wrapper.batch_predict(images,*a,**kw)
    class Manager:
        def get_atom_model(self,**kw):
            kind=kw['atom_model_name']
            return orientation if kind=='img_ori_cls'else table if kind=='table'else worker if kw.get('det_db_unclip_ratio')==1.6 else body
        def get_model(self,**kw):return SimpleNamespace(ocr_model=body,layout_model=Layout(),table_model=table,formula_model=SimpleNamespace(batch_predict=lambda imgs,**kw:['x+y']*len(imgs)),device='cpu')
    class Writer:
        _parent_dir='images'
        def __init__(self):self.assets=[]
        def write(self,name,data):
            with Image.open(BytesIO(data))as im:self.assets.append(dict(name='images/'+name,mode=im.mode,width=im.width,height=im.height,pixels_sha256=digest(im.tobytes())))
    originals=[(mod,mod.AtomModelSingleton)for mod in (batch,analyze,middle)];cases=[];native_references=[]
    os.environ['MINERU_DONOT_CLEAN_MEM']='1'
    try:
        for mod,_ in originals:mod.AtomModelSingleton=Manager
        for native in (True,False):
            images=[];facts=[]
            with pdfium.PdfDocument(source)as doc:
                for p in doc:
                    facts.append(get_page(p));facts[-1]['ori_image_list']=[]
                    bitmap=p.render(scale=1);images.append(bitmap.to_pil().convert('RGB'));bitmap.close();p.close()
            if profile=='native'and native:
                from dataclasses import asdict
                from rapid_doc.model.docvortex_adapter import recover_docvortex_table
                for i,pd in enumerate(facts):
                    if pd['blocks']:pd.update(_source_pdf_bytes=source.read_bytes(),_source_page_index=i)
                bbox=[18.,38.,222.,162.];r=recover_docvortex_table(facts[2],bbox)
                native_references.append(dict(page_index=2,bbox=bbox,result=None if r is None else asdict(r)))
            analyzer=batch.BatchAnalyze(Manager(),1,True,True,ocr_config={'seal_enable':False,'use_det_mode':'auto'if native else'ocr'},table_config={'use_rule_table':profile=='native','force_ocr':not native})
            analyzer.use_doc_orientation_classify=False
            layouts=analyzer([(im,1.,not native,None,pd)for im,pd in zip(images,facts)])
            writer=Writer();output=middle.result_to_middle_json([dict(layout_dets=d)for d in layouts],[dict(img_pil=im,scale=1.)for im in images],facts,writer,ocr_enable=not native,formula_enabled=True)
            pages=output['pdf_info']
            cases.append(dict(native_text=native,expected=dict(pages=pages,markdown=render.union_make(copy.deepcopy(pages),'mm_markdown',''),content_list=render.union_make(copy.deepcopy(pages),'content_list','')),assets=writer.assets))
    finally:
        for mod,value in originals:mod.AtomModelSingleton=value
    refs=['rapid_doc/backend/pipeline/batch_analyze.py','rapid_doc/backend/pipeline/analyze_utils.py','rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py','rapid_doc/model/table/rapid_table.py']
    write_json(target,dict(source=source.relative_to(ROOT).as_posix(),source_sha256=digest(source.read_bytes()),regions=regions,cases=cases,native_references=native_references,
        models={k:metadata[k]for k in ('det_model','det_sha256','rec_model','rec_sha256','orientation_model','orientation_sha256')}|dict(table_model=table_ref['model'],table_sha256=table_ref['sha256']),
        sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual old five-page BatchAnalyze→result_to_middle_json→Markdown/content_list; fixed layout regions/formula x+y, real PDF render/native facts and CPU OCR/SLANet; does not prove actual layout/formula model document parity.'))
    print('Frozen 2 five-page old document compositions (native/OCR), including table, image, formula and blank pages')
if __name__=='__main__':main()
