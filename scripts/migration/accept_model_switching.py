"""CPU acceptance for new model switches: original Python versus native Rust.

Uses identical original weights, inputs and explicit thresholds. Does not run
the full corpus or download model optimizations. Run after installing the wheel.
"""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=Path,required=True);parser.add_argument('--pdf-only',action='store_true');parser.add_argument('--include-plus-pdf',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'build/model-switch-acceptance');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['RAPID_MODELS_DIR']=str(args.models.resolve())
    from rapid_doc_rust import OnnxSession,RapidDoc
    from rapid_doc_rust._runtime import runtime_path,CATALOG
    source=ROOT/'demo/pdfs/示例1-论文模板.pdf'
    if args.pdf_only:
        report=json.loads((args.output/'raw-report.json').read_text(encoding='utf-8'))
    else:
        import pypdfium2
        source=ROOT/'demo/pdfs/示例1-论文模板.pdf'
        doc=pypdfium2.PdfDocument(source);page=doc[0];bitmap=page.render(scale=2);pil=bitmap.to_pil().convert('RGB');rgb=np.asarray(pil).copy();bitmap.close();page.close();doc.close()
        stream=io.BytesIO();pil.save(stream,format='PNG');png=stream.getvalue()
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.log_severity_level=3
        def sessions(model_id):
            metadata=CATALOG['models'][model_id];path=args.models/metadata['filename'];sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert sha==metadata['sha256'],model_id
            return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']),OnnxSession(runtime_path(),path,sha,threads=1,max_elements=sys.maxsize//4)
        report={'layout':{},'formula':{},'table':{},'pdf':{}}
        from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
        from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess
        from rapid_doc.model.layout.rapid_layout_self.model_handler.utils import ModelType
        from rapid_doc.model.layout.rapid_layout_self.utils.typings import PP_DOCLAYOUT_PLUS_L_layout_merge_bboxes_mode,PP_DOCLAYOUTV2_layout_merge_bboxes_mode
        for model_id in ['pp_doclayout_plus_l','pp_doclayoutv2','pp_doclayoutv3']:
            py,native=sessions(model_id);labels=py.get_modelmeta().custom_metadata_map['character'].splitlines();model_type=ModelType(model_id)
            inputs={'image':PPPreProcess((800,800),model_type)(rgb),'im_shape':np.array([[800,800]],np.float32),'scale_factor':np.array([[800/rgb.shape[0],800/rgb.shape[1]]],np.float32)}
            out=py.run(None,inputs);merge=PP_DOCLAYOUT_PLUS_L_layout_merge_bboxes_mode if model_id=='pp_doclayout_plus_l'else PP_DOCLAYOUTV2_layout_merge_bboxes_mode
            expected=PPPostProcess(labels,.3,.5,layout_merge_bboxes_mode=merge,layout_unclip_ratio=[1.,1.],scale_size=(800,800))(out[0],(rgb.shape[1],rgb.shape[0]),None,'rect')
            expected=list(expected)
            actual=json.loads(native._native.layout_polygons_png_json(png,.3,'rect',None))
            assert [o['label']for o in actual]==[o['label']for o in expected],(model_id,'labels')
            assert np.allclose([o['coordinate']for o in actual],[o['coordinate']for o in expected],rtol=0,atol=.001),(model_id,'boxes')
            assert np.allclose([o['score']for o in actual],[o['score']for o in expected],rtol=0,atol=1e-5),(model_id,'scores')
            report['layout'][model_id]=dict(regions=len(actual),python_matched=True);print('layout matched',model_id,len(actual),flush=True)
            del py,native;gc.collect()
        from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.pre_process import PPPreProcess as FormulaPre
        from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.post_process import PPPostProcess as FormulaPost
        im=Image.new('RGB',(240,64),'white');ImageDraw.Draw(im).text((12,12),'x + y = 2',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32),fill='black')
        stream=io.BytesIO();im.save(stream,format='PNG');formula_png=stream.getvalue();bgr=np.asarray(im)[:,:,::-1].copy()
        for model_id in ['pp_formulanet_plus_s','pp_formulanet_plus_l']:
            py,native=sessions(model_id);side=py.get_inputs()[0].shape[2];tensor=np.concatenate(FormulaPre((side,side))([bgr]),axis=0)
            out=py.run(None,{py.get_inputs()[0].name:tensor})[0]
            characters=json.loads(py.get_modelmeta().custom_metadata_map['character'])
            expected=FormulaPost(characters)([p.reshape(-1)for p in out]);actual=native.recognize_formula_batch_png([formula_png])
            assert actual==expected,(model_id,actual,expected);report['formula'][model_id]=dict(text=actual,python_matched=True);print('formula matched',model_id,actual,flush=True)
            del py,native;gc.collect()
        from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer.main import preprocess,outputs_to_objects,ocr_to_tokens,THRESHOLDS
        from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer import structure
        from copy import deepcopy
        table_image=ROOT/'build/notes81-route-check/original.png'
        if not table_image.is_file():table_image=ROOT/'tests/fixtures/model_switching/raster-grid.png'
        image=Image.open(table_image).convert('RGB')
        image_rgb=np.asarray(image);py,native=sessions('table_transformer_v1_1_all')
        logits,boxes=py.run(None,{'images':preprocess(image_rgb[:,:,::-1])})
        tokens=[dict(bbox=[20.,20.,80.,40.],text='A & <B>',block_num=0,line_num=0,span_num=0)]
        objects=outputs_to_objects(logits,boxes,image.size);tables=[o for o in objects if o['label']=='table'and o['score']>=.5]
        expected=[]
        if tables:
            table=max(tables,key=lambda t:max(0,t['bbox'][2]-t['bbox'][0])*max(0,t['bbox'][3]-t['bbox'][1]))
            structs=structure.objects_to_structures(deepcopy([o for o in objects if o['label']!='table']+[table]),deepcopy(tokens),THRESHOLDS)
            if structs and structs[0]['rows']and structs[0]['columns']:expected=structure.structure_to_cells(structs[0],deepcopy(tokens))[0]
        stream=io.BytesIO();image.save(stream,format='PNG');actual=json.loads(native._native.table_transformer_png_json(stream.getvalue(),json.dumps(tokens)))
        assert [c['row_nums']for c in actual]==[c['row_nums']for c in expected]
        assert [c['column_nums']for c in actual]==[c['column_nums']for c in expected]
        assert [c['text']for c in actual]==[c['cell text']for c in expected]
        report['table']['table_transformer']=dict(cells=len(actual),python_matched=True);print('TATR matched',len(actual),flush=True)
        del py,native;gc.collect()
        (args.output/'raw-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for model_id in (['pp_doclayout_plus_l']if args.include_plus_pdf else[])+['pp_doclayoutv2']:
        output=args.output/model_id
        result=RapidDoc(output_dir=output,layout_config={'model_type':model_id},formula_enable=False)(source,end_page_id=0)
        assert result.markdown.strip();report['pdf'][model_id]=dict(markdown_characters=len(result.markdown))
    result=RapidDoc(output_dir=args.output/'ocr-rule',parse_method='ocr',formula_enable=False,table_config={'model_type':'table_transformer','rule_table_engine':'docvortex_ocr'})(ROOT/'demo/pdfs/small_ocr.pdf',end_page_id=0)
    assert result.markdown.strip();report['pdf']['ocr-rule']=dict(markdown_characters=len(result.markdown))
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
