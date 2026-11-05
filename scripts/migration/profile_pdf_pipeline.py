"""Read-only default PDF phase diagnostics, run after performance measurements.

Python records ORT run calls; Rust records existing native progress summaries.
Those boundaries differ and must not be compared or summed as identical stages.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine',choices=['python','rust'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calls',type=int,default=2)
    parser.add_argument('--native-rule',action='store_true')
    parser.add_argument('--python-phases',action='store_true',help='Opt-in coarse Python phase timers; nested rows overlap')
    parser.add_argument('--capture-unet',type=Path,help='Diagnostic only: save actual UNET BGR pixels, tensors and class maps')
    args=parser.parse_args()
    if args.capture_unet:
        if args.engine!='python' or not args.python_phases:parser.error('--capture-unet requires Python phase diagnostics')
        args.capture_unet.mkdir(parents=True,exist_ok=False)
    from loguru import logger
    logger.remove(); model_names={}; timings={}; native=[];sessions=[];python_phases={}
    if args.engine=='python':
        import onnxruntime as ort
        init=ort.InferenceSession.__init__;run=ort.InferenceSession.run
        def initialize(self,path,*pos,**kw):
            result=init(self,path,*pos,**kw)
            model_names[id(self)]=str(path) if isinstance(path,(str,Path)) else '<in-memory model>'
            options=self.get_session_options()
            sessions.append(dict(model=model_names[id(self)],intra_op_threads=options.intra_op_num_threads,
                                 inter_op_threads=options.inter_op_num_threads,providers=self.get_providers()))
            return result
        def measured(self,*pos,**kw):
            start=time.perf_counter();result=run(self,*pos,**kw);elapsed=time.perf_counter()-start
            name=model_names.get(id(self),'<unknown>')
            value=timings.setdefault(name,dict(seconds=0.,calls=0,shapes=Counter()))
            value['seconds']+=elapsed;value['calls']+=1
            feeds=pos[1] if len(pos)>1 else kw.get('input_feed',{})
            value['shapes'][str({k:list(v.shape) for k,v in feeds.items() if hasattr(v,'shape')})]+=1
            return result
        ort.InferenceSession.__init__=initialize;ort.InferenceSession.run=measured
        from rapid_doc import RapidDoc
        if args.python_phases:
            import functools
            import importlib
            def wrap(module,name,label):
                original=getattr(module,name)
                @functools.wraps(original)
                def measured_phase(*pos,**kw):
                    started=time.perf_counter()
                    try:return original(*pos,**kw)
                    finally:
                        row=python_phases.setdefault(label,dict(calls=0,seconds=0.))
                        row['calls']+=1;row['seconds']+=time.perf_counter()-started
                setattr(module,name,measured_phase)
            analyze=importlib.import_module('rapid_doc.backend.pipeline.pipeline_analyze')
            assemble=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
            pipeline=importlib.import_module('rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline')
            for module,name,label in [
                (analyze,'analyze_docvortex_layout','native_analysis'),
                (analyze,'load_images_from_pdf','render'),
                (assemble,'page_model_info_to_page_info','assemble'),
                (pipeline,'_collect_document_sources','collect_native_pages'),
                (pipeline,'_prepare_document_sources','prepare_document_pages'),
                (pipeline,'_finalize_prepared_page','finalize_native_page'),
            ]:wrap(module,name,label)
            # The same production method boundaries as Rust's coarse stages.
            from rapid_doc.backend.pipeline.batch_analyze import BatchAnalyze
            from rapid_doc.model.table.rapid_table import RapidTableModel
            analyze_utils=importlib.import_module('rapid_doc.backend.pipeline.analyze_utils')
            for module,name,label in [
                (BatchAnalyze,'_run_layout_detection','layout'),
                (BatchAnalyze,'_collect_detection_regions','region_crops'),
                (BatchAnalyze,'_run_formula_recognition','formula'),
                (BatchAnalyze,'_run_traditional_ocr','ocr_detection_and_crops'),
                (BatchAnalyze,'_run_table_recognition','table'),
                (analyze_utils,'_run_ocr_rec_postprocess','ocr_recognition'),
                (analyze_utils,'_detect_table_text','table_text_detection'),
                (analyze_utils,'_extract_table_text_from_pdf','table_native_text'),
                (analyze_utils,'_run_table_ocr','table_text_recognition'),
                (RapidTableModel,'predict','table_predict'),
            ]:wrap(module,name,label)
            # batch_analyze imported this function before it was wrapped.
            batch=importlib.import_module('rapid_doc.backend.pipeline.batch_analyze')
            batch._run_ocr_rec_postprocess=analyze_utils._run_ocr_rec_postprocess
            from rapid_doc.model.table.rapid_table_self.table_structure.unet.main import TSRUnetStructurer
            for name in ['preprocess','infer','postprocess']:
                wrap(TSRUnetStructurer,name,'unet_'+name)
            if args.capture_unet:
                import numpy as np
                preprocess=TSRUnetStructurer.preprocess
                postprocess=TSRUnetStructurer.postprocess
                def save_tensor(self,*pos,**kw):
                    result=preprocess(self,*pos,**kw)
                    self._diagnostic_unet_tensor=result['img']
                    return result
                def save_case(self,img,pred,*pos,**kw):
                    pixels=np.ascontiguousarray(img)
                    mask=np.ascontiguousarray(pred,dtype=np.uint8)
                    key=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()+mask.tobytes()).hexdigest()
                    folder=args.capture_unet/key
                    if not folder.exists():
                        folder.mkdir()
                        pixels.tofile(folder/'pixels.bgr');mask.tofile(folder/'mask.u8')
                        tensor=np.ascontiguousarray(self._diagnostic_unet_tensor,dtype='<f4')
                        tensor.tofile(folder/'input.f32')
                        (folder/'case.json').write_text(json.dumps(dict(width=pixels.shape[1],height=pixels.shape[0],
                            mask_width=mask.shape[1],mask_height=mask.shape[0],tensor_shape=list(tensor.shape))))
                    return postprocess(self,img,pred,*pos,**kw)
                TSRUnetStructurer.preprocess=save_tensor
                TSRUnetStructurer.postprocess=save_case
    else:
        from rapid_doc_rust import RapidDoc
        from rapid_doc_rust._runtime import LazySession
        descriptor=LazySession.descriptor
        def recorded(self):
            value=descriptor(self);parsed=json.loads(value)
            if parsed not in sessions:sessions.append(parsed)
            return value
        LazySession.descriptor=recorded
        def collect(message):
            match=re.search(r': ([a-z_]+): (\d+) items in ([0-9.]+)s$',message.record['message'])
            if match:native.append(dict(stage=match[1],items=int(match[2]),seconds=float(match[3])))
        logger.add(collect,level='INFO')
    options=dict(layout_config=dict(model_type='docvortex_rule'),formula_enable=False,table_enable=False,parse_method='txt',pdf_pages_batch=64) if args.native_rule else {}
    engine=RapidDoc(**options);data=args.source.read_bytes();rows=[]
    for i in range(args.calls):
        gc.collect();timings.clear();native.clear();python_phases.clear();start=time.perf_counter();result=engine(data)
        elapsed=time.perf_counter()-start
        rows.append(dict(call='cold' if i==0 else 'warm',seconds=elapsed,pages=len(result.middle_json['pdf_info']),
            markdown_sha256=hashlib.sha256(result.markdown.encode()).hexdigest(),
            python_ort_runs=dict(timings),rust_native_stages=list(native)))
        if args.python_phases:rows[-1]['python_phases']={k:dict(v) for k,v in python_phases.items()}
        del result
    args.output.write_text(json.dumps(dict(engine=args.engine,source=str(args.source),options=options,rows=rows,sessions=sessions,
        notes=['Diagnostic timings include instrumentation; use independent acceptance for speed ratios.',
               'Native timings can be nested; do not sum every row.',
               'Python ORT run times exclude model initialization and all preprocessing/postprocessing.']),indent=2),encoding='utf8')


if __name__=='__main__':main()
