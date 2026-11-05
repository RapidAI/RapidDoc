"""Run with the clean environment's Python from an empty working directory."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--phase',choices=['download','parse'],required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--reference-geos',type=Path,
                        help='Diagnostic only: compare the bundled runtime to a prior explicitly configured GEOS')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    forbidden=['rapid_doc','rapidocr','pdftext','cv2','onnx','shapely','pypdfium2']
    absent={name:importlib.util.find_spec(name) is None for name in forbidden}
    assert all(absent.values()),absent
    if args.reference_geos:
        os.environ['RAPIDDOC_GEOS_LIBRARY']=str(args.reference_geos.resolve(strict=True))
    from rapid_doc_rust import RapidDoc,parse_pdf_text
    from rapid_doc_rust._runtime import download_directory,resolve_model,runtime_path,geos_runtime,operator_library
    from rapid_doc_rust._pdfium_runtime import resolve_pdfium_library
    report=dict(python=sys.version,device=args.device,phase=args.phase,unrelated_parsers_absent=absent,
                prefix=sys.prefix,system_site_packages=False,checks=[])
    report['runtime_paths']=dict(ort=str(runtime_path()),geos=str(geos_runtime()),pdfium=resolve_pdfium_library())
    assert all(Path(name).is_file() for name in report['runtime_paths'].values())
    report['packages']={name:importlib.metadata.version(name) for name in
        ['rapid-doc-rust','Pillow','numpy','onnxruntime-gpu' if args.device=='cuda' else 'onnxruntime']}
    # Standalone fallback directory is shared with the original Python package.
    report['standalone_model_directory']=str(download_directory('orientation'))
    if args.phase=='download':
        started=time.perf_counter()
        first=resolve_model('orientation')
        second=resolve_model('orientation')
        assert first==second and hashlib.sha256(first[0].read_bytes()).hexdigest()==first[1]
        report['checks'].append(dict(kind='real_original_model_download_and_reuse',path=str(first[0]),
            sha256=first[1],bytes=first[0].stat().st_size,seconds=time.perf_counter()-started))
        assert not list(first[0].parent.glob('*.part'))
    else:
        engine=RapidDoc()
        for suffix in ['doc','docx','xls','xlsx','ppt','pptx']:
            source=next((args.sources/'ms_office_docs').glob('*.'+suffix))
            result=engine(source)
            assert result.markdown.strip(),suffix
            report['checks'].append(dict(kind='office',format=suffix,pages=len(result.middle_json['pdf_info']),
                markdown_characters=len(result.markdown),assets=len(result.images)))
        # PDFium/GEOS/ORT are found without any library override environment.
        prohibited=['RAPIDDOC_ONNXRUNTIME_LIBRARY','RAPIDDOC_PDFIUM_LIBRARY',
                    'RAPIDDOC_OPERATOR_LIBRARY','ONNXRUNTIME_OPS_LIBRARY']
        if not args.reference_geos:
            prohibited.append('RAPIDDOC_GEOS_LIBRARY')
        assert not any(os.environ.get(key) for key in prohibited)
        if args.device=='cuda':
            os.environ['MINERU_DEVICE_MODE']='cuda'
            report['operator_library']=str(operator_library())
            import onnxruntime
            assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
        else:
            os.environ['MINERU_DEVICE_MODE']='cpu'
        for filename,index in [('small_ocr.pdf',0),('示例1-论文模板.pdf',5),('49－长表格-密集型.pdf',0),('含印章.pdf',0)]:
            started=time.perf_counter()
            result=engine(args.sources/'pdfs'/filename,start_page_id=index,end_page_id=index)
            assert len(result.middle_json['pdf_info'])==1 and result.markdown.strip(),filename
            target=args.output/(Path(filename).stem+'.md')
            target.write_text(result.markdown,encoding='utf8')
            identity=dict(markdown=result.markdown,middle_json=result.middle_json,
                content_list_json=result.content_list_json,
                assets={name:hashlib.sha256(data).hexdigest() for name,data in result.images.items()})
            report['checks'].append(dict(kind='model_pdf',source=filename,page_index=index,
                seconds=time.perf_counter()-started,markdown_characters=len(result.markdown),assets=len(result.images),
                markdown_sha256=hashlib.sha256(result.markdown.encode()).hexdigest(),
                structure_and_assets_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,
                    separators=(',',':')).encode()).hexdigest()))
        report['models_loaded']={key:json.loads(value.descriptor()) for key,value in engine._models.items()
                                 if hasattr(value,'descriptor')}
        if args.device=='cuda':
            assert all(item['device']=='cuda' for item in report['models_loaded'].values())
            query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
            report['own_gpu_mib']=[float(line.split(',')[1]) for line in query.stdout.splitlines()
                if line.split(',')[0].strip()==str(os.getpid())]
            assert report['own_gpu_mib'] and max(report['own_gpu_mib'])>0
            if sys.platform.startswith('linux'):
                maps=Path('/proc/self/maps').read_text()
                report['cuda_libraries']=sorted({line.split()[-1] for line in maps.splitlines()
                    if any(name in line for name in ['libcublas','libcudnn','libcudart'])})
                assert report['cuda_libraries'] and all(Path(path).is_relative_to(Path(sys.prefix))
                    for path in report['cuda_libraries']),report['cuda_libraries']
        assert not any(name in sys.modules for name in forbidden),[name for name in forbidden if name in sys.modules]
    report['success']=True
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
