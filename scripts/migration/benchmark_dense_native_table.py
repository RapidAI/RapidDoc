"""General native table geometry/recovery/render timing on saved identical input."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
from pathlib import Path
import hashlib,json,statistics,subprocess,sys,time
root=Path(__file__).resolve().parents[2]
if '--dev'in sys.argv:sys.path.insert(0,str(root/'build/api-dev-sdk'))
import rapid_doc_rust as sdk
from rapid_doc_rust._native import call_host_page
directory=root/'build/dense-table-overhead-20261029'
if '--prepare'in sys.argv:
    directory.mkdir(exist_ok=False)
    started=time.perf_counter()
    page=json.loads(sdk.native_pdf_table_page_json(str(Path(sdk.__file__).parent/'runtime/rapiddoc_pdfium.dll'),(root/'demo/pdfs/49－长表格-密集型.pdf').read_bytes(),0,None))
    print('Geometry preparation',time.perf_counter()-started,flush=True)
    request=dict(table_input={k:v for k,v in page.items()if k not in ['tight_bboxes','origins']},tight_bboxes=page['tight_bboxes'],origins=page['origins'])
    request['table_input'].update(table_bbox=[48.82,55.66,544.57,786.77],angle=0)
    (directory/'request.json').write_text(json.dumps(request,allow_nan=False),encoding='utf-8')
    result=json.loads(call_host_page(sdk.legacy_pipeline_json,('recover_native_table_result',json.dumps(request,allow_nan=False)),{}))
    assert result and 'recovery_error'not in result,result
    (directory/'expected.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    request['result']=result['result']
    (directory/'render.json').write_text(json.dumps(request,allow_nan=False),encoding='utf-8')
    raise SystemExit
expected=json.loads((directory/'expected.json').read_text(encoding='utf-8'))
if '--worker'in sys.argv:
    requests={key:(operation,(directory/file).read_text(encoding='utf-8'))for key,operation,file in [('recover','recover_native_table_result','request.json'),('render','render_native_table_html_with_scripts','render.json'),('engine','recover_native_table_primitives','request.json')]}
    print('ready',flush=True)
    for line in sys.stdin:
        key=line.strip();operation,request=requests[key];started=time.perf_counter()
        raw=call_host_page(sdk.legacy_pipeline_json,(operation,request),{});elapsed=time.perf_counter()-started
        assert json.loads(raw)==(expected if key=='recover'else expected['result']if key=='engine'else expected['content'])
        print(json.dumps(dict(seconds=elapsed,sha256=hashlib.sha256(raw.encode()).hexdigest())),flush=True)
    raise SystemExit
if '--paired'in sys.argv:
    stages=sys.argv[sys.argv.index('--stages')+1].split(',')if '--stages'in sys.argv else['recover','render']
    assert stages and set(stages)<= {'recover','render','engine'}
    workers={key:subprocess.Popen([sys.executable,__file__,'--worker',*(['--dev']if key=='new'else[])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')for key in ['old','new']}
    results={key:{stage:[]for stage in stages}for key in workers}
    try:
        for worker in workers.values():assert worker.stdout.readline().strip()=='ready'
        for stage in stages:
            digest=None
            for iteration in range(6):
                for key in (['old','new']if iteration%2==0 else['new','old']):
                    worker=workers[key];worker.stdin.write(stage+'\n');worker.stdin.flush();value=json.loads(worker.stdout.readline())
                    if digest is None:digest=value['sha256']
                    assert digest==value['sha256']
                    if iteration:results[key][stage].append(value['seconds'])
            print(stage,{key:statistics.median(results[key][stage])for key in workers},flush=True)
    finally:
        for worker in workers.values():worker.stdin.close();assert worker.wait()==0
    filename=sys.argv[sys.argv.index('--output')+1]if '--output'in sys.argv else'paired.json'
    assert filename.endswith('.json')and Path(filename).name==filename
    path=directory/filename;assert not path.exists();path.write_text(json.dumps(results,indent=2))
    raise SystemExit
results={}
for key,operation,file in [('recover','recover_native_table_result','request.json'),('render','render_native_table_html_with_scripts','render.json')]:
    request=(directory/file).read_text(encoding='utf-8');samples=[]
    for _ in range(3):
        started=time.perf_counter();raw=call_host_page(sdk.legacy_pipeline_json,(operation,request),{});samples.append(time.perf_counter()-started)
        result=json.loads(raw);assert result==(expected if key=='recover'else expected['content'])
    results[key]=dict(seconds=samples,median_seconds=statistics.median(samples),request_sha256=hashlib.sha256(request.encode()).hexdigest())
    print(key,results[key],flush=True)
path=directory/('after.json'if '--dev'in sys.argv else'before.json');assert not path.exists()
path.write_text(json.dumps(results,indent=2),encoding='utf-8')
