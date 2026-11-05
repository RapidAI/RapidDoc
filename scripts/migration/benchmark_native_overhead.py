"""Compare unchanged formula/table component inputs on installed or dev SDKs."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import hashlib,json,statistics,subprocess,sys,time
from pathlib import Path

root=Path(__file__).resolve().parents[2]
if '--dev' in sys.argv:sys.path.insert(0,str(root/'build/api-dev-sdk'))
import rapid_doc_rust as sdk

formula=json.loads((root/'build/migration/formula-preprocess.json').read_text(encoding='utf-8'))
inputs=[((root/c['source']).read_bytes(),c['target_size'])for c in formula['cases']]
tables=json.loads((root/'build/migration/native-table-materialization.json').read_text(encoding='utf-8'))
requests=[json.dumps(c['request'],ensure_ascii=False,allow_nan=False)for c in tables['renders']]

def formulas():
    h=hashlib.sha256()
    for png,(height,width) in inputs:
        r=sdk.prepare_formula_input(png,height=height,width=width)
        h.update(r['rgb']);h.update(r['tensor'])
    return h.hexdigest()

def cells():
    h=hashlib.sha256()
    for q in requests:
        h.update(sdk.legacy_pipeline_json('render_native_table_html_with_scripts',q).encode('utf-8'))
    return h.hexdigest()

if '--worker' in sys.argv:
    print('ready',flush=True)
    for line in sys.stdin:
        fn={'formula':formulas,'table':cells}[line.strip()]
        started=time.perf_counter();digest=fn();elapsed=time.perf_counter()-started
        print(json.dumps(dict(seconds=elapsed,sha256=digest)),flush=True)
    raise SystemExit

if '--paired' in sys.argv:
    workers={key:subprocess.Popen([sys.executable,__file__,'--worker',*(['--dev']if key=='new'else[])],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')for key in ['old','new']}
    results={key:{stage:[]for stage in ['formula','table']}for key in workers}
    try:
        for worker in workers.values():assert worker.stdout.readline().strip()=='ready'
        for stage in ['formula','table']:
            digest=None
            for round_ in range(8):
                for key in (['old','new']if round_%2==0 else['new','old']):
                    worker=workers[key];worker.stdin.write(stage+'\n');worker.stdin.flush()
                    result=json.loads(worker.stdout.readline())
                    if digest is None:digest=result['sha256']
                    assert result['sha256']==digest
                    if round_>0:results[key][stage].append(result['seconds'])
            print(stage,{key:statistics.median(results[key][stage])for key in workers},flush=True)
    finally:
        for worker in workers.values():worker.stdin.close();assert worker.wait()==0
    path=root/'build/overhead-paired-workers.json';assert not path.exists()
    path.write_text(json.dumps(results,indent=2),encoding='utf-8')
    raise SystemExit

results={}
for name,fn in [('formula_preprocessing_88',formulas),('table_materialization_526',cells)]:
    expected=fn();samples=[]
    for _ in range(3):
        started=time.perf_counter();assert fn()==expected;samples.append(time.perf_counter()-started)
    results[name]=dict(seconds=samples,median_seconds=statistics.median(samples),sha256=expected)
    print(name,results[name],flush=True)
filename=sys.argv[sys.argv.index('--output')+1]if '--output'in sys.argv else('overhead-after.json'if '--dev'in sys.argv else'overhead-before.json')
assert filename.endswith('.json') and Path(filename).name==filename
destination=root/'build'/filename
assert not destination.exists(),f'Keep benchmark evidence: {destination}'
destination.write_text(json.dumps(results,indent=2),encoding='utf-8')
