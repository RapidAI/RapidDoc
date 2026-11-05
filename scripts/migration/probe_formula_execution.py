"""Isolated, identical-tensor formula execution diagnostic; no PDFium loading."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
from pathlib import Path
import hashlib,json,statistics,struct,sys,time

root=Path(__file__).resolve().parents[2]
directory=root/'build/formula-execution-20261029'
mode=sys.argv[1]
if mode=='prepare':
    import rapid_doc_rust as sdk
    directory.mkdir(exist_ok=False)
    f=json.loads((root/'build/migration/formula-decode.json').read_text(encoding='utf-8'))
    for i in range(3):
        p=sdk.prepare_formula_input(root/f'build/migration/formula-real/{i}.png')
        (directory/f'{i}.f32').write_bytes(p['tensor'])
    (directory/'model.json').write_text(json.dumps(f['model']),encoding='utf-8')
    raise SystemExit

backend,threads=mode.split('-');threads=int(threads)
f=json.loads((directory/'model.json').read_text())
model=root/f['path'];assert hashlib.sha256(model.read_bytes()).hexdigest()==f['sha256']
data=[(directory/f'{i}.f32').read_bytes()for i in range(3)]
if backend=='python':
    import numpy as np
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=threads
    options.enable_cpu_mem_arena=False
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    name=session.get_inputs()[0].name
    prepared=[np.frombuffer(d,dtype='<f4').reshape(1,1,384,384)for d in data]
    def call(i):
        output=session.run(None,{name:prepared[i]})[0]
        return [int(n)for n in output.reshape(-1)]
else:
    import rapid_doc_rust as sdk
    session=sdk.OnnxSession(root/'.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll',model,f['sha256'],threads=threads)
    # Input name obtained separately from the Python-only diagnostic.
    name=json.loads((directory/'input-name.json').read_text())
    prepared=[json.dumps({name:dict(shape=[1,1,384,384],values=[v[0]for v in struct.iter_unpack('<f',d)])},allow_nan=False)for d in data]
    def call(i):return next(iter(json.loads(session._native.run_named_json(prepared[i])).values()))['values']
if backend=='python':(directory/'input-name.json').write_text(json.dumps(name))
expected=[call(i)for i in range(3)];samples=[]
for _ in range(3):
    started=time.perf_counter()
    for i in range(3):assert call(i)==expected[i]
    samples.append(time.perf_counter()-started)
result=dict(mode=mode,seconds=samples,median_seconds=statistics.median(samples),tokens=expected,
    tensor_sha256=[hashlib.sha256(d).hexdigest()for d in data])
path=directory/f'{mode}.json';assert not path.exists();path.write_text(json.dumps(result),encoding='utf-8')
print(mode,'median for three identical tensors',result['median_seconds'],flush=True)
