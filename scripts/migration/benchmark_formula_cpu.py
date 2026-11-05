"""Identical-tensor CPU arena diagnostic in one isolated ORT process per run."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import onnxruntime as ort
root=Path(__file__).resolve().parents[2]
d=root/'build/formula-execution-20261029'
f=json.loads((d/'model.json').read_text()); model=root/f['path']
assert hashlib.sha256(model.read_bytes()).hexdigest()==f['sha256']
expected=json.loads((d/'python-0.json').read_text())['tokens']
arena=sys.argv[1]=='on'; output=root/'build'/sys.argv[2]; assert not output.exists()
options=ort.SessionOptions(); options.log_severity_level=4; options.enable_cpu_mem_arena=arena
options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
threads=int(sys.argv[3]) if len(sys.argv)>3 else 0
if threads: options.intra_op_num_threads=threads
scheduling=sys.argv[4] if len(sys.argv)>4 else 'default'
assert scheduling in ['default','dynamic','no_spin','static_batch1','bounded_spin']
if scheduling=='dynamic': options.add_session_config_entry('session.dynamic_block_base','4')
if scheduling=='static_batch1': options.add_free_dimension_override_by_name('DynamicDimension.0',1)
if scheduling=='bounded_spin':
    options.add_session_config_entry('session.intra_op.spin_duration_us','1000')
    options.add_session_config_entry('session.intra_op.spin_backoff_max','8')
if scheduling=='no_spin':
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
started=time.perf_counter(); session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
load=time.perf_counter()-started
if scheduling=='static_batch1': assert session.get_inputs()[0].shape[0]==1
inputs=[np.frombuffer((d/f'{i}.f32').read_bytes(),dtype='<f4').reshape(1,1,384,384) for i in range(3)]
name=session.get_inputs()[0].name
for i in range(3): assert session.run(None,{name:inputs[i]})[0].reshape(-1).tolist()==expected[i]
samples=[]
for _ in range(4):
    cpu=time.process_time(); started=time.perf_counter()
    for i in range(3): assert session.run(None,{name:inputs[i]})[0].reshape(-1).tolist()==expected[i]
    samples.append(dict(seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu))
result=dict(cpu_arena=arena,threads=threads,scheduling=scheduling,ort_version=ort.__version__,load_seconds=load,samples=samples,tokens_match_reference=True,
 model_sha256=f['sha256'],tensor_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in inputs])
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(output.name,[(round(x['seconds'],3),round(x['cpu_seconds'],3)) for x in samples],flush=True)
