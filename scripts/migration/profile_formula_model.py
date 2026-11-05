"""Read-only model investigation on frozen real tensors, not a PDF benchmark."""
from pathlib import Path
import hashlib
import json
import time
import argparse
import numpy as np
import onnxruntime as ort

root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,default=root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx')
parser.add_argument('--sha256',default='52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d')
parser.add_argument('--output',type=Path,default=root/'build/formula-model-analysis-20261029')
args=parser.parse_args()
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=True)
assert out.is_dir() and not (out/'profile-runs.json').exists()
source=root/'build/formula-tensor-audit-v3-20261029/native-model'
jobs=[]
for p in sorted(source.glob('*.tokens.json')):
    metadata=json.loads(p.read_text())
    jobs.append((metadata['shape'][1],p,metadata))
assert len(jobs)==74
short=min(jobs,key=lambda x:x[0]);long=max(jobs,key=lambda x:x[0])
model=args.model.resolve()
with model.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==args.sha256
options=ort.SessionOptions()
options.enable_cpu_mem_arena=True
options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level=4
options.enable_profiling=True
options.profile_file_prefix=str(out/'operators')
options.optimized_model_filepath=str(out/'optimized.onnx')
options.add_session_config_entry('session.optimized_model_external_initializers_file_name','optimized-weights.bin')
options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes','1024')
session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
name=session.get_inputs()[0].name
runs=[]
for label,job in [('warmup_short',short),('short',short),('long',long)]:
    length,p,expected=job
    meta=json.loads(p.with_name(p.name.replace('.tokens.json','.input.json')).read_text())
    raw=p.with_name(p.name.replace('.tokens.json','.f32')).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==meta['sha256']
    tensor=np.frombuffer(raw,dtype='<f4').reshape(meta['shape'])
    cpu=time.process_time();started=time.perf_counter()
    result=session.run(None,{name:tensor})[0]
    seconds=time.perf_counter()-started
    assert list(result.shape)==expected['shape'] and result.reshape(-1).tolist()==expected['values']
    runs.append(dict(label=label,source=p.name,tokens=length,seconds=seconds,cpu_seconds=time.process_time()-cpu,
                     raw_tokens_match=True))
    print(json.dumps(runs[-1]),flush=True)
profile=Path(session.end_profiling())
with model.open('rb')as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==args.sha256
result=dict(runs=runs,profile_path=str(profile),profile_bytes=profile.stat().st_size,
            model_path=str(model),model_sha256=args.sha256,
            ort_version=ort.__version__,providers=session.get_providers(),profiling_overhead_included=True,
            original_model_unmodified=True)
(out/'profile-runs.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Profile saved:',profile.name,profile.stat().st_size,flush=True)
