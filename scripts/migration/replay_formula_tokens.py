"""Check whether real preprocessing differences change autoregressive output."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[2]
directory = root/'build/formula-tensor-audit-v3-20261029'
destination = directory/'token-comparison.json'
assert not destination.exists()
native = {json.loads(p.read_text())['sources'][0]['rgb_sha256']:p
          for p in (directory/'native-model').glob('*.input.json')}
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 4
model = root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'
assert hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() == '52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d'
session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
name = session.get_inputs()[0].name
rows = []
for p in sorted((directory/'python').glob('*.input.json')):
    meta = json.loads(p.read_text())
    q = native[meta['sources'][0]['rgb_sha256']]
    native_meta = json.loads(q.read_text())
    raw = p.with_name(p.name.replace('.input.json','.f32')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta['sha256']
    tensor = np.frombuffer(raw, dtype='<f4').reshape(meta['shape'])
    other = np.fromfile(q.with_name(q.name.replace('.input.json','.f32')), dtype='<f4').reshape(native_meta['shape'])
    expected = json.loads(q.with_name(q.name.replace('.input.json','.tokens.json')).read_text())
    started = time.perf_counter()
    actual = session.run(None, {name:tensor})[0]
    seconds = time.perf_counter()-started
    row = dict(index=p.name, native_index=q.name, input_bytes_equal=meta['sha256']==native_meta['sha256'],
               different_values=int(np.count_nonzero(tensor!=other)), max_absolute_difference=float(np.max(np.abs(tensor-other))),
               tokens_equal=list(actual.shape)==expected['shape'] and actual.reshape(-1).tolist()==expected['values'],
               python_shape=list(actual.shape), native_shape=expected['shape'], python_replay_seconds=seconds)
    rows.append(row)
    destination.write_text(json.dumps(dict(cases=rows, diagnostic_not_speed_comparison=True), indent=2), encoding='utf-8')
print(json.dumps(dict(cases=len(rows), identical_outputs=sum(r['tokens_equal'] for r in rows),
    identical_inputs=sum(r['input_bytes_equal'] for r in rows), max_absolute_difference=max(r['max_absolute_difference'] for r in rows))), flush=True)
