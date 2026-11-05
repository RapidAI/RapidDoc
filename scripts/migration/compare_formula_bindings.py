"""Same-process alternating ORT Python/C API diagnostic; no PDFium is loaded.

Native timing includes JSON parsing, unlike the production native PDF pipeline.
It is an upper-bound diagnostic, not an end-to-end speed claim.
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
import rapid_doc_rust as sdk

root = Path(__file__).resolve().parents[2]
destination = root/'build'/sys.argv[1]
assert not destination.exists()
from rapid_doc_rust._runtime import MANIFEST
manifest = MANIFEST['formula']
model = root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'
digest = hashlib.file_digest(model.open('rb'), 'sha256').hexdigest()
assert digest == manifest['sha256']
directory = root/'demo/output/rust/20261029-155105-653606/03/auto/images'
sources = []
for path in sorted(directory.glob('*.png')):
    with Image.open(path) as im:
        if im.width >= 100 and im.height <= 150:
            sources.append(path)
assert len(sources) >= 3
sources = sources[:3]
options = ort.SessionOptions()
options.enable_cpu_mem_arena = True
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 4
python = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
native = sdk.OnnxSession(root/'.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll', model, digest,
                         threads=0, max_elements=32_000_000)
name = python.get_inputs()[0].name
jobs = []
for source in sources:
    metadata, _, raw = sdk.prepare_formula_input_bytes(source.read_bytes(), 384, 384)
    shape = json.loads(metadata)['shape']
    array = np.frombuffer(raw, dtype='<f4').reshape(shape)
    payload = json.dumps({name:dict(shape=shape, values=array.reshape(-1).tolist())}, allow_nan=False)
    expected = python.run(None, {name:array})[0]
    def checked_native(payload=payload, expected=expected):
        output = json.loads(native._native.run_named_json(payload))
        result = next(iter(output.values()))
        assert result['dtype']=='int64' and result['shape']==list(expected.shape)
        assert result['values']==expected.reshape(-1).tolist()
    checked_native()
    jobs.append((array, checked_native))
samples = []
for backend in ['python', 'native', 'native', 'python'] * 2:
    cpu = time.process_time()
    started = time.perf_counter()
    for array, checked_native in jobs:
        if backend == 'python':
            python.run(None, {name:array})
        else:
            checked_native()
    samples.append(dict(backend=backend, seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu))
result = dict(ort_version=ort.__version__, model_sha256=digest,
              sources=[str(p.relative_to(root)) for p in sources], samples=samples,
              native_tokens_match_python=True, native_includes_json_overhead=True)
destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
