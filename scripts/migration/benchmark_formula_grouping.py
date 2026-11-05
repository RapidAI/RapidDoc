"""Replay all real formula tensors; test geometry grouping without changing text."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time
import numpy as np
import onnxruntime as ort

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.post_process import PPPostProcess
directory = root/'build/formula-tensor-audit-v3-20261029/native-model'
bounded = len(sys.argv)>1 and sys.argv[1]=='bounded'
destination = root/'build'/('formula-grouping-bounded-20261029.json' if bounded else 'formula-grouping-20261029.json')
assert not destination.exists()
model = root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx'
assert hashlib.file_digest(model.open('rb'), 'sha256').hexdigest() == '52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d'
options = ort.SessionOptions()
options.enable_cpu_mem_arena = True
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.log_severity_level = 4
session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
name = session.get_inputs()[0].name
decode = PPPostProcess(json.loads(session.get_modelmeta().custom_metadata_map['character']))
jobs = []
for p in sorted(directory.glob('*.input.json')):
    meta = json.loads(p.read_text())
    raw = p.with_name(p.name.replace('.input.json','.f32')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == meta['sha256']
    tensor = np.frombuffer(raw, dtype='<f4').reshape(meta['shape'])
    expected = json.loads(p.with_name(p.name.replace('.input.json','.tokens.json')).read_text())
    text = decode([np.array(expected['values'], dtype=np.int64)])[0]
    rgb = meta['sources'][0]
    jobs.append(dict(tensor=tensor, text=text, ratio=rgb['width']/rgb['height']))
assert len(jobs) == 74
session.run(None, {name:jobs[0]['tensor']})
counts = [int(n) for n in re.findall(r'03: page \d+/\d+ prepared \(layout=\d+ formulas=(\d+)',
    (root/'build/speed-gap-rust-before-20261029.txt').read_text(encoding='utf-8'))]
assert len(counts)==10 and sum(counts)==74
trials = []
sequence = [('single',1),('two_page_ratio',2),('two_page_ratio',2),('single',1)] if bounded else [('single',1), ('ratio',4), ('ratio',2), ('single',1)]
for mode, size in sequence:
    groups = [list(range(len(jobs)))]
    if mode == 'two_page_ratio':
        groups=[]; offset=0
        for i in range(0,len(counts),2):
            count=sum(counts[i:i+2]);groups.append(list(range(offset,offset+count)));offset+=count
    batches=[]
    for indices in groups:
        if mode != 'single':indices.sort(key=lambda i:jobs[i]['ratio'])
        batches.extend((indices[i:i+size], np.concatenate([jobs[k]['tensor'] for k in indices[i:i+size]], axis=0))
                       for i in range(0,len(indices),size))
    elapsed = 0.
    cpu = time.process_time()
    for batch, tensor in batches:
        started = time.perf_counter()
        output = session.run(None, {name:tensor})[0]
        elapsed += time.perf_counter()-started
        texts = decode([row.reshape(-1) for row in output])
        assert len(texts) == len(batch)
        for index, text in zip(batch, texts):
            assert text == jobs[index]['text'], (mode,size,index,text,jobs[index]['text'])
    record = dict(mode=mode, batch_size=size, inference_seconds=elapsed, cpu_seconds=time.process_time()-cpu,
                  all_74_outputs_match=True, inference_calls=len(batches))
    trials.append(record)
    destination.write_text(json.dumps(dict(trials=trials, diagnostic_not_pdf_speed=True), indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
