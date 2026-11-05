"""Compare production table execution choices using captured identical tensors."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import onnxruntime as ort
    ort.preload_dlls(directory='')
    from rapid_doc.model.table.rapid_table_self.inference_engine.onnxruntime.optimization import partition_slanet
    from rapid_doc.backend.onnx_ops import create_session
    rows=[]
    for name in ['slanet-plus.onnx','unet.onnx']:
        case=next(json.loads(f.read_text()) for f in sorted(args.cases.glob('*/case.json'))
                  if Path(json.loads(f.read_text()).get('origin')or'').name==name)
        source=Path(case['origin']);inputs={v['name']:np.fromfile(v['file'],dtype='<f4').reshape(v['shape'])for v in case['inputs']}
        variants=['full_cuda','split_cuda_cpu'] if name.startswith('slanet') else ['full_cuda','cached_conv_cuda']
        previous=None
        for variant in variants:
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=4;options.enable_cpu_mem_arena=False
            decoder=None
            if variant=='split_cuda_cpu':
                encoder,decoded=partition_slanet(source)
                session=ort.InferenceSession(encoder,options,providers=['CUDAExecutionProvider'])
                decoder=ort.InferenceSession(decoded,options,providers=['CPUExecutionProvider'])
            elif variant=='cached_conv_cuda':
                session=create_session(source,providers=['CUDAExecutionProvider'],sess_options=options,conv_api='classic',reductions=False,fallback=False)
            else:session=ort.InferenceSession(str(source),options,providers=['CUDAExecutionProvider'])
            assert session.get_providers()[0]=='CUDAExecutionProvider'
            elapsed=[]
            for i in range(13):
                start=time.perf_counter();outputs=session.run(None,inputs)
                if decoder is not None:
                    encoded=dict(zip([v.name for v in session.get_outputs()],outputs))
                    outputs=decoder.run(None,{v.name:encoded[v.name]for v in decoder.get_inputs()})
                duration=time.perf_counter()-start
                if i>=3:elapsed.append(duration)
            exact=previous is None or all(np.array_equal(a,b)for a,b in zip(previous,outputs))
            difference=None if previous is None else [dict(count=int(np.count_nonzero(a!=b)),max_abs=float(np.max(np.abs(a-b))))for a,b in zip(previous,outputs)]
            previous=[v.copy()for v in outputs]
            directory=args.output/(name+'-'+variant);directory.mkdir()
            descriptions=[]
            for i,value in enumerate(outputs):
                filename=f'output-{i}.bin';np.ascontiguousarray(value).tofile(directory/filename)
                descriptions.append(dict(shape=list(value.shape),dtype=str(value.dtype),file=filename))
            row=dict(model=name,variant=variant,seconds=elapsed,median_seconds=statistics.median(elapsed),exact_against_full=exact,differences=difference,outputs=descriptions,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),threads=4)
            (directory/'report.json').write_text(json.dumps(row,indent=2));rows.append(row)
            print(name,variant,row['median_seconds'],'exact',exact,'differences',difference,flush=True)
            del session,decoder;gc.collect()
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
