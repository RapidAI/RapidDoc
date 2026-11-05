"""Save the original Python partition output for native same-tensor validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import onnxruntime as ort
    ort.preload_dlls(directory='')
    from rapid_doc.model.table.rapid_table_self.inference_engine.onnxruntime.optimization import partition_slanet
    case=json.loads(args.case.read_text());entry=json.loads(args.manifest.read_text())['models']['table'];base=args.manifest.resolve().parent
    if hashlib.sha256(Path(case['origin']).read_bytes()).hexdigest()!=entry['source_sha256']:raise ValueError('Source model differs')
    inputs={v['name']:np.fromfile(v['file'],dtype='<f4').reshape(v['shape'])for v in case['inputs']}
    encoder,decoder=partition_slanet(case['origin'])
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.enable_cpu_mem_arena=False
    enc=ort.InferenceSession(encoder,options,providers=['CUDAExecutionProvider'])
    dec=ort.InferenceSession(decoder,options,providers=['CPUExecutionProvider'])
    encoded=dict(zip([v.name for v in enc.get_outputs()],enc.run(None,inputs)))
    outputs=dec.run(None,{v.name:encoded[v.name]for v in dec.get_inputs()})
    descriptions=[]
    for i,(meta,value) in enumerate(zip(dec.get_outputs(),outputs)):
        name=f'output-{i}.bin';np.ascontiguousarray(value).tofile(args.output/name)
        descriptions.append(dict(name=meta.name,shape=list(value.shape),file=name))
    (args.output/'report.json').write_text(json.dumps(dict(outputs=descriptions),indent=2))
    descriptor=dict(runtime=case['runtime'],model=str(base/entry['model']),sha256=entry['sha256'],device='cuda',threads=4,
                    table_decoder=dict(model=str(base/entry['decoder']['model']),sha256=entry['decoder']['sha256']))
    (args.output/'descriptor.json').write_text(json.dumps(descriptor,indent=2))


if __name__=='__main__':main()
