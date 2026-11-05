"""Capture real ONNX inputs, then benchmark identical graphs outside PDF parsing.

The Rust counterpart is experiments/onnx-probe/src/bin/model_benchmark.rs.
JSON, file IO, preprocessing, decoding and model loading are outside run timing.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import statistics
import shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def capture(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    os.environ['MINERU_DEVICE_MODE'] = 'cuda'
    ort.preload_dlls(directory='')
    args.output.mkdir(parents=True, exist_ok=True)
    initialize, run = ort.InferenceSession.__init__, ort.InferenceSession.run
    seen = set()

    def initialized(self, source, *pos, **kw):
        initialize(self, source, *pos, **kw)
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        hint = str(source) if not isinstance(source, bytes) else None
        if hint is None:
            for context in inspect.stack():
                for key in ('model', 'model_path', 'source'):
                    value = context.frame.f_locals.get(key)
                    if isinstance(value, (str, Path)) and str(value).endswith('.onnx') and Path(value).is_file():
                        hint = str(value)
                        break
                if hint:
                    break
        self._benchmark_source = raw, hint

    def save(self, inputs, outputs):
        raw, hint = self._benchmark_source
        sha = hashlib.sha256(raw).hexdigest()
        if sha in seen or any(value.dtype != np.float32 for value in inputs.values()):
            return
        seen.add(sha)
        directory = (args.output/sha[:16]).resolve()
        if (directory/'case.json').exists():
            return
        directory.mkdir(exist_ok=True)
        model = directory/'model.onnx'
        model.write_bytes(raw)
        graph = onnx.load_model_from_string(raw)
        libraries = []
        if any(v.domain == 'onnxruntime_ops' for v in graph.opset_import):
            from rapid_doc.backend.onnx_ops.native import library_path
            libraries.append(str(library_path().resolve()))
        from rapid_doc_rust._runtime import runtime_path
        case = dict(runtime=str(runtime_path()), model=str(model), sha256=sha,
                    origin=hint, production_providers=self.get_providers(),
                    operator_libraries=libraries, inputs=[], expected=[])
        for index, (name, value) in enumerate(inputs.items()):
            path = directory/f'input-{index}.f32'
            np.ascontiguousarray(value, dtype='<f4').tofile(path)
            case['inputs'].append(dict(name=name, shape=list(value.shape), file=str(path)))
        for index, (name, value) in enumerate(zip([v.name for v in self.get_outputs()], outputs)):
            path = directory/f'expected-{index}.bin'
            np.ascontiguousarray(value).tofile(path)
            case['expected'].append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype), file=str(path)))
        write_json(directory/'case.json', case)
        print('captured', hint, {k: list(v.shape) for k, v in inputs.items()}, flush=True)

    def called(self, names, inputs, *pos, **kw):
        result = run(self, names, inputs, *pos, **kw)
        if names is None or list(names) == [v.name for v in self.get_outputs()]:
            save(self, inputs, result)
        return result
    ort.InferenceSession.__init__, ort.InferenceSession.run = initialized, called
    # Layout normally uses IO Binding, so collect at its adapter boundary too.
    from rapid_doc.model.layout.rapid_layout_self.inference_engine.onnxruntime.main import OrtInferSession
    layout_call = OrtInferSession.__call__
    def layout(self, image, scale_factor=None):
        result = layout_call(self, image, scale_factor)
        if scale_factor is not None:
            shape = np.array([list(image.shape[-2:])]*image.shape[0], dtype=np.float32)
            save(self.session, dict(image=image, im_shape=shape, scale_factor=scale_factor), result)
        return result
    OrtInferSession.__call__ = layout
    from rapidocr import EngineType
    from rapid_doc import RapidDoc
    engine_config = dict(engine_cfg=dict(use_cuda=True))
    ocr = {'engine_type': EngineType.ONNXRUNTIME, 'Det.engine_type': EngineType.ONNXRUNTIME,
           'Rec.engine_type': EngineType.ONNXRUNTIME, 'EngineConfig.onnxruntime.use_cuda': True}
    engine = RapidDoc(layout_config=engine_config, formula_config=engine_config,
                      table_config=engine_config, ocr_config=ocr)
    engine(args.source, start_page_id=args.start, end_page_id=args.end)


def benchmark_python(args):
    import numpy as np
    import onnxruntime as ort
    ort.preload_dlls(directory='')
    case = json.loads(args.case.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for library in case['operator_libraries']:
        options.register_custom_ops_library(library)
    started = time.perf_counter()
    session = ort.InferenceSession(case['model'], sess_options=options, providers=['CUDAExecutionProvider'])
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA provider failed')
    load_seconds = time.perf_counter()-started
    inputs = {v['name']: np.fromfile(v['file'], dtype='<f4').reshape(v['shape']) for v in case['inputs']}
    times = []
    for index in range(13):
        started = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed = time.perf_counter()-started
        if index >= 3:
            times.append(elapsed)
    descriptions = []
    for index, (name, value) in enumerate(zip([v.name for v in session.get_outputs()], outputs)):
        filename = f'output-{index}.bin'
        np.ascontiguousarray(value).tofile(args.output/filename)
        descriptions.append(dict(name=name, dtype=str(value.dtype), shape=list(value.shape), file=filename))
    write_json(args.output/'report.json', dict(engine='python', warmup=3, seconds=times,
               load_seconds=load_seconds, threads=1, outputs=descriptions,
               scope='ORT run with prebuilt CPU tensors, CPU input transfer and CPU outputs; excludes preprocessing, decoding, JSON and file IO'))


def compare(args):
    rows = []
    for file in sorted(args.cases.glob('*/case.json')):
        case = json.loads(file.read_text())
        key = file.parent.name
        if key in args.exclude_case:
            continue
        folders = {engine: args.results/key/engine for engine in ('python', 'rust')}
        reports = {e: json.loads((folder/'report.json').read_text()) for e, folder in folders.items()}
        outputs = {e: {v['name']: v for v in r['outputs']} for e, r in reports.items()}
        exact = outputs['python'].keys() == outputs['rust'].keys()
        for name, py in outputs['python'].items():
            rs = outputs['rust'].get(name)
            exact = exact and rs is not None and py['dtype'] == rs['dtype'] and py['shape'] == rs['shape']
            if rs is not None:
                exact = exact and (folders['python']/py['file']).read_bytes() == (folders['rust']/rs['file']).read_bytes()
        row = dict(case=key, origin=case['origin'], production_providers=case['production_providers'],
                   benchmark_variant=case.get('benchmark_variant','captured_graph'),
                   inputs={v['name']: v['shape'] for v in case['inputs']}, exact_outputs=exact,
                   median_seconds={e: statistics.median(r['seconds']) for e, r in reports.items()})
        rows.append(row)
        print(Path(case['origin'] or key).name, row['median_seconds'], 'exact', exact)
    write_json(args.results/'summary.json', rows)
    write_json(args.results/'excluded-cases.json', args.exclude_case)
    if not rows or not all(r['exact_outputs'] for r in rows):
        raise RuntimeError('Independent model output comparison failed')


def monolithic(args):
    """Keep a split decoder prefill separate from a whole-model control case."""
    import onnx
    case = json.loads(args.case.read_text())
    source = Path(case['origin'])
    raw = source.read_bytes()
    graph = onnx.load_model_from_string(raw)
    if {v.name for v in graph.graph.input} != {v['name'] for v in case['inputs']}:
        raise ValueError('Reference graph does not accept the same captured inputs')
    sha = hashlib.sha256(raw).hexdigest()
    directory = (args.output/sha[:16]).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'model.onnx').write_bytes(raw)
    for index, value in enumerate(case['inputs']):
        destination = directory/f'input-{index}.f32'
        shutil.copyfile(value['file'], destination)
        value['file'] = str(destination)
    case.update(model=str(directory/'model.onnx'), sha256=sha, operator_libraries=[], expected=[],
                benchmark_variant='monolithic_original_control', split_graph_case=str(args.case.resolve()))
    write_json(directory/'case.json', case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('capture')
    collect.add_argument('--source', type=Path, required=True)
    collect.add_argument('--start', type=int, default=0)
    collect.add_argument('--end', type=int, default=1)
    collect.add_argument('--output', type=Path, required=True)
    bench = commands.add_parser('python')
    bench.add_argument('--case', type=Path, required=True)
    bench.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('compare')
    check.add_argument('--cases', type=Path, required=True)
    check.add_argument('--results', type=Path, required=True)
    check.add_argument('--exclude-case', action='append', default=[])
    reference = commands.add_parser('monolithic')
    reference.add_argument('--case', type=Path, required=True)
    reference.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'capture': capture, 'python': benchmark_python, 'compare': compare, 'monolithic': monolithic}[args.command](args)


if __name__ == '__main__':
    main()
