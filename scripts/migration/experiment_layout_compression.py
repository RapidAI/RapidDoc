"""Independent V2/V3 layout compression and real-page diagnostics. No default replacement."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from build_formula_compression import digest, graphs, store_half

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'v2': ('pp_doclayoutv2.onnx', '9fedca3a2ebfdce73fc36f7842a4cabe0a4fe8c0ee33a446ca358ed500907b29'),
    'v3': ('pp_doclayoutv3.onnx', 'b9a2759e51ee2cc9d98f10cea21caf5862043b882b5f57c683eaf6fa247196a3'),
}


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                              default=lambda value: value.tolist()), encoding='utf-8')


def prepare(destination):
    import cv2
    import pypdfium2 as pdfium
    destination.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((ROOT/'build/shared-arena-reference-python-20261029.json').read_text(encoding='utf-8'))
    cases = []
    for source in fixture['cases']:
        path = ROOT/source['source']
        assert digest(path) == source['sha256']
        with pdfium.PdfDocument(path) as document:
            for index in sorted({0, len(document)-1}):
                page = document[index]
                bitmap = page.render(scale=2)
                pil = bitmap.to_pil()
                image = np.array(pil.convert('RGB'), copy=True)
                pil.close()
                bitmap.close()
                page.close()
                height, width = image.shape[:2]
                resized = cv2.resize(image, (800, 800), interpolation=cv2.INTER_CUBIC)
                tensor = np.ascontiguousarray((resized.astype(np.float32)*(1/255.0)).transpose(2, 0, 1)[None])
                name = f'{len(cases):03d}'
                np.savez_compressed(destination/(name+'.npz'), image=tensor,
                                    im_shape=np.array([[800, 800]], np.float32),
                                    scale_factor=np.array([[800/height, 800/width]], np.float32))
                cases.append(dict(id=name, source=source['source'], pdf_sha256=source['sha256'],
                                  page=index, width=width, height=height, input_sha256=digest(destination/(name+'.npz'))))
    dump(destination/'manifest.json', dict(cases=cases, render_scale=2, color='RGB', resize='INTER_CUBIC', normalization='FP32 / 255'))
    print('Frozen pages:', len(cases), flush=True)


def build(kind, destination, precision='fp16-storage', batch_one=False, half_ops=None):
    import onnx
    name, sha = SOURCES[kind]
    source = ROOT.parent/'models'/name
    assert digest(source) == sha
    destination.mkdir(parents=True, exist_ok=False)
    model = onnx.load(source)
    weights = {value.name for graph in graphs(model.graph) for value in graph.initializer
               if value.data_type == onnx.TensorProto.FLOAT}
    if half_ops:
        consumers = {}
        for graph in graphs(model.graph):
            for node in graph.node:
                for value in node.input:
                    if value in weights:
                        consumers.setdefault(value, set()).add(node.op_type)
        weights = {name for name in weights if consumers.get(name) and consumers[name] <= set(half_ops)}
    exclusions = []
    for graph in graphs(model.graph):
        for value in graph.initializer:
            if value.name in weights:
                array = onnx.numpy_helper.to_array(value)
                if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float16).max):
                    exclusions.append(value.name)
    # Large negative mask constants must stay FP32, not overflow or be clipped.
    converted = store_half(model, weights, set(exclusions)) if precision == 'fp16-storage' else []
    if batch_one:
        for value in model.graph.input:
            assert value.name in ('image', 'im_shape', 'scale_factor')
            assert len(value.type.tensor_type.shape.dim) >= 1
            value.type.tensor_type.shape.dim[0].dim_value = 1
    suffix = precision + ('.batch1' if batch_one else '')
    target = destination/(source.stem+'.'+suffix+'.onnx')
    onnx.save_model(model, target)
    del model
    onnx.checker.check_model(str(target))
    assert digest(source) == sha
    dump(destination/'manifest.json', dict(kind=kind, source=str(source), source_sha256=sha,
         candidate=str(target), candidate_sha256=digest(target), source_bytes=source.stat().st_size,
         candidate_bytes=target.stat().st_size, weights=converted, preserved_fp32_constants=exclusions, compute='FP32',
         storage='FP16' if converted else 'FP32', batch_one_only=batch_one,
         half_storage_consumer_ops=half_ops,
         lossy_weight_storage=bool(converted), source_modified=False, installed_model_modified=False, onnx_checker_passed=True))
    print(kind, 'MiB', round(source.stat().st_size/2**20, 2), '->', round(target.stat().st_size/2**20, 2), flush=True)


def worker(model, inputs, destination, kind):
    import onnxruntime as ort
    from validate_formula_projection import memory
    from runtime_telemetry import Telemetry
    sys.path.insert(0, str(ROOT))
    from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.main import PPDocLayoutModelHandler
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    for case in manifest['cases']:
        assert digest(inputs/(case['id']+'.npz')) == case['input_sha256']
    telemetry = Telemetry(destination/'telemetry.jsonl')
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.enable_cpu_mem_arena = True
        started = time.perf_counter()
        session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        load_seconds = time.perf_counter()-started
        labels = session.get_modelmeta().custom_metadata_map['character'].splitlines()
        handler = PPDocLayoutModelHandler(labels, .5, .5, None, ModelType('pp_doclayout'+kind), 'auto')
        with np.load(inputs/(manifest['cases'][0]['id']+'.npz')) as stored:
            session.run(None, {key: stored[key] for key in stored.files})
        rows = []
        for case in manifest['cases']:
            telemetry.document = case['id']
            with np.load(inputs/(case['id']+'.npz')) as stored:
                feeds = {key: stored[key] for key in stored.files}
            cpu = time.process_time()
            started = time.perf_counter()
            result = session.run(None, feeds)
            seconds = time.perf_counter()-started
            cpu_seconds = time.process_time()-cpu
            np.savez_compressed(destination/(case['id']+'.npz'), **{f'output_{i}': value for i, value in enumerate(result)})
            processed = handler.pp_postprocess(result[0].copy(), [case['width'], case['height']],
                                               result[2].copy() if len(result) == 3 else None, 'auto')
            dump(destination/(case['id']+'.json'), processed)
            rows.append(dict(id=case['id'], source=case['source'], page=case['page'], seconds=seconds,
                             cpu_seconds=cpu_seconds, output_shapes=[list(value.shape) for value in result],
                             postprocessed_count=len(processed), memory=memory()))
            print(kind, destination.name, len(rows), '/', len(manifest['cases']), round(seconds, 3), flush=True)
        dump(destination/'report.json', dict(model=str(model), model_sha256=digest(model), kind=kind,
             ort_version=ort.__version__, profiling=False, threads=0, load_seconds=load_seconds,
             total_seconds=sum(row['seconds'] for row in rows), cases=rows, memory=memory(),
             postprocess=dict(threshold=.5, shape_mode='auto', implementation='existing Python PPDocLayout handler')))
    finally:
        telemetry.close()


def compare(baseline, candidate, output):
    assert not output.exists()
    a = json.loads((baseline/'report.json').read_text(encoding='utf-8'))
    b = json.loads((candidate/'report.json').read_text(encoding='utf-8'))
    assert a['kind'] == b['kind'] and [r['id'] for r in a['cases']] == [r['id'] for r in b['cases']]
    rows = []
    for case in a['cases']:
        name = case['id']
        left = json.loads((baseline/(name+'.json')).read_text(encoding='utf-8'))
        right = json.loads((candidate/(name+'.json')).read_text(encoding='utf-8'))
        structural_left = [{k: v for k, v in row.items() if k != 'score'} for row in left]
        structural_right = [{k: v for k, v in row.items() if k != 'score'} for row in right]
        row = dict(id=name, source=case['source'], page=case['page'], postprocess_exact=left == right,
                   postprocess_except_scores_exact=structural_left == structural_right,
                   original_count=len(left), candidate_count=len(right), raw={})
        with np.load(baseline/(name+'.npz')) as old, np.load(candidate/(name+'.npz')) as new:
            assert old.files == new.files
            for key in old.files:
                x, y = old[key], new[key]
                item = dict(original_shape=list(x.shape), candidate_shape=list(y.shape), exact=np.array_equal(x, y))
                if x.shape == y.shape:
                    item['different_elements'] = int(np.count_nonzero(x != y))
                    if np.issubdtype(x.dtype, np.floating):
                        item['max_absolute_difference'] = float(np.max(np.abs(x-y))) if x.size else 0.0
                row['raw'][key] = item
        rows.append(row)
    result = dict(kind=a['kind'], original_seconds=a['total_seconds'], candidate_seconds=b['total_seconds'],
                  observed_time_reduction_percent=100*(1-b['total_seconds']/a['total_seconds']),
                  original_memory=a['memory'], candidate_memory=b['memory'], cases=rows,
                  exact_postprocessed_pages=sum(row['postprocess_exact'] for row in rows),
                  structural_match_pages=sum(row['postprocess_except_scores_exact'] for row in rows),
                  labeled_accuracy_evaluated=False, default_model_replaced=False)
    dump(output, result)
    print(a['kind'], 'exact postprocess', result['exact_postprocessed_pages'], '/', len(rows),
          'structure', result['structural_match_pages'], 'time reduction', round(result['observed_time_reduction_percent'], 2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'build', 'worker', 'compare'])
    parser.add_argument('--kind', choices=['v2', 'v3'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--precision', choices=['fp32', 'fp16-storage'], default='fp16-storage')
    parser.add_argument('--batch-one', action='store_true')
    parser.add_argument('--half-ops', nargs='+', choices=['MatMul', 'Conv'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output.resolve())
    elif args.action == 'build':
        build(args.kind, args.output.resolve(), args.precision, args.batch_one, args.half_ops)
    elif args.action == 'worker':
        worker(args.model.resolve(), args.inputs.resolve(), args.output.resolve(), args.kind)
    else:
        compare(args.baseline.resolve(), args.candidate.resolve(), args.output.resolve())
