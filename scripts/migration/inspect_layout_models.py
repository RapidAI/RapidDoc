"""Inspect local ONNX layout graphs; optionally profile a real page without editing models."""
import argparse
import collections
import gc
import hashlib
import json
from pathlib import Path
import time

import onnx


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def specification(value):
    return dict(name=value.name, dtype=onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
                shape=[dim.dim_value if dim.HasField('dim_value') else dim.dim_param
                       for dim in value.type.tensor_type.shape.dim])


def inspect(path):
    model = onnx.load(path)
    operations = collections.Counter()
    graph_rows, weight_rows = [], []
    savings = 0

    def walk(graph, scope):
        nonlocal savings
        counts = collections.Counter(node.op_type for node in graph.node)
        operations.update(counts)
        graph_rows.append(dict(scope=scope, nodes=len(graph.node), operations=dict(counts)))
        for weight in graph.initializer:
            elements = 1
            for dim in weight.dims:
                elements *= dim
            if weight.data_type == onnx.TensorProto.FLOAT:
                weight_rows.append(dict(name=weight.name, scope=scope, shape=list(weight.dims), bytes=elements * 4))
                if elements >= 128:
                    savings += elements * 2
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    walk(attribute.g, scope+'/'+node.name+'/'+attribute.name)

    walk(model.graph, 'main')
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size,
                inputs=[specification(value) for value in model.graph.input],
                outputs=[specification(value) for value in model.graph.output],
                graphs=graph_rows, operations=dict(operations),
                estimated_half_storage_bytes=path.stat().st_size-savings,
                estimate_excludes_cast_metadata_and_does_not_guarantee_accuracy=True,
                largest_float_weights=sorted(weight_rows, key=lambda row: row['bytes'], reverse=True)[:10])


def profile(path, output, page_input):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_cpu_mem_arena = True
    options.enable_profiling = True
    options.profile_file_prefix = str(output/'operators')
    options.optimized_model_filepath = str(output/'optimized.onnx')
    options.add_session_config_entry('session.optimized_model_external_initializers_file_name', 'optimized-weights.bin')
    options.add_session_config_entry('session.optimized_model_external_initializers_min_size_in_bytes', '1024')
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    feeds = {value.name: page_input[value.name] for value in session.get_inputs()}
    started = time.perf_counter()
    warmup = session.run(None, feeds)
    warmup_seconds = time.perf_counter()-started
    started = time.perf_counter()
    actual = session.run(None, feeds)
    seconds = time.perf_counter()-started
    assert len(warmup) == len(actual) and all(np.array_equal(left, right) for left, right in zip(warmup, actual))
    np.savez_compressed(output/'outputs.npz', **{value.name: result for value, result in zip(session.get_outputs(), actual)})
    trace = Path(session.end_profiling())
    del session
    gc.collect()
    events = json.loads(trace.read_text(encoding='utf-8'))
    run = sorted([event for event in events if event.get('name') == 'model_run'], key=lambda event: event['ts'])[-1]
    nodes = [event for event in events if event.get('cat') == 'Node'
             and event.get('name', '').endswith('_kernel_time')
             and run['ts'] <= event['ts'] < run['ts']+run['dur']
             and event.get('args', {}).get('op_name') not in ('Loop', 'If', 'Scan')]
    operations = collections.Counter()
    for node in nodes:
        operations[node['args'].get('op_name', '?')] += node['dur']
    optimized = onnx.load(output/'optimized.onnx', load_external_data=False)
    return dict(warmup_seconds=warmup_seconds, seconds=seconds, profiling_overhead_included=True,
                ort_version=ort.__version__, providers=['CPUExecutionProvider'], representative_page_only=True,
                repeated_outputs_identical=True, trace=str(trace), leaf_calls=len(nodes),
                operation_ms={name: value/1000 for name, value in operations.most_common()},
                optimized_main_nodes=len(optimized.graph.node),
                optimized_operations=dict(collections.Counter(node.op_type for node in optimized.graph.node)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile-pdf', type=Path)
    parser.add_argument('--page', type=int, default=0)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    page_input = None
    if args.profile_pdf:
        import cv2
        import numpy as np
        import pypdfium2 as pdfium
        pdf_path = args.profile_pdf.resolve()
        with pdfium.PdfDocument(pdf_path) as document:
            page = document[args.page]
            bitmap = page.render(scale=2)
            pil = bitmap.to_pil()
            image = np.array(pil.convert('RGB'), copy=True)
            pil.close()
            bitmap.close()
            page.close()
        height, width = image.shape[:2]
        image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_CUBIC)
        image = np.ascontiguousarray((image.astype(np.float32)*(1/255.0)).transpose(2, 0, 1)[None])
        page_input = dict(image=image, im_shape=np.array([[800, 800]], np.float32),
                          scale_factor=np.array([[800/height, 800/width]], np.float32))
        np.savez_compressed(destination/'inputs.npz', **page_input)
        (destination/'input-source.json').write_text(json.dumps(dict(pdf=str(pdf_path), pdf_sha256=digest(pdf_path),
            page=args.page, render_scale=2, color='RGB', resize='INTER_CUBIC', shape=list(image.shape),
            image_tensor_sha256=hashlib.sha256(image.tobytes()).hexdigest()), indent=2), encoding='utf-8')
    reports = []
    for index, path in enumerate(args.model):
        path = path.resolve(strict=True)
        output = destination/f'{index+1:02d}-{path.stem}'
        output.mkdir()
        report = inspect(path)
        if page_input is not None:
            report['profile'] = profile(path, output, page_input)
        assert digest(path) == report['sha256']
        reports.append(report)
        (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(path.name, 'MiB', round(report['bytes']/2**20, 2), 'profile', report.get('profile', {}).get('seconds'), flush=True)
    (destination/'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
