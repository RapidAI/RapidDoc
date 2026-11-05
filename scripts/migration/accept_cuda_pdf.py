"""Isolated CPU/CUDA worker; compare saved snapshots in separate processes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def verify_requested_provider(requested, actual, cuda_required):
    names = [p[0] if isinstance(p, (tuple, list)) else p for p in (requested or [])]
    if ('CUDAExecutionProvider' in names or (cuda_required and not names)) and 'CUDAExecutionProvider' not in actual:
        raise RuntimeError('Requested CUDA session fell back to CPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['python', 'rust'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--calls', type=int, default=2)
    parser.add_argument('--parse-method', choices=['auto', 'txt', 'ocr'], default='auto',
                        help='Keep the whole-document route when isolating selected pages')
    parser.add_argument('--threads', type=int, help='Diagnostic CPU thread count for both engines, including CUDA host work')
    parser.add_argument('--plain-onnx', action='store_true', help='Control: disable original Python CUDA graph/operator optimizations')
    parser.add_argument('--gpu-peak', action='store_true', help='Sample this process GPU memory with optional pynvml, including engine setup')
    parser.add_argument('--settle-seconds', type=float, default=0,
                        help='Idle observation after result release and after each call; excluded from parse timing')
    parser.add_argument('--measurement-marker', type=Path,
                        help='Signal an external sampler to stop before snapshot verification')
    parser.add_argument('--sequence-manifest', type=Path,
                        help='Repeat all frozen cases in this manifest with one engine; calls counts cycles')
    args = parser.parse_args()
    if args.calls < 1 or args.settle_seconds < 0:
        parser.error('calls must be positive and settle-seconds nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['MINERU_DEVICE_MODE'] = args.device
    report = dict(engine=args.engine, device=args.device, source=str(args.source),
                  input_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  start=args.start, end=args.end, success=False, rows=[], sessions=[])
    sequence = None
    if args.sequence_manifest:
        sequence = json.loads(args.sequence_manifest.read_text(encoding='utf8'))['cases']
        for item in sequence:
            path = Path(item['frozen'])
            if item['pages'] < 1 or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Sequence input changed')
        if not sequence:
            raise ValueError('Sequence manifest has no cases')
        report['sequence'] = sequence
    stop = None
    thread = None
    nvml = None
    samples = []
    gpu_samples = []
    opened_models = set()
    def audit_model_open(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            path = os.fsdecode(arguments[0])
            if path.lower().endswith('.onnx'):
                opened_models.add(str(Path(path).resolve()))
    sys.addaudithook(audit_model_open)
    try:
        manifest = os.environ.get('RAPIDDOC_CUDA_MODEL_MANIFEST')
        if manifest:
            path = Path(manifest).resolve()
            report['cuda_model_manifest'] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                contents=json.loads(path.read_text()))
        import psutil
        process = psutil.Process()
        samples = []
        gpu_samples = []
        gpu_sample_errors = []
        if args.gpu_peak:
            import pynvml as nvml
            nvml.nvmlInit()
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
        stop = threading.Event()
        def sample():
            while not stop.wait(.05):
                samples.append(process.memory_info().rss)
                if nvml is not None:
                    try:
                        gpu_samples.append(sum(p.usedGpuMemory for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                               if p.pid == process.pid))
                    except nvml.NVMLError as error:
                        gpu_sample_errors.append(str(error))
        thread = threading.Thread(target=sample, daemon=True)
        # Include Python imports and model/session initialization in sampled peaks.
        thread.start()
        import onnxruntime as ort
        report['ort_version'] = ort.__version__
        report['available_providers'] = ort.get_available_providers()
        if args.device == 'cuda' and 'CUDAExecutionProvider' not in report['available_providers']:
            raise RuntimeError('CUDA provider unavailable')
        if args.engine == 'python':
            initialize = ort.InferenceSession.__init__
            def checked(self, path, *pos, **kw):
                initialize(self, path, *pos, **kw)
                providers = self.get_providers()
                requested = kw.get('providers', pos[1] if len(pos) > 1 else None)
                report['sessions'].append(dict(model=str(path) if isinstance(path, (str, Path)) else '<bytes>',
                                               providers=providers, requested_providers=requested))
                verify_requested_provider(requested, providers, args.device == 'cuda')
            ort.InferenceSession.__init__ = checked
            from rapid_doc import RapidDoc
        else:
            from rapid_doc_rust import RapidDoc
            from rapid_doc_rust._runtime import LazySession
            from rapid_doc_rust._runtime import reference_crop_sampling
            report['crop_sampling'] = reference_crop_sampling()
            descriptor = LazySession.descriptor
            def recorded(self):
                value = descriptor(self)
                entry = json.loads(value)
                if entry not in report['sessions']:
                    report['sessions'].append(entry)
                return value
            LazySession.descriptor = recorded
        from rapidocr import EngineType
        cuda = args.device == 'cuda'
        engine_config = dict(engine_cfg=dict(use_cuda=cuda))
        ocr_config = {'engine_type': EngineType.ONNXRUNTIME,
                      'Det.engine_type': EngineType.ONNXRUNTIME, 'Rec.engine_type': EngineType.ONNXRUNTIME,
                      'EngineConfig.onnxruntime.use_cuda': cuda}
        if args.threads is not None:
            engine_config['engine_cfg']['intra_op_num_threads'] = args.threads
            engine_config['engine_cfg']['inter_op_num_threads'] = 1
            ocr_config['EngineConfig.onnxruntime.intra_op_num_threads'] = args.threads
            ocr_config['EngineConfig.onnxruntime.inter_op_num_threads'] = 1
        report['threads'] = args.threads
        layout_config = dict(engine_cfg=dict(engine_config['engine_cfg']))
        formula_config = dict(engine_cfg=dict(engine_config['engine_cfg']))
        if args.plain_onnx:
            layout_config['engine_cfg'].update(mask_uint8=False, io_binding=False)
            formula_config['engine_cfg']['onnx_ops'] = False
            ocr_config['onnx_ops'] = False
        report['plain_onnx'] = args.plain_onnx
        report['parse_method'] = args.parse_method
        report['requested_configs'] = dict(layout=layout_config, formula=formula_config,
            table=engine_config, ocr={key: getattr(value, 'value', value) for key, value in ocr_config.items()})
        engine = RapidDoc(parse_method=args.parse_method, layout_config=layout_config, formula_config=formula_config, table_config=engine_config,
                          ocr_config=ocr_config)
        report['public_defaults'] = {key: getattr(engine, key) for key in
            ['parse_method', 'formula_enable', 'table_enable', 'lang', 'image_dir_name',
             'image_output_mode', 'pdf_pages_batch']}
        report['baseline_rss_bytes'] = process.memory_info().rss
        report['memory_checkpoints'] = []
        def checkpoint(label):
            if args.settle_seconds:
                time.sleep(args.settle_seconds)
            value = dict(label=label, rss_bytes=process.memory_info().rss)
            if nvml is not None:
                value['gpu_bytes'] = sum(p.usedGpuMemory for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                         if p.pid == process.pid)
            report['memory_checkpoints'].append(value)
        checkpoint('engine_constructed_before_lazy_load')
        result = None
        try:
            calls = [(cycle, item) for cycle in range(args.calls) for item in sequence] if sequence else [
                (index, None) for index in range(args.calls)]
            for index, (cycle, item) in enumerate(calls):
                result = None
                gc.collect()
                checkpoint(f'before_call_{index}_previous_result_released')
                start = time.perf_counter()
                source = Path(item['frozen']) if item else args.source
                first = 0 if item else args.start
                last = item['pages'] - 1 if item else args.end
                result = engine(source, start_page_id=first, end_page_id=last)
                report['rows'].append(dict(call='cold' if index == 0 else 'warm', seconds=time.perf_counter()-start,
                                           pages=len(result.middle_json['pdf_info']), markdown_characters=len(result.markdown)))
                report['rows'][-1].update(cycle=cycle, case=item['case'] if item else args.source.stem,
                    markdown_sha256=hashlib.sha256(result.markdown.encode('utf8')).hexdigest())
                checkpoint(f'after_call_{index}_result_retained')
            if sequence:
                result = None
                gc.collect()
                checkpoint('sequence_finished_final_result_released')
                # The regular exporter below expects a last result. A sequence
                # run verifies per-call Markdown fingerprints against the frozen
                # full-structure suite, without retaining all document outputs.
                report['sequence_completed'] = True
        finally:
            stop.set()
            thread.join()
        if args.measurement_marker:
            args.measurement_marker.write_text('parse measurement finished', encoding='utf8')
        report['sampled_peak_rss_bytes'] = max(samples + [process.memory_info().rss])
        report['memory_sampling_seconds'] = .05
        report['memory_sampling_scope'] = 'imports, engine initialization and parse calls; excludes snapshot serialization'
        if args.gpu_peak:
            if gpu_sample_errors:
                raise RuntimeError(f'GPU memory sampling failed: {gpu_sample_errors[0]}')
            report['sampled_peak_gpu_bytes'] = max(gpu_samples, default=0)
            report['gpu_memory_samples'] = len(gpu_samples)
        if sys.platform.startswith('linux'):
            maps = Path('/proc/self/maps').read_text()
            report['gpu_runtime_libraries'] = sorted({line.split()[-1] for line in maps.splitlines()
                if any(name in line for name in ('libcublas', 'libcudnn', 'libcudart'))})
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                             text=True, capture_output=True, timeout=10)
        report['own_gpu_mib_after_parse'] = [float(line.split(',')[1]) for line in gpu.stdout.splitlines()
                                             if line.split(',')[0].strip() == str(os.getpid())]
        if cuda and not report['own_gpu_mib_after_parse']:
            raise RuntimeError('No GPU allocation observed for this parser process')
        from accept_docx_delivery import snapshot
        if result is not None:
            canonical = snapshot(result)
            (args.output/'snapshot.json').write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding='utf8')
            (args.output/'document.md').write_text(result.markdown, encoding='utf8')
            report['snapshot_sha256'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        # Hash observed original files after memory/timing measurement. Optimized
        # Python sessions can be constructed from bytes, so session paths alone
        # do not identify every source model.
        for session in report['sessions']:
            name = session.get('model', '')
            if isinstance(name, str) and name.endswith('.onnx'):
                opened_models.add(str(Path(name).resolve()))
        report['original_model_files'] = [dict(path=name, sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest())
                                          for name in sorted(opened_models) if Path(name).is_file()]
        report['success'] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join()
        if nvml is not None:
            nvml.nvmlShutdown()
        if samples:
            report.setdefault('sampled_peak_rss_bytes', max(samples))
        if args.gpu_peak:
            report.setdefault('sampled_peak_gpu_bytes', max(gpu_samples, default=0))
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['sessions', 'traceback']}, ensure_ascii=False))
    if not report['success']:
        print(report['traceback'])
        raise SystemExit(1)


if __name__ == '__main__':
    main()
