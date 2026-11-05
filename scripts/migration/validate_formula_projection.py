"""Serial ABBA comparison of separate original/candidate sessions on frozen tensors."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in
        ('peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool',
         'peak_nonpaged_pool', 'nonpaged_pool', 'pagefile', 'peak_pagefile', 'private')]


def memory():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {name + '_bytes': getattr(counters, name) for name in
            ('working_set', 'peak_working_set', 'private', 'peak_pagefile')}


def worker(model, output, record_mismatches=False, denormal_as_zero=False, threads=0):
    import numpy as np
    import onnxruntime as ort
    from runtime_telemetry import Telemetry

    assert not output.exists()
    source = ROOT / 'build/formula-tensor-audit-v3-20261029/native-model'
    jobs = []
    for path in sorted(source.glob('*.tokens.json')):
        expected = json.loads(path.read_text())
        raw_path = path.with_name(path.name.replace('.tokens.json', '.f32'))
        metadata = json.loads(path.with_name(path.name.replace('.tokens.json', '.input.json')).read_text())
        raw = raw_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
        jobs.append((path.name, np.frombuffer(raw, dtype='<f4').reshape(metadata['shape']), expected))
    assert len(jobs) == 74
    with model.open('rb') as stream:
        model_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    telemetry = Telemetry(output.with_suffix('.telemetry.jsonl'))
    try:
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 4
        if threads:
            assert threads > 0
            options.intra_op_num_threads = threads
        if denormal_as_zero:
            options.add_session_config_entry('session.set_denormal_as_zero', '1')
        before = memory()
        started = time.perf_counter()
        session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        load_seconds = time.perf_counter() - started
        after_load = memory()
        name = session.get_inputs()[0].name
        # Identical short warmup in each isolated process; excluded from timings.
        warmup = min(jobs, key=lambda job: job[2]['shape'][1])
        result = session.run(None, {name: warmup[1]})[0]
        warmup_matches = list(result.shape) == warmup[2]['shape'] and result.reshape(-1).tolist() == warmup[2]['values']
        if not record_mismatches:
            assert warmup_matches
        samples = []
        for index, (filename, tensor, expected) in enumerate(jobs):
            telemetry.document = filename
            cpu_started = time.process_time()
            started = time.perf_counter()
            result = session.run(None, {name: tensor})[0]
            seconds = time.perf_counter() - started
            cpu_seconds = time.process_time() - cpu_started
            matches = list(result.shape) == expected['shape'] and result.reshape(-1).tolist() == expected['values']
            samples.append(dict(source=filename, tokens=expected['shape'][1], seconds=seconds,
                                cpu_seconds=cpu_seconds, raw_tokens_match=matches, memory=memory()))
            if not matches:
                samples[-1].update(actual_shape=list(result.shape), actual_tokens=result.reshape(-1).tolist())
                if not record_mismatches:
                    raise AssertionError(f'Raw token mismatch: {filename}')
            if (index + 1) % 10 == 0 or index + 1 == len(jobs):
                print(output.stem, index + 1, '/', len(jobs),
                      round(sum(row['seconds'] for row in samples), 3), flush=True)
        report = dict(model=str(model), model_sha256=model_sha256, ort_version=ort.__version__,
                      profiling=False, threads=threads, cpu_arena=True, warmup_source=warmup[0],
                      denormal_as_zero=denormal_as_zero,
                      load_seconds=load_seconds, before=before, after_load=after_load,
                      final_memory=memory(), samples=samples,
                      total_seconds=sum(row['seconds'] for row in samples),
                      cpu_seconds=sum(row['cpu_seconds'] for row in samples),
                      warmup_raw_tokens_match=warmup_matches,
                      all_raw_tokens_match=all(row['raw_tokens_match'] for row in samples))
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    finally:
        telemetry.close()


def compare(directory, first_source_report=None):
    manifest = json.loads((directory / 'manifest.json').read_text())
    runs = []
    for index, kind in enumerate(('source', 'candidate', 'candidate', 'source')):
        output = directory / f'{index + 1:02d}-{kind}.json'
        assert not output.exists(), f'Refusing to overwrite {output}'
        if index == 0 and first_source_report is not None:
            previous = json.loads(first_source_report.read_text())
            assert Path(previous['model']).resolve() == Path(manifest['source']).resolve()
            assert previous['all_raw_tokens_match'] and len(previous['samples']) == 74
            assert previous['profiling'] is False and previous['threads'] == 0 and previous['cpu_arena']
            assert not previous.get('denormal_as_zero', False)
            output = first_source_report
        else:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                            '--model', manifest[kind], '--output', str(output)], check=True, cwd=ROOT)
        report = json.loads(output.read_text())
        assert report['model_sha256'] == manifest[kind + '_sha256']
        runs.append(dict(kind=kind, path=str(output), total_seconds=report['total_seconds'],
                         cpu_seconds=report['cpu_seconds'], load_seconds=report['load_seconds'],
                         final_memory=report['final_memory'], all_raw_tokens_match=True))
        (directory / 'runs-progress.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
    means = {kind: statistics.mean(row['total_seconds'] for row in runs if row['kind'] == kind)
             for kind in ('source', 'candidate')}
    result = dict(order='ABBA', cases_per_run=74, runs=runs, mean_seconds=means,
                  candidate_time_reduction_percent=100 * (1 - means['candidate'] / means['source']),
                  all_raw_tokens_match=True, profiling=False, installed_model_modified=False)
    (directory / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def batch_smoke(directory):
    """Check dynamic batch and mixed EOS lengths against the original graph."""
    import gc
    import numpy as np
    import onnxruntime as ort

    manifest = json.loads((directory / 'manifest.json').read_text())
    output = directory / 'batch-smoke.json'
    assert not output.exists()
    source = ROOT / 'build/formula-tensor-audit-v3-20261029/native-model'
    pairs = [(0, 1), (16, 51)]
    tensors = []
    for pair in pairs:
        images = []
        for index in pair:
            metadata = json.loads((source / f'{index:06d}.input.json').read_text())
            raw = (source / f'{index:06d}.f32').read_bytes()
            assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
            images.append(np.frombuffer(raw, dtype='<f4').reshape(metadata['shape']))
        tensors.append(np.concatenate(images))
    reference = []
    cases = []
    for kind in ('source', 'candidate'):
        options = ort.SessionOptions()
        options.log_severity_level = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(manifest[kind], options, providers=['CPUExecutionProvider'])
        name = session.get_inputs()[0].name
        for index, tensor in enumerate(tensors):
            result = session.run(None, {name: tensor})[0]
            if kind == 'source':
                reference.append(result.copy())
            else:
                assert np.array_equal(result, reference[index]), f'Batch mismatch: {pairs[index]}'
                cases.append(dict(inputs=pairs[index], shape=list(result.shape), raw_tokens_match=True))
        del session
        gc.collect()
    output.write_text(json.dumps(dict(cases=cases, all_raw_tokens_match=True), indent=2), encoding='utf-8')
    print('Dynamic batch smoke:', cases, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--batch-smoke', action='store_true')
    parser.add_argument('--record-mismatches', action='store_true')
    parser.add_argument('--denormal-as-zero', action='store_true')
    parser.add_argument('--first-source-report', type=Path)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()
    if args.worker:
        worker(args.model.resolve(), args.output.resolve(), args.record_mismatches, args.denormal_as_zero, args.threads)
    elif args.batch_smoke:
        batch_smoke(args.directory.resolve())
    else:
        compare(args.directory.resolve(), args.first_source_report.resolve() if args.first_source_report else None)
