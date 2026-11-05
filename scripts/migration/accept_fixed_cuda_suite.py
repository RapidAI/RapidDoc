"""Frozen full-document CUDA comparison; run workers sequentially, never rebuild.

Run from a validation checkout with the final Rust SDK on PYTHONPATH. The
reference report supplies the accepted original model paths, not cached graphs.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def differences(left, right, path='$'):
    """Portable exact structural diagnostics; no Windows memory module import."""
    if type(left) is not type(right):
        yield dict(path=path, kind='type', expected=type(left).__name__, actual=type(right).__name__)
    elif isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                yield dict(path=f'{path}.{key}', kind='missing', expected=key in left, actual=key in right)
            else:
                yield from differences(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, list):
        if len(left) != len(right):
            yield dict(path=path, kind='length', expected=len(left), actual=len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            yield from differences(a, b, f'{path}[{index}]')
    elif left != right:
        yield dict(path=path, kind='value', expected=repr(left)[:180], actual=repr(right)[:180])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--reference-report', type=Path, required=True)
    parser.add_argument('--long-source', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=3)
    parser.add_argument('--resume', action='store_true', help='Reuse complete observations after verifying frozen production files')
    args = parser.parse_args()
    if args.calls < 2:
        parser.error('Use at least one cold and one reused call')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cases = [('scan', 'small_ocr'), ('paper', '示例1-论文模板'),
             ('dense', '49－长表格-密集型'), ('mixed', 'demo1'),
             ('finance', '比亚迪财报'), ('report', '示例7-研究报告'), ('seal', '含印章')]
    sources = [(label, ROOT / 'demo/pdfs' / (name + '.pdf')) for label, name in cases]
    sources.append(('long129', args.long_source.resolve()))
    # Page counts are obtained in this coordinator only, never the measured worker.
    import pypdfium2
    frozen = []
    for label, source in sources:
        destination = args.output / (label + '.pdf')
        if args.resume:
            assert digest(destination) == digest(source), 'Frozen input changed'
        else:
            destination.write_bytes(source.read_bytes())
        with pypdfium2.PdfDocument(destination) as pdf:
            pages = len(pdf)
        frozen.append(dict(case=label, source=str(source), frozen=str(destination),
                           sha256=digest(source), pages=pages))
    reference = json.loads(args.reference_report.read_text())
    originals = sorted({Path(item['model']).resolve() for item in reference['sessions']})
    # Python's default RapidOCR also initializes its unused angle classifier.
    # Freeze that auxiliary source before running, although text_cls is disabled
    # and the native pipeline does not create its unused session.
    auxiliary = Path(os.environ['RAPID_MODELS_DIR']) / 'ch_ppocr_mobile_v2.0_cls_mobile.onnx'
    originals = sorted(set(originals) | {auxiliary.resolve()})
    code_files = sorted(set((ROOT / 'rapid_doc').rglob('*.py')) |
                        set((ROOT / 'rapid_doc').rglob('*.yaml')) |
                        set((ROOT / 'rapid_doc').rglob('*.json')) |
                        {p for p in (args.sdk / 'rapid_doc_rust').rglob('*')
                         if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'} |
                        {Path(__file__).resolve(), Path(__file__).with_name('accept_cuda_pdf.py'),
                         Path(__file__).with_name('accept_docx_delivery.py')})
    original_ops = ROOT / 'rapid_doc/backend/onnx_ops'
    code_files += sorted(original_ops.glob('*.so'))
    locked_files = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p))
                    for p in originals + code_files]
    # The immutable SDK catalog also pins lazy auxiliary models not exercised by
    # the reference run (for example the table orientation classifier). Its native binary
    # was already among locked_files; never learn acceptable hashes from outputs.
    sys.path.insert(0, str(args.sdk))
    from rapid_doc_rust._runtime import MANIFEST as model_catalog
    manifest = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    python=sys.version, calls=args.calls, device='cuda', cases=frozen,
                    locked_files=locked_files,
                    model_catalog={role: item['sha256'] for role, item in model_catalog.items()},
                    packages={name: importlib.metadata.version(name) for name in
                        ['onnxruntime-gpu', 'onnx', 'numpy', 'rapidocr', 'pypdfium2', 'pdftext']},
                    environment={key: os.environ.get(key) for key in
                        ['RAPID_MODELS_DIR', 'PYTHONPATH', 'MINERU_DEVICE_MODE',
                         'RAPIDDOC_GEOS_LIBRARY', 'RAPIDDOC_PDFIUM_LIBRARY', 'XDG_CACHE_HOME']},
                    policy='full documents; auto; original production defaults; sequential isolated processes; cold plus two warm calls; 50ms process RSS/GPU sampling')
    if args.resume:
        before = json.loads((args.output / 'manifest.json').read_text())
        for key in ['calls', 'cases', 'packages', 'environment']:
            assert before[key] == manifest[key], f'Resume configuration changed: {key}'
        revised = [item for item in before['locked_files'] if digest(Path(item['path'])) != item['sha256']]
        assert all(Path(item['path']) == Path(__file__).resolve() for item in revised), 'Frozen production/worker/model files changed'
        manifest['validation_only_revisions'] = revised
        dump(args.output / 'manifest-resumed.json', manifest)
    else:
        dump(args.output / 'manifest.json', manifest)
    comparisons = []
    observed_models = set()
    import psutil
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    for index, case in enumerate(frozen):
        reports = {}
        for engine in (['python', 'rust'] if index % 2 == 0 else ['rust', 'python']):
            destination = args.output / (case['case'] + '-' + engine)
            marker = args.output / (destination.name + '-measurement-finished.txt')
            report_path = destination / 'report.json'
            external_path = args.output / (destination.name + '-external-report.json')
            if args.resume and report_path.exists() and external_path.exists():
                report = json.loads(report_path.read_text())
                assert report['input_sha256'] == case['sha256']
                assert len(report['rows']) == args.calls
                report['external_sampling'] = json.loads(external_path.read_text())
                reports[engine] = report
                print(case['case'], engine, 'REUSED_COMPLETE_OBSERVATION', flush=True)
                continue
            print(case['case'], engine, 'START', flush=True)
            command = [sys.executable, str(Path(__file__).with_name('accept_cuda_pdf.py')),
                       '--engine', engine, '--source', case['frozen'], '--end', str(case['pages'] - 1),
                       '--calls', str(args.calls), '--gpu-peak', '--settle-seconds', '1',
                       '--measurement-marker', str(marker),
                       '--output', str(destination)]
            samples = []
            sampling_errors = []
            with (args.output / (destination.name + '.log')).open('w', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                process = psutil.Process(child.pid)
                started = time.monotonic()
                while child.poll() is None:
                    if time.monotonic() - started > 3600:
                        child.kill()
                        break
                    if not marker.exists():
                        try:
                            rss = process.memory_info().rss
                            gpu = sum(p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
                                      if p.pid == child.pid)
                            if not marker.exists():
                                samples.append([time.monotonic()-started, rss, gpu])
                        except psutil.NoSuchProcess:
                            pass
                        except pynvml.NVMLError as error:
                            sampling_errors.append(str(error))
                    time.sleep(.05)
                exit_code = child.wait()
            reports[engine] = json.loads(report_path.read_text()) if report_path.exists() else dict(success=False, error=str(exit_code))
            external = dict(interval_seconds=.05, samples=len(samples),
                            peak_rss_bytes=max((item[1] for item in samples), default=0),
                            peak_gpu_bytes=max((item[2] for item in samples), default=0),
                            measurement_marker_seen=marker.exists(), errors=sampling_errors,
                            scope='child process imports, model initialization and parse calls; stops before output verification; independent of worker GIL')
            reports[engine]['external_sampling'] = external
            dump(args.output / (destination.name + '-external-samples.json'), samples)
            dump(args.output / (destination.name + '-external-report.json'), external)
            print(case['case'], engine, 'DONE', reports[engine]['success'], flush=True)
        py, ru = reports['python'], reports['rust']
        for report in reports.values():
            observed_models.update((item['path'], item['sha256']) for item in report.get('original_model_files', []))
        good = py['success'] and ru['success']
        row = dict(case=case['case'], pages=case['pages'], success=good)
        if good:
            row['same_configs'] = py['requested_configs'] == ru['requested_configs'] and py['public_defaults'] == ru['public_defaults']
            row['same_input'] = py['input_sha256'] == ru['input_sha256'] == case['sha256']
            row['snapshot_exact'] = py['snapshot_sha256'] == ru['snapshot_sha256']
            row['markdown_bytes_exact'] = (args.output / (case['case']+'-python/document.md')).read_bytes() == (args.output / (case['case']+'-rust/document.md')).read_bytes()
            row['complete_pages'] = all(item['pages'] == case['pages'] for report in [py, ru] for item in report['rows'])
            locked_models = {item['sha256'] for item in locked_files if item['path'].endswith('.onnx')}
            locked_models.update(manifest['model_catalog'].values())
            row['models_from_frozen_set'] = all(report['original_model_files'] and
                all(item['sha256'] in locked_models for item in report['original_model_files'])
                for report in [py, ru])
            row['python_model_files_observed'] = len(py['original_model_files'])
            for engine, report in reports.items():
                row[engine] = dict(cold_seconds=report['rows'][0]['seconds'],
                    warm_seconds=[item['seconds'] for item in report['rows'][1:]],
                    warm_median_seconds=statistics.median(item['seconds'] for item in report['rows'][1:]),
                    peak_rss_bytes=report['external_sampling']['peak_rss_bytes'], peak_gpu_bytes=report['external_sampling']['peak_gpu_bytes'],
                    checkpoints=report['memory_checkpoints'])
            row['warm_speedup_python_over_rust'] = row['python']['warm_median_seconds'] / row['rust']['warm_median_seconds']
            row['external_sampling_valid'] = all(report['external_sampling']['samples'] > 0 and
                report['external_sampling']['measurement_marker_seen'] and not report['external_sampling']['errors']
                for report in [py, ru])
            if not row['snapshot_exact']:
                left, right = [json.loads((args.output / (case['case']+'-'+engine+'/snapshot.json')).read_text()) for engine in ['python', 'rust']]
                row['first_differences'] = list(itertools.islice(differences(left, right), 12))
            row['accepted'] = all(row[key] for key in ['same_configs', 'same_input', 'snapshot_exact',
                'markdown_bytes_exact', 'complete_pages', 'models_from_frozen_set', 'external_sampling_valid'])
        else:
            row.update(accepted=False, errors={engine: report.get('error') for engine, report in reports.items() if not report['success']})
        comparisons.append(row)
        dump(args.output / 'summary.json', dict(cases=comparisons, complete=False))
        print(case['case'], 'ACCEPTED', row['accepted'], flush=True)
    changed = [item['path'] for item in locked_files if digest(Path(item['path'])) != item['sha256']]
    changed += [path for path, expected in observed_models if digest(Path(path)) != expected]
    pynvml.nvmlShutdown()
    for case in frozen:
        if digest(Path(case['frozen'])) != case['sha256'] or digest(Path(case['source'])) != case['sha256']:
            changed.append(case['source'])
    summary = dict(cases=comparisons, complete=True, changed_locked_files=changed,
                   all_accepted=not changed and all(row['accepted'] for row in comparisons))
    dump(args.output / 'summary.json', summary)
    print('FIXED_SUITE_COMPLETE', summary['all_accepted'], flush=True)
    if not summary['all_accepted']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
