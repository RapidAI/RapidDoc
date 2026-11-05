"""Sequential DOC parity/time/memory acceptance against the installed Rust wheel.

Each phase runs in its own process. Memory is sampled by the parent on Windows,
so Python's GIL and verification/image decoding do not hide or inflate samples.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def installed_binary():
    from importlib.metadata import distribution
    package = distribution('rapid-doc-rust')
    files = []
    for entry in package.files or []:
        if entry.name.startswith('_native') and entry.suffix.lower() == '.pyd':
            path = Path(package.locate_file(entry))
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return dict(distribution_version=package.version, native_files=files)


def worker(args):
    from accept_pdf_delivery import memory
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(engine=args.engine, phase=args.phase, success=False)
    try:
        data = args.source.read_bytes()
        started = time.perf_counter()
        from loguru import logger
        logger.remove()
        sessions = []
        if args.source.suffix.lower() == '.pdf' and args.engine == 'python':
            import onnxruntime as ort
            original = ort.InferenceSession.__init__
            def record(self, path, *pos, **kw):
                sessions.append(dict(model=str(path) if isinstance(path,(str,Path)) else '<in-memory model>',
                    model_bytes=len(path) if isinstance(path,bytes) else None))
                return original(self, path, *pos, **kw)
            ort.InferenceSession.__init__ = record
        if args.engine == 'python':
            from rapid_doc import RapidDoc
        else:
            from rapid_doc_rust import RapidDoc
            if args.source.suffix.lower() == '.pdf':
                from rapid_doc_rust._runtime import LazySession
                original = LazySession.descriptor
                seen = set()
                def record(self):
                    value = original(self)
                    if value not in seen:
                        seen.add(value);sessions.append(json.loads(value))
                    return value
                LazySession.descriptor = record
        engine = RapidDoc()
        report['import_and_init_seconds'] = time.perf_counter() - started
        gc.collect()
        report['baseline'] = memory()
        if args.phase == 'memory':
            print(json.dumps({'ready': True}), flush=True)
            if sys.stdin.readline().strip() != 'GO':
                raise RuntimeError('Missing measurement handshake')
        times = []
        for _ in range(args.repeats + 1 if args.phase == 'time' else 1):
            result = None
            gc.collect()
            started = time.perf_counter()
            result = engine(data)
            times.append(time.perf_counter() - started)
        report.update(parse_seconds=times, after_parse=memory(),
                      pages=len(result.middle_json['pdf_info']), assets=len(result.images), sessions=sessions)
        if args.phase == 'verify':
            from accept_docx_delivery import snapshot
            canonical = snapshot(result)
            dump(args.output / 'snapshot.json', canonical)
            report['canonical_sha256'] = hashlib.sha256(json.dumps(
                canonical, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
        report['success'] = True
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    dump(args.output / 'report.json', report)
    if not report['success']:
        raise SystemExit(1)


def process_memory(process):
    import ctypes
    from ctypes import wintypes
    from accept_pdf_delivery import Counters, PSAPI
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not PSAPI(wintypes.HANDLE(int(process._handle)), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(working_set=counters.working_set, private_commit=counters.private_commit)


def run(args, source, case, engine, phase, round_id):
    destination = args.output / f'{case}-{round_id}-{phase}-{engine}'
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--source', str(source),
               '--engine', engine, '--phase', phase, '--output', str(destination), '--repeats', str(args.repeats)]
    samples = []
    gaps = []
    with (args.output / (destination.name + '.log')).open('w', encoding='utf-8') as log:
        if phase != 'memory':
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        else:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, encoding='utf-8')
            try:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        raise RuntimeError(f'Memory worker did not become ready: {destination}')
                    log.write(line)
                    if json.loads(line).get('ready'):
                        break
                samples.append(process_memory(process))
                process.stdin.write('GO\n')
                process.stdin.flush()
                start = last = time.perf_counter()
                while process.poll() is None and not (destination / 'report.json').exists():
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now
                    samples.append(process_memory(process))
                    if now - start > 300:
                        raise TimeoutError('Memory worker exceeded 300 seconds')
                    time.sleep(.002)
                process.wait(timeout=30)
                if process.returncode:
                    raise RuntimeError(f'Memory worker failed: {destination}')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()
    report = json.loads((destination / 'report.json').read_text(encoding='utf-8'))
    if phase == 'memory':
        before = report['baseline']
        after = report['after_parse']
        report['external_sampling'] = dict(
            samples=len(samples), target_interval_seconds=.002,
            maximum_gap_seconds=max(gaps, default=0),
            peak_working_set=max([after['working_set'], *[s['working_set'] for s in samples]]),
            peak_private_commit=max([after['private_commit'], *[s['private_commit'] for s in samples]]),
            baseline_working_set=before['working_set'], baseline_private_commit=before['private_commit'],
            os_lifetime_peak_working_set=after['os_peak_working_set'],
            os_lifetime_peak_commit=after['os_peak_commit'])
        dump(destination / 'report.json', report)
    report.update(case=case, round=round_id)
    print(f'{case}: round {round_id + 1} {phase} {engine} complete', flush=True)
    return report


def stress(paragraphs):
    from export_doc_compat_reference import doc
    sys.path.insert(0, str(ROOT / 'tests/native_office'))
    from _mtef_test_utils import _build_nested_cfb
    text = 'DOC stress: 中文与 English, emoji 😀, numbers 12345.\r' * paragraphs
    word, table = doc(text.encode('utf-16le'), compressed=False, clx=True)
    return _build_nested_cfb([
        dict(name='Root Entry', kind=5, child=1),
        dict(name='1Table', kind=2, right=2, data=table),
        dict(name='WordDocument', kind=2, data=word)])


def summary(reports):
    rows = []
    for case in sorted({r['case'] for r in reports}):
        chosen = [r for r in reports if r['case'] == case]
        verified = [r for r in chosen if r['phase'] == 'verify']
        row = dict(case=case, parity=len(verified) == 2 and len({r['canonical_sha256'] for r in verified}) == 1)
        for engine in ['python', 'rust']:
            timings = [r for r in chosen if r['engine'] == engine and r['phase'] == 'time']
            memory_reports = [r for r in chosen if r['engine'] == engine and r['phase'] == 'memory']
            memory = [r['external_sampling'] for r in memory_reports]
            row[engine] = dict(
                cold_seconds=statistics.median(r['parse_seconds'][0] for r in timings),
                warm_median_seconds=statistics.median(t for r in timings for t in r['parse_seconds'][1:]),
                import_init_seconds=statistics.median(r['import_and_init_seconds'] for r in timings),
                sampled_parse_peak_rss=max(r['peak_working_set'] for r in memory),
                sampled_parse_peak_private_commit=max(r['peak_private_commit'] for r in memory),
                baseline_rss=statistics.median(r['baseline_working_set'] for r in memory),
                baseline_private_commit=statistics.median(r['baseline_private_commit'] for r in memory),
                os_lifetime_peak_rss=max(r['os_lifetime_peak_working_set'] for r in memory),
                os_peak_before_parse_rss=max(r['baseline']['os_peak_working_set'] for r in memory_reports),
                sampled_rss_growth=max(r['peak_working_set'] - r['baseline_working_set'] for r in memory),
                sampled_private_growth=max(r['peak_private_commit'] - r['baseline_private_commit'] for r in memory))
        row['warm_speedup'] = row['python']['warm_median_seconds'] / row['rust']['warm_median_seconds']
        # The sampler can miss sub-millisecond Rust peaks. Use Windows' retained
        # high-water mark for the headline memory comparison, including imports.
        row['process_peak_rss_rust_over_python'] = row['rust']['os_lifetime_peak_rss'] / row['python']['os_lifetime_peak_rss']
        rows.append(row)
    return dict(cases=rows, all_parity=all(r['parity'] for r in rows), notes=[
        'Sequential independent workers; execution order alternates by round and case.',
        'Timing, cold-parse memory and verification use separate processes; no models or output files enabled.',
        'Imports/init and input reads are excluded from parse timing; Windows lifetime RSS includes imports.',
        'External working-set/private-commit sampling can miss spikes; gaps and Windows peaks are retained.',
        'Headline memory ratio uses the Windows lifetime high-water mark, not the lower sampled peak.',
        'Warm latency aggregates repeated calls; cold latency is the first call after import/engine construction.',
        'Input snapshots are identical; output verification includes full middle, Markdown, content list and exact asset pixels/SVG.',
        'Text stress is generated, repetitive and not additional real-document coverage.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT / 'demo/ms_office_docs/docx_01.doc')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--stress-paragraphs', type=int, default=6000)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--summarize', action='store_true', help='Rebuild summary from saved reports without rerunning parses')
    parser.add_argument('--engine', choices=['python', 'rust'])
    parser.add_argument('--phase', choices=['time', 'memory', 'verify'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        return worker(args)
    if args.summarize:
        result = summary(json.loads((args.output / 'reports.json').read_text(encoding='utf-8')))
        dump(args.output / 'summary.json', result)
        fingerprint = args.output / 'installed-binary-after-run.json'
        if not fingerprint.exists():
            dump(fingerprint, installed_binary())
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if not result['all_parity']:
            raise SystemExit('Output parity failed')
        return
    if os.name != 'nt':
        raise RuntimeError('This acceptance uses Windows process memory counters')
    if args.rounds < 1 or args.repeats < 1 or args.stress_paragraphs < 0:
        raise ValueError('rounds/repeats must be positive; stress-paragraphs must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=False)
    real = args.output / 'source.doc'
    real.write_bytes(args.source.read_bytes())
    sources = [('real', real)]
    if args.stress_paragraphs:
        generated = args.output / 'text-stress.doc'
        generated.write_bytes(stress(args.stress_paragraphs))
        sources.append(('text-stress', generated))
    dump(args.output / 'manifest.json', dict(
        python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
        rounds=args.rounds, warm_repeats=args.repeats, stress_paragraphs=args.stress_paragraphs,
        installed_rust=installed_binary(),
        sources=[dict(case=case, path=str(path), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for case, path in sources]))
    reports = []
    for index, (case, source) in enumerate(sources):
        for round_id in range(args.rounds):
            order = ['python', 'rust'] if (round_id + index) % 2 == 0 else ['rust', 'python']
            for phase in ['time', 'memory']:
                for engine in order:
                    reports.append(run(args, source, case, engine, phase, round_id))
                    dump(args.output / 'reports.json', reports)
        for engine in ['python', 'rust']:
            reports.append(run(args, source, case, engine, 'verify', 0))
            dump(args.output / 'reports.json', reports)
    result = summary(reports)
    dump(args.output / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['all_parity']:
        raise SystemExit('Output parity failed; performance numbers cannot be accepted')


if __name__ == '__main__':
    main()
