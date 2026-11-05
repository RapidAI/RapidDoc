"""Five mixed-document cycles per engine; sequential isolated CUDA workers."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from accept_fixed_cuda_suite import dump, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=5)
    args = parser.parse_args()
    if args.cycles < 3:
        parser.error('At least three cycles are needed to observe reuse')
    # The parent shell starts this only after the frozen full suite completes.
    suite = json.loads((args.suite / 'summary.json').read_text())
    assert suite['complete'], 'Complete the full suite first'
    manifest = json.loads((args.suite / 'manifest.json').read_text())
    assert not suite['changed_locked_files'], 'Frozen production files changed'
    assert all(row['accepted'] for row in suite['cases']), 'Output parity must pass first'
    cases = [item for item in manifest['cases'] if item['case'] != 'long129']
    args.output.mkdir(parents=True, exist_ok=False)
    sequence = args.output / 'sequence.json'
    dump(sequence, dict(cases=cases, cycles=args.cycles,
        rationale='all seven original PDFs in fixed order; one engine reused throughout; previous result released before each call'))
    import psutil
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    reports = {}
    for engine in ['python', 'rust']:
        output = args.output / engine
        marker = args.output / (engine + '-measurement-finished.txt')
        command = [sys.executable, str(Path(__file__).with_name('accept_cuda_pdf.py')),
            '--engine', engine, '--source', cases[0]['frozen'], '--output', str(output),
            '--sequence-manifest', str(sequence), '--calls', str(args.cycles),
            '--gpu-peak', '--settle-seconds', '1', '--measurement-marker', str(marker)]
        samples = []
        errors = []
        print(engine, 'CONTINUOUS_START', flush=True)
        with (args.output / (engine + '.log')).open('w', encoding='utf8') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            child = psutil.Process(process.pid)
            started = time.monotonic()
            while process.poll() is None:
                if time.monotonic()-started > 1800:
                    process.kill()
                    errors.append('timeout')
                    break
                if not marker.exists():
                    try:
                        rss = child.memory_info().rss
                        used = sum(p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu)
                                   if p.pid == child.pid)
                        if not marker.exists():
                            samples.append([time.monotonic()-started, rss, used])
                    except psutil.NoSuchProcess:
                        pass
                    except pynvml.NVMLError as error:
                        errors.append(str(error))
                time.sleep(.05)
            code = process.wait()
        report = json.loads((output / 'report.json').read_text())
        assert code == 0 and report['success'], report.get('error')
        assert len(report['rows']) == args.cycles * len(cases)
        report['external_sampling'] = dict(samples=len(samples), errors=errors,
            peak_rss_bytes=max((item[1] for item in samples), default=0),
            peak_gpu_bytes=max((item[2] for item in samples), default=0),
            measurement_marker_seen=marker.exists())
        assert not errors and samples and marker.exists()
        dump(args.output / (engine + '-samples.json'), samples)
        reference_hashes = {item['case']: hashlib.sha256(
            (args.suite / (item['case']+'-'+engine) / 'document.md').read_bytes()).hexdigest()
            for item in cases}
        report['all_markdown_exact_to_full_suite'] = all(
            row['markdown_sha256'] == reference_hashes[row['case']] for row in report['rows'])
        checkpoints = {item['label']: item for item in report['memory_checkpoints']}
        report['cycle_boundaries'] = []
        for cycle in range(args.cycles):
            last = (cycle+1)*len(cases)-1
            label = f'before_call_{last+1}_previous_result_released' if cycle+1 < args.cycles else 'sequence_finished_final_result_released'
            report['cycle_boundaries'].append(dict(cycle=cycle+1, **checkpoints[label]))
        dump(args.output / (engine + '-analysis.json'), report)
        reports[engine] = report
        print(engine, 'CONTINUOUS_DONE', 'markdown_exact', report['all_markdown_exact_to_full_suite'], flush=True)
    pynvml.nvmlShutdown()
    unchanged = all(digest(Path(item['path'])) == item['sha256'] for item in manifest['locked_files'])
    unchanged = unchanged and all(digest(Path(item['frozen'])) == item['sha256'] for item in cases)
    result = dict(complete=True, production_files_unchanged=unchanged,
        cycles=args.cycles, documents_per_engine=args.cycles*len(cases),
        pages_per_engine=args.cycles*sum(item['pages'] for item in cases),
        engines={engine: dict(peak=report['external_sampling'], boundaries=report['cycle_boundaries'],
            markdown_exact=report['all_markdown_exact_to_full_suite']) for engine, report in reports.items()},
        note='Cycle boundary RSS/GPU compares identical completed workloads. Five cycles cannot prove absence of a long-term leak.')
    dump(args.output / 'summary.json', result)
    assert unchanged and all(report['all_markdown_exact_to_full_suite'] for report in reports.values())
    print('CONTINUOUS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
