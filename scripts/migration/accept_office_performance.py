"""Default six-format Office performance using isolated sequential workers.

Shares DOC's external Windows sampling and exact output snapshots. No models,
parser settings, source documents or runtime libraries are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

from accept_doc_performance import ROOT, dump, installed_binary, run, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pdf-only', action='store_true', help='Two full representative PDFs with default models')
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert args.rounds > 0 and args.repeats > 0 and os.name == 'nt'
    originals = sorted(p for p in (ROOT / 'demo/ms_office_docs').iterdir()
        if p.suffix in {'.doc','.docx','.xls','.xlsx','.ppt','.pptx'} and not p.name.startswith('~$'))
    assert {p.suffix for p in originals} == {'.doc','.docx','.xls','.xlsx','.ppt','.pptx'}
    if args.pdf_only:
        originals = [ROOT/'demo/pdfs/49－长表格-密集型.pdf', ROOT/'demo/pdfs/small_ocr.pdf']
    def case_id(p): return p.stem if args.pdf_only else p.suffix[1:]
    def frozen(p):
        return args.output / (('source-' + p.stem if args.pdf_only else 'source') + p.suffix)
    manifest = dict(installed_rust=installed_binary(), rounds=args.rounds, repeats=args.repeats,
                    python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
                    sources=[dict(case=case_id(p), original=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in originals])
    if args.resume:
        assert json.loads((args.output / 'manifest.json').read_text(encoding='utf8')) == manifest, 'Inputs/config/binary changed'
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / 'manifest.json', manifest)
        for p in originals:
            frozen(p).write_bytes(p.read_bytes())
    reports = []

    def measure(source, case, engine, phase, round_id):
        destination = args.output / f'{case}-{round_id}-{phase}-{engine}'
        report = destination / 'report.json'
        if args.resume and report.exists():
            value = json.loads(report.read_text(encoding='utf8'))
            assert value['success'] and (phase != 'memory' or 'external_sampling' in value), destination
            value.update(case=case, round=round_id)
            return value
        return run(args, source, case, engine, phase, round_id)

    for index, p in enumerate(originals):
        source = frozen(p)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['sources'][index]['sha256']
        case = case_id(p)
        for round_id in range(args.rounds):
            order = ['python','rust'] if (round_id + index) % 2 == 0 else ['rust','python']
            for phase in ['time','memory']:
                for engine in order:
                    reports.append(measure(source, case, engine, phase, round_id))
                    dump(args.output / 'reports.json', reports)
        for engine in ['python','rust']:
            reports.append(measure(source, case, engine, 'verify', 0))
            dump(args.output / 'reports.json', reports)
        dump(args.output / 'summary.json', summary(reports))
    assert manifest['installed_rust'] == installed_binary(), 'Binary changed during measurement'
    result = summary(reports)
    if args.pdf_only:
        result['notes'] = [n for n in result['notes'] if 'no models' not in n.lower()]
        result['notes'].append('Two full PDFs using default auto/model settings; cold calls include lazy model loading; session paths are recorded.')
    else:
        result['notes'].append('Six full original Office documents; performance is specific to this local corpus and machine.')
    dump(args.output / 'summary.json', result)
    if not result['all_parity']:
        raise SystemExit('Output parity failed; timing is not an acceptance result')
    for row in result['cases']:
        print(f"{row['case']}: {row['warm_speedup']:.2f}x, parity={row['parity']}", flush=True)


if __name__ == '__main__':
    main()
