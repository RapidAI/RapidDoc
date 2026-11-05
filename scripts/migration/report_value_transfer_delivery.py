"""Verify saved native JSON transfer optimization evidence without reparsing."""
import hashlib
import json
from pathlib import Path
import zipfile

from accept_doc_performance import installed_binary

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def exact(a, b):
    # Preserve integer/float types, strings, array order and all unknown fields.
    return json.dumps(a, sort_keys=True, separators=(',', ':')) == json.dumps(b, sort_keys=True, separators=(',', ':'))


def main():
    before = read('build/value-transfer-before-20261101.json')
    after = read('build/value-transfer-after-20261101.json')
    assert before['options'] == after['options']
    assert len(before['rows']) == len(after['rows']) == 2
    assert [r['markdown_sha256'] for r in before['rows']] == [r['markdown_sha256'] for r in after['rows']]
    timings = []
    for a, b in zip(before['rows'], after['rows']):
        assert a['call'] == b['call'] and a['pages'] == b['pages'] == 129
        stage = lambda r: {s['stage']: s['seconds'] for s in r['rust_native_stages']}
        timings.append(dict(call=a['call'], before_seconds=a['seconds'], after_seconds=b['seconds'],
            time_reduction_percent=100*(1-b['seconds']/a['seconds']),
            native_analysis_before_seconds=stage(a)['native_analysis'],
            native_analysis_after_seconds=stage(b)['native_analysis']))
    for count in [16, 65, 129]:
        assert exact(read(f'build/native-owned-window-verify-20261101/{count}.json'),
                     read(f'build/value-transfer-window-verify-20261101/{count}.json')), count
    assert read('build/value-transfer-window-verify-20261101/report.json')['success']
    assert exact(read('build/pdf-performance-20261101/49－长表格-密集型-0-verify-rust/snapshot.json'),
                 read('build/value-transfer-dense-verify-20261101/snapshot.json'))
    public = read('build/value-transfer-pdf-public-20261101/summary.json')
    assert public['all_passed'] and len(public['checks']) == 6
    installed = installed_binary()
    assert installed == read('build/value-transfer-pdf-public-20261101/manifest.json')['installed_rust']
    wheel = ROOT / 'build/value-transfer-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl'
    with zipfile.ZipFile(wheel) as archive:
        binary = next(n for n in archive.namelist() if n.endswith('/_native.pyd'))
        assert hashlib.sha256(archive.read(binary)).hexdigest() == installed['native_files'][0]['sha256']
    for name, success in [('number-test', '1 passed; 0 failed'),
                          ('owned-reference-tests', '1 passed; 0 failed'),
                          ('resource-tests', '5 passed')]:
        assert success in (ROOT / f'build/value-transfer-{name}-20261101.log').read_text(encoding='utf8')
    files = [
        'mod.rs', 'document/native_pdf_document.rs', 'document/native_document.rs',
        'document/page_containers/mod.rs', 'document/inline_output/payloads.rs',
        'graphics/strong_graphics.rs', 'table/table_page.rs',
        'table/closed_rule_pipeline/mod.rs', 'table/closed_rule_pipeline/supplemental/mod.rs',
        'layout/native_geometry/mod.rs',
    ]
    files = ['rust/rapiddoc/src/processing/' + name for name in files]
    result = dict(installed_rust=installed, wheel=str(wheel),
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        baseline_installed_rust=read('build/native-owned-delivery-receipt-20261101.json')['installed_rust'],
        timings=timings, complete_window_snapshots_exact=[16,65,129], dense_table_snapshot_exact=True,
        public_default_pdf_checks=6, focused_python_reference_cases=271, number_boundary_tests=1,
        installed_resource_tests=5, wheel_matches_installed=True,
        notes=['Two sequential calls per version (cold and warm), not an alternating multi-round speed benchmark.',
               'Both versions use the same lightweight phase instrumentation; no stack sampling or cProfile in these runs.',
               'This targets native-rule analysis. No claim of the same speedup for model-dominated scans.',
               'Arbitrary precision, float roundtrip, candidate ordering and recognition rules retained.',
               'No new memory benchmark, full corpus, Office or CUDA regression.',
               'Native-rule PDF remains slower than the prior Python result; performance parity is not achieved.'])
    target = ROOT / 'build/value-transfer-delivery-receipt-20261101.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(timings=timings, strict_snapshots=4, public_checks=6, wheel_matches_installed=True), indent=2))


if __name__ == '__main__':
    main()
