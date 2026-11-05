"""Verify saved PDF ownership, borrowed script context and RGB delivery evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

from accept_doc_performance import installed_binary
from report_value_transfer_delivery import exact

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf8'))


def main():
    prefix = 'build/borrowed-native-r2'
    timing = read(prefix+'-comparison-20261101/report.json')
    assert timing['success'] and timing['markdown_equal']
    installed = installed_binary()
    measurements = {m['version']:m for m in timing['measurements']}
    assert measurements['candidate']['installed_rust'] == installed
    assert measurements['baseline']['installed_rust'] == read('build/value-transfer-delivery-receipt-20261101.json')['installed_rust']
    assert read(prefix+'-public-20261101/manifest.json')['installed_rust'] == installed
    public = read(prefix+'-public-20261101/summary.json')
    assert public['all_passed'] and len(public['checks']) == 6
    windows = read(prefix+'-window-verify-20261101/report.json')
    assert windows['success'] and len(windows['rows']) == 3
    for count in [16,65,129]:
        assert exact(read(f'build/value-transfer-window-verify-20261101/{count}.json'),
                     read(f'{prefix}-window-verify-20261101/{count}.json')), count
    assert exact(read('build/value-transfer-dense-verify-20261101/snapshot.json'),
                 read(prefix+'-dense-verify-20261101/snapshot.json'))
    before_rgb = read('build/borrowed-native-rgb-before-20261101.json')
    assert before_rgb['installed_rust'] == measurements['baseline']['installed_rust']
    after_rgb = read(prefix+'-rgb-20261101.json')
    assert after_rgb['installed_rust'] == installed and after_rgb['matches_reference']
    assert exact(before_rgb['rows'],after_rgb['rows']) and len(after_rgb['rows']) == 30
    for path, token in [
        ('build/borrowed-native-padding-tests-20261101.log','1 passed; 0 failed'),
        ('build/borrowed-native-script-map-tests-20261101.log','1 passed; 0 failed'),
        (prefix+'-focused-tests-20261101.log','4 passed; 0 failed'),
        (prefix+'-owned-tests-20261101.log','1 passed; 0 failed'),
        (prefix+'-resource-tests-20261101.log','5 passed'),
    ]:
        assert token in (ROOT/path).read_text(encoding='utf8'), path
    wheel = ROOT/'build/borrowed-native-r2-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl'
    with zipfile.ZipFile(wheel) as archive:
        binary = next(n for n in archive.namelist() if n.endswith('/_native.pyd'))
        assert hashlib.sha256(archive.read(binary)).hexdigest() == installed['native_files'][0]['sha256']
    rows=[]
    for a,b,p in zip(measurements['baseline']['rows'],measurements['candidate']['rows'],measurements['python']['rows']):
        assert a['call']==b['call']==p['call']
        stage = lambda r:{s['stage']:s['seconds'] for s in r['rust_native_stages']}
        rows.append(dict(call=a['call'],before_seconds=a['seconds'],after_seconds=b['seconds'],python_seconds=p['seconds'],
            time_reduction_percent=100*(1-b['seconds']/a['seconds']),
            native_analysis_before=stage(a)['native_analysis'],native_analysis_after=stage(b)['native_analysis'],
            render_before=stage(a)['render'],render_after=stage(b)['render']))
    files = ['rust/rapiddoc-pdf/src/render.rs','rust/rapiddoc-native/src/script_geometry/refine.rs']
    files += ['rust/rapiddoc/src/processing/'+n for n in [
        'document/native_pdf_document.rs','document/page_containers/mod.rs',
        'document/page_containers/forms.rs','document/page_containers/index.rs',
        'document/inline_output/payloads.rs','graphics/strong_graphics.rs',
        'graphics/raster_charts.rs','text/text_noise.rs','table/table_page.rs','table/table_block/commit.rs']]
    report = dict(installed_rust=installed,wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        timing_source=prefix+'-comparison-20261101/report.json',timings=rows,
        strict_window_snapshots=[16,65,129],dense_snapshot_exact=True,public_model_checks=6,
        rendered_byte_checks=30,focused_tests=4,script_map_boundary_cases=32,rgb_padding_cases=288,
        fixed_python_references=271,installed_resource_tests=5,wheel_matches_installed=True,
        notes=['No background compilation or regression during the final sequential timing comparison.',
               'Two calls per version, ordered baseline then candidate then Python, not alternating multi-round benchmarking.',
               'The first candidate timing file borrowed-native-after included overlapping compilation and is excluded.',
               'All timings here are native-rule PDF; no equivalent scan speedup or new memory claim.',
               'No changes to model precision, recognition algorithms, candidate order or numeric precision.',
               'No whole-workspace, full PDF corpus, Office or CUDA regression.'])
    output=ROOT/'build/borrowed-native-delivery-receipt-20261101.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(timings=rows,strict_snapshots=4,public_checks=6,render_checks=30,wheel_matches_installed=True),indent=2))


if __name__=='__main__':
    main()
