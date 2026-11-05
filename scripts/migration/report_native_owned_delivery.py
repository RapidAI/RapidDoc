"""Validate saved native-owned-line delivery evidence without rerunning parses."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

from accept_doc_performance import installed_binary

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf8'))


def exact(left,right):
    # Keep integer/float distinctions, strings, ordering and unknown fields.
    return json.dumps(left,sort_keys=True,separators=(',',':'))==json.dumps(right,sort_keys=True,separators=(',',':'))


def main():
    before=read('build/native-owned-public-before-20261101.json')
    after=read('build/native-owned-public-after-20261101.json')
    assert before['options']==after['options']
    assert [r['markdown_sha256'] for r in before['rows']]==[r['markdown_sha256'] for r in after['rows']]
    for count in [16,65,129]:
        assert exact(read(f'build/document-memory-20261101/pdf-verify-rust/{count}.json'),read(f'build/native-owned-window-verify-20261101/{count}.json')),count
    windows=read('build/native-owned-window-verify-20261101/report.json')
    assert windows['success'] and len(windows['rows'])==3
    public=read('build/native-owned-pdf-public-20261101/summary.json')
    assert public['all_passed'] and len(public['checks'])==6
    assert read('build/native-owned-pdf-public-20261101/manifest.json')['installed_rust']==installed_binary()
    fixture_path=ROOT/'tests/fixtures/native_owned_line_reference.json.gz'
    fixture=json.loads(gzip.decompress(fixture_path.read_bytes()))
    for name,digest in fixture['source_sha256'].items():
        assert hashlib.sha256((ROOT/'rapid_doc/_vendor/docvortex_core'/name).read_bytes()).hexdigest()==digest,name
    files=[
        'rust/rapiddoc/src/processing/text/native_text_roles/mod.rs',
        'rust/rapiddoc/src/processing/text/native_footnotes/mod.rs',
        'rust/rapiddoc/src/processing/document/native_page_preparation/mod.rs',
        'rust/rapiddoc/src/processing/document/inline_output/mod.rs',
        'rust/rapiddoc/src/processing/document/inline_output/payloads.rs',
        'rust/rapiddoc/src/processing/table/table_text_styles/clusters.rs',
        'rust/rapiddoc/src/processing/table/table_text_styles/preparation.rs',
        'rust/rapiddoc/src/processing/document/page_containers/mod.rs',
        'rust/rapiddoc/src/processing/layout/native_geometry/mod.rs',
        'rust/rapiddoc/src/processing/table/table_page.rs',
    ]
    baseline=read('build/native-owned-ab-20261101.json')
    assert baseline['all_outputs_equal']
    keys={(r['case'],r['trial']) for r in baseline['rows']}
    assert all(len({r['sha256'] for r in baseline['rows'] if (r['case'],r['trial'])==key})==1 for key in keys)
    wheel=ROOT/'build/native-owned-lines-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl'
    installed=installed_binary()
    with zipfile.ZipFile(wheel) as archive:
        binary=next(n for n in archive.namelist() if n.endswith('/_native.pyd'))
        assert hashlib.sha256(archive.read(binary)).hexdigest()==installed['native_files'][0]['sha256']
    result=dict(installed_rust=installed_binary(),wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        python_reference_cases={k:len(fixture[k]) for k in ['roles','notes','clusters','scripts']},
        window_sizes=[16,65,129],window_outputs_exact=True,public_pdf_checks=6,
        prior_raw_ab_exact=True,wheel_native_matches_installed=True,
        validation=dict(native_focused_tests=3,python_resource_policy_tests=5,original_related_tests=2,public_pdf_cases=6,strict_window_snapshots=3),
        public_native_seconds_before=[r['seconds'] for r in before['rows']],
        public_native_seconds_after=[r['seconds'] for r in after['rows']],
        notes=['Public timings are one cold and one warm call per version, not an alternating multi-round benchmark.',
               'Native-window complete snapshots are strict; no text/structure tolerances added.',
               'No new Office, full PDF corpus, GPU or full workspace regression.',
               'This reduces ownership copies and unbounded byte counting, not native parsing rules.'])
    target=ROOT/'build/native-owned-delivery-receipt-20261101.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ['python_reference_cases','window_outputs_exact','public_pdf_checks','public_native_seconds_before','public_native_seconds_after']},indent=2))


if __name__=='__main__':main()
