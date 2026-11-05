"""Verify the full parser copy and its three explicitly recorded f64 patches."""
import argparse,json
from common import ROOT,write_json,digest
from license_bundle import read_notice

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');args=parser.parse_args()
    folder=ROOT/'rust/vendor/lopdf-f64';upstream=json.loads((folder/'UPSTREAM.json').read_text(encoding='utf-8'));target=folder/'PATCHES.json'
    allowed={'src/object.rs','src/parser/mod.rs','src/xobject.rs'};patched={};packaging={}
    formatting=json.loads(target.read_text(encoding='utf-8')).get('formatting',{}) if target.exists() else {}
    for path,expected in upstream['files'].items():
        actual=digest((folder/path).read_bytes())
        if actual!=expected:
            if path in formatting:
                assert formatting[path]==dict(upstream_sha256=expected,patched_sha256=actual),f'Stale formatting record: {path}'
                continue
            if path=='Cargo.toml':
                packaging[path]=dict(upstream_sha256=expected,patched_sha256=actual)
                continue
            assert path in allowed,f'Unexpected upstream change: {path}'
            patched[path]=dict(upstream_sha256=expected,patched_sha256=actual)
    assert set(patched)==allowed
    record=dict(crate='lopdf 0.42.0',purpose='Preserve f64 PDF Real parsing; retain f32 accessor compatibility.',files=patched)
    if packaging:
        record['packaging']=packaging
    if formatting:
        record['formatting']=formatting
    if args.record:
        if target.exists():raise FileExistsError(target)
        write_json(target,record)
    else:assert json.loads(target.read_text(encoding='utf-8'))==record,'Stale parser patch manifest'
    for path in ['licenses/LOPDF-LICENSE.txt','bindings/python/python/rapid_doc_rust/licenses/LOPDF-LICENSE.txt']:
        dest=ROOT/path
        if args.record:
            if dest.exists():raise FileExistsError(dest)
            dest.write_bytes((folder/'LICENSE').read_bytes())
        assert read_notice(dest)==(folder/'LICENSE').read_bytes()
    print(len(upstream['files']),'upstream files;',len(patched),'recorded f64 patches; MIT licenses preserved')
if __name__=='__main__':main()
