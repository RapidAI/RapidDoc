"""Consolidate generated release notices; never change parser/runtime binaries."""
import argparse
import base64
import csv
import io
import json
from pathlib import Path
import zipfile

from license_bundle import INDEX, TEXT, compact, compact_assets, read_tree, sha
from release_notice_policy import filter_entries

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'bindings/python/python/rapid_doc_rust'


def stage():
    results = {}
    manifest = PACKAGE / 'runtime/manifest.json'
    platform=json.loads(manifest.read_text('utf8')).get('platform') if manifest.is_file() else None
    profile={'win32':'windows','linux':'linux'}.get(platform,'all')
    if manifest.is_file() and 'musl' in str(json.loads(manifest.read_text('utf8')).get('version',{}).get('origin','')).lower():
        profile='all'  # Do not apply a glibc exclusion to an unknown/musl binary.
    for relative in ('licenses', 'runtime/licenses', 'runtime/onnx_ops/licenses'):
        directory = PACKAGE / relative
        if directory.is_dir():
            entries=read_tree(directory)
            kind={'licenses':'main','runtime/licenses':'pdfium'}.get(relative,'ops')
            selected,removed=filter_entries(entries,kind,profile)
            results[relative] = compact(directory,entries=selected,shared_terms=True)
            results[relative].update(removed=removed,original_text_bytes_before=sum(map(len,entries.values())),
                                     original_text_bytes_after=sum(map(len,selected.values())))
    if manifest.is_file():
        data = json.loads(manifest.read_text('utf8'))
        data['license_files'] = {p.relative_to(manifest.parent).as_posix(): sha(p.read_bytes())
                                 for p in sorted((manifest.parent/'licenses').rglob('*')) if p.is_file()}
        manifest.write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
        (ROOT/'packaging/pdfium-runtime-manifest.json').write_bytes(manifest.read_bytes())
    return results


def repack(source, destination):
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Use a new output wheel; retain the original artifact')
    with zipfile.ZipFile(source) as wheel:
        original = {name: wheel.read(name) for name in wheel.namelist() if not name.endswith('/')}
    record = next(name for name in original if name.endswith('.dist-info/RECORD'))
    files = {name: data for name, data in original.items() if name != record}
    profile='windows' if 'win_amd64' in source.name else 'linux' if 'manylinux' in source.name else 'all'
    decisions=[]
    for prefix,kind in [('rapid_doc_rust/licenses/','main'),('rapid_doc_rust/runtime/licenses/','pdfium'),
                       ('rapid_doc_rust/runtime/onnx_ops/licenses/','ops')]:
        def select(entries):
            selected,removed=filter_entries(entries,kind,profile)
            decisions.extend(dict(tree=prefix,**item) for item in removed)
            return selected
        compact_assets(files, prefix, transform=select,shared_terms=True)
    manifest = 'rapid_doc_rust/runtime/manifest.json'
    if manifest in files:
        data = json.loads(files[manifest])
        prefix='rapid_doc_rust/runtime/'
        data['license_files'] = {name[len(prefix):]: sha(raw) for name,raw in files.items()
                                 if name.startswith(prefix+'licenses/')}
        files[manifest] = (json.dumps(data, indent=2)+'\n').encode()
    # Only the notice trees, their runtime hashes, and wheel RECORD may differ.
    changed = set(original) | set(files)
    changed = {n for n in changed if original.get(n) != files.get(n)}
    if any('/licenses/' not in name and name not in (record, manifest) for name in changed):
        raise ValueError('Notice-only repack attempted to change a parser/runtime file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as wheel:
        for name, data in sorted(files.items()):
            wheel.writestr(name, data)
            checksum = base64.urlsafe_b64encode(bytes.fromhex(sha(data))).rstrip(b'=').decode()
            rows.append((name, 'sha256='+checksum, str(len(data))))
        output = io.StringIO(newline='')
        csv.writer(output, lineterminator='\n').writerows(rows+[(record, '', '')])
        wheel.writestr(record, output.getvalue())
    with zipfile.ZipFile(destination) as wheel:
        assert all(wheel.read(name) == data for name, data in files.items())
    return dict(before=sum('/licenses/' in n for n in original),
                after=sum('/licenses/' in n for n in files),
                parser_and_runtime_bytes_unchanged=True, wheel=str(destination),
                sha256=sha(destination.read_bytes()),profile=profile,removed=decisions,
                license_bytes_before=sum(len(data) for n,data in original.items() if '/licenses/' in n),
                license_bytes_after=sum(len(data) for n,data in files.items() if '/licenses/' in n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.wheel:
        if not args.output:
            parser.error('--wheel requires --output')
        result = repack(args.wheel, args.output)
    else:
        result = stage()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
