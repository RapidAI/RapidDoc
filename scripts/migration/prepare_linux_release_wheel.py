"""Stage Linux runtime assets into a built wheel without recompiling Rust.

The optional CUDA profile changes only distribution dependencies, not parser
code. Install one profile in an environment; CPU/GPU ORT packages overlap files.
"""
import argparse
import base64
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from license_bundle import compact_assets
from release_notice_policy import filter_entries


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pdfium-root', type=Path, required=True)
    parser.add_argument('--pdfium-licenses', type=Path, required=True)
    parser.add_argument('--geos-root', type=Path, required=True)
    parser.add_argument('--geos-licenses', type=Path, required=True,
                        help='Original notices from the distribution supplying GEOS shared libraries')
    parser.add_argument('--patchelf', type=Path, required=True,
                        help='Build-only tool: set the bundled GEOS C library RUNPATH to $ORIGIN')
    parser.add_argument('--cuda-dependencies', type=Path,
                        help='JSON array of exact validated ORT GPU/NVIDIA requirements')
    args = parser.parse_args()
    parts = args.base.name[:-4].split('-')
    if len(parts) != 5 or 'manylinux' not in parts[-1]:
        parser.error('Expected an unmodified Linux wheel filename')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    profile = 'cuda' if args.cuda_dependencies else 'cpu'
    destination = args.output_dir / ('-'.join(parts[:2] + ['1'+profile] + parts[2:]) + '.whl')
    assets = {'rapid_doc_rust/runtime/librapiddoc_pdfium.so':
              (args.pdfium_root/'libpdfium.so').read_bytes()}
    prefix = 'rapid_doc_rust/runtime/'
    libraries = sorted(args.geos_root.glob('libgeos*.so*'))
    if len(libraries) != 2 or not any(p.name.startswith('libgeos_c') for p in libraries):
        raise ValueError('Exactly the paired GEOS and GEOS C shared libraries are required')
    if not any(args.pdfium_licenses.rglob('*.txt')):
        raise ValueError('PDFium original notices are required')
    for path in libraries:
        assets[prefix+'geos/'+path.name] = path.read_bytes()
    # Shapely preloads GEOS itself; its distributed C library has no RUNPATH.
    # This standalone package must locate the adjacent core without importing
    # Shapely or requiring LD_LIBRARY_PATH. Change only ELF loader metadata.
    geos_c=next(path for path in libraries if path.name.startswith('libgeos_c'))
    with tempfile.TemporaryDirectory() as temporary:
        copied=Path(temporary)/geos_c.name
        copied.write_bytes(assets[prefix+'geos/'+geos_c.name])
        subprocess.run([str(args.patchelf.resolve()),'--set-rpath','$ORIGIN',str(copied)],check=True)
        actual=subprocess.check_output([str(args.patchelf.resolve()),'--print-rpath',str(copied)],text=True).strip()
        if actual!='$ORIGIN':
            raise ValueError('Bundled GEOS RUNPATH was not set')
        assets[prefix+'geos/'+geos_c.name]=copied.read_bytes()
    if not any(args.geos_licenses.rglob('*')):
        raise ValueError('GEOS distributor notices are required')
    for path in sorted(args.geos_licenses.rglob('*')):
        if path.is_file():
            assets[prefix+'geos/licenses/'+path.relative_to(args.geos_licenses).as_posix()] = path.read_bytes()
    for path in sorted(args.pdfium_licenses.rglob('*')):
        if path.is_file():
            assets[prefix+'licenses/'+path.relative_to(args.pdfium_licenses).as_posix()] = path.read_bytes()
    assets[prefix+'manifest.json'] = json.dumps(dict(platform='linux',
        library='librapiddoc_pdfium.so', sha256=sha(assets[prefix+'librapiddoc_pdfium.so']),
        version=json.loads((args.pdfium_root/'version.json').read_text()),
        source='Installed pypdfium2 build; original binary and notices, private library filename',
        license_files={name[len(prefix):]:sha(data) for name,data in assets.items() if name.startswith(prefix+'licenses/')}), indent=2).encode()
    preserved = {}
    with zipfile.ZipFile(args.base) as wheel:
        for name in wheel.namelist():
            if name.endswith('/RECORD') or name.endswith('.pyc') or '/__pycache__/' in name:
                continue
            if name.startswith(prefix+'licenses/') or name == prefix+'manifest.json':
                continue
            if name.startswith(prefix+'geos/') and name.endswith(('.dll', '.so')):
                continue
            if name == prefix+'geos/LICENSE_win32':
                continue
            preserved[name] = wheel.read(name)
    # GEOS 3.13.1 source and license were already carried by the native wheel.
    source_archives = [n for n in preserved if n.startswith(prefix+'geos/') and n.endswith('3.13.1.tar.bz2')]
    if not source_archives or not any('3.13.1' in p.name for p in libraries):
        raise ValueError('GEOS paired libraries must match the carried 3.13.1 source archive')
    assets[prefix+'geos/manifest.json'] = json.dumps(dict(platform='linux',version='3.13.1',
        libraries={path.name:sha(assets[prefix+'geos/'+path.name]) for path in libraries},
        original_libraries={path.name:sha(path.read_bytes()) for path in libraries},
        loader_change='GEOS C RUNPATH=$ORIGIN; core library and algorithm code unchanged',
        source_archive={name:sha(preserved[name]) for name in source_archives},
        source='Paired unmodified Shapely-distributed GEOS libraries; original distributor notices and matching LGPL source retained'),indent=2).encode()
    if args.cuda_dependencies:
        requirements = json.loads(args.cuda_dependencies.read_text())
        if not any(item.startswith('onnxruntime-gpu==') for item in requirements):
            raise ValueError('CUDA profile must pin the validated GPU runtime')
        name = next(n for n in preserved if n.endswith('.dist-info/METADATA'))
        metadata = BytesParser(policy=policy.compat32).parsebytes(preserved[name])
        retained = [v for v in metadata.get_all('Requires-Dist', []) if not v.startswith('onnxruntime')]
        del metadata['Requires-Dist']
        for value in retained + requirements:
            metadata['Requires-Dist'] = value
        preserved[name] = metadata.as_bytes()
    merged = {**preserved, **assets}
    for notice_root in ('rapid_doc_rust/licenses/', prefix+'licenses/', prefix+'onnx_ops/licenses/'):
        kind='main' if notice_root=='rapid_doc_rust/licenses/' else 'pdfium' if notice_root==prefix+'licenses/' else 'ops'
        compact_assets(merged, notice_root,transform=lambda entries:filter_entries(entries,kind,'linux')[0],shared_terms=True)
    # Refresh the PDFium attribution hashes after consolidating Linux notices.
    pdfium_manifest=json.loads(merged[prefix+'manifest.json'])
    pdfium_manifest['license_files']={name[len(prefix):]:sha(data) for name,data in merged.items()
                                    if name.startswith(prefix+'licenses/')}
    merged[prefix+'manifest.json']=json.dumps(pdfium_manifest,indent=2).encode()
    assets={name:data for name,data in merged.items() if name in assets or '/licenses/' in name}
    record = next(n.rsplit('/',1)[0]+'/RECORD' for n in merged if n.endswith('.dist-info/METADATA'))
    entries = []
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as wheel:
        for name,data in sorted(merged.items()):
            wheel.writestr(name, data)
            checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            entries.append((name, 'sha256='+checksum, str(len(data))))
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\n').writerows(entries+[(record,'','')])
        wheel.writestr(record, stream.getvalue())
    with zipfile.ZipFile(destination) as wheel:
        for name,checksum,length in entries:
            data=wheel.read(name)
            assert len(data)==int(length)
            assert 'sha256='+base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()==checksum
    receipt = dict(base=str(args.base),base_sha256=sha(args.base.read_bytes()),
        wheel=str(destination),wheel_sha256=sha(destination.read_bytes()),profile=profile,
        staged_assets={name:sha(data) for name,data in assets.items()},rust_recompiled=False,
        parser_files_unchanged=True, record_verified=True)
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
