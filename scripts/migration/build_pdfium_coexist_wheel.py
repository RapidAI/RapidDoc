"""Refresh only the PDFium Python boundary in an existing wheel; no Rust compilation."""
import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CHANGED = ['rapid_doc_rust/__init__.py', 'rapid_doc_rust/_pdfium_runtime.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sdk-dir', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve(strict=True)
    destination = args.output_dir.resolve()
    sdk = args.sdk_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    sdk.mkdir(parents=True, exist_ok=False)
    target = destination/base.name
    records = []
    with zipfile.ZipFile(base) as original, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as output:
        record = next(name for name in original.namelist() if name.endswith('.dist-info/RECORD'))
        retired_bytecode = [name for name in original.namelist() if '/__pycache__/' in name or name.endswith('.pyc')]
        names = [name for name in original.namelist() if name != record and name not in retired_bytecode]
        names.extend(name for name in CHANGED if name not in names)
        for name in names:
            data = ((ROOT/'bindings/python/python'/name).read_bytes() if name in CHANGED else original.read(name))
            output.writestr(name, data)
            checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
            records.append((name, 'sha256='+checksum, str(len(data))))
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerows(records + [(record, '', '')])
        output.writestr(record, stream.getvalue().encode('utf-8'))
    with zipfile.ZipFile(target) as package, zipfile.ZipFile(base) as original:
        for name, checksum, length in records:
            data = package.read(name)
            assert len(data) == int(length)
            assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode() == checksum[7:]
            if name not in CHANGED:
                assert data == original.read(name), name
        package.extractall(sdk)
    manifest = dict(base=str(base), base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                    wheel=str(target), wheel_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    sdk=str(sdk), changed_files=CHANGED, other_files_unchanged=True,
                    removed_generated_bytecode=retired_bytecode,
                    rust_recompiled=False, record_verified=True)
    (destination/'build-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
