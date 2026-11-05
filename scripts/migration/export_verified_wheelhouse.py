"""Export a RECORD-verified installed dependency closure for offline clean tests.

No source distribution or environment is modified. Console scripts are recreated
by pip from entry_points; bytecode and installation bookkeeping are omitted.
"""
import argparse
import base64
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import zipfile

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-wheel',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(args.release_wheel) as wheel:
        name=next(n for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))
        info=BytesParser(policy=policy.compat32).parsebytes(wheel.read(name))
    pending=list(info.get_all('Requires-Dist',[]))
    selected={}
    while pending:
        req=Requirement(pending.pop())
        if req.marker and not req.marker.evaluate({'extra':''}):
            continue
        installed=metadata.distribution(req.name)
        if not req.specifier.contains(installed.version):
            raise ValueError(f'Installed {req.name} {installed.version} does not satisfy {req}')
        key=canonicalize_name(req.name)
        if key in selected:
            continue
        selected[key]=installed
        pending.extend(installed.requires or [])
    receipts=[]
    supported_tags={str(tag) for tag in sys_tags()}
    for key,dist in sorted(selected.items()):
        root=Path(dist.locate_file('')).resolve()
        files=[]
        wheel_metadata=next(p for p in dist.files if str(p).endswith('.dist-info/WHEEL'))
        headers=BytesParser(policy=policy.compat32).parsebytes(Path(dist.locate_file(wheel_metadata)).read_bytes())
        tag=next(tag for tag in headers.get_all('Tag') if tag in supported_tags)
        destination=args.output/f"{key.replace('-','_')}-{dist.version}-{tag}.whl"
        entries=[]
        verified=0
        omitted=[]
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as wheel:
            for entry in dist.files:
                name=entry.as_posix()
                path=Path(dist.locate_file(entry)).resolve()
                if '..' in entry.parts or name.endswith(('.pyc','/RECORD','/INSTALLER','/REQUESTED','/direct_url.json')):
                    omitted.append(name)
                    continue
                if not path.is_relative_to(root):
                    raise ValueError(f'Unexpected external installed asset: {path}')
                raw=path.read_bytes()
                if entry.hash:
                    checksum=base64.urlsafe_b64encode(hashlib.new(entry.hash.mode,raw).digest()).rstrip(b'=').decode()
                    if checksum!=entry.hash.value:
                        raise ValueError(f'Installed dependency differs from original RECORD: {path}')
                    verified+=1
                wheel.writestr(name,raw)
                checksum=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b'=').decode()
                entries.append((name,'sha256='+checksum,str(len(raw))))
            record=str(wheel_metadata).rsplit('/',1)[0]+'/RECORD'
            text=io.StringIO(newline='')
            csv.writer(text,lineterminator='\n').writerows(entries+[(record,'','')])
            wheel.writestr(record,text.getvalue())
        receipt=dict(package=key,version=dist.version,file=destination.name,
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            files=len(entries),original_record_verified_files=verified,omitted=omitted,
            provenance='Repacked installed distribution, validated against original installed RECORD; not an upstream wheel download')
        receipts.append(receipt)
        print(key,dist.version,'RECORD_VERIFIED',verified,flush=True)
    (args.output/'manifest.json').write_text(json.dumps(receipts,indent=2))


if __name__=='__main__':
    main()
