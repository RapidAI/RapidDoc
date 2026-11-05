"""Sequential installed-wheel PDF timing; final environment retains candidate.

Run after all builds/tests stop. Uses existing lightweight phase diagnostics,
not stack sampling, cProfile or verification work inside the parse timer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from accept_doc_performance import installed_binary

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calls', type=int, default=2)
    parser.add_argument('--native-rule', action='store_true')
    parser.add_argument('--python', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    args.source = args.source.resolve()
    wheels = {k:getattr(args,k).resolve() for k in ['baseline','candidate']}
    assert all(p.is_file() for p in wheels.values())
    manifest = dict(source=str(args.source), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        wheels={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in wheels.items()},
        calls=args.calls,native_rule=args.native_rule,measurements=[])

    def save():
        (args.output/'report.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

    def install(name):
        with (args.output/(name+'-install.log')).open('w',encoding='utf8') as log:
            subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--force-reinstall',str(wheels[name])],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)

    installed_candidate = False
    save()
    try:
        for name in ['baseline','candidate'] + (['python'] if args.python else []):
            if name != 'python':
                install(name); installed_candidate = name == 'candidate'
            print('Measuring '+name,flush=True)
            binary = installed_binary() if name != 'python' else None
            output = args.output/(name+'.json')
            with (args.output/(name+'.log')).open('w',encoding='utf8') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/migration/profile_pdf_pipeline.py'),
                    '--engine','python' if name=='python' else 'rust','--source',str(args.source),
                    '--output',str(output),'--calls',str(args.calls),
                    *(['--native-rule'] if args.native_rule else [])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    check=True,timeout=900)
            result=json.loads(output.read_text(encoding='utf8'))
            manifest['measurements'].append(dict(version=name,installed_rust=binary,rows=result['rows']))
            save()
        hashes={r['markdown_sha256'] for m in manifest['measurements'] for r in m['rows']}
        assert len(hashes)==1,'Markdown changed; do not accept timing results'
        assert hashlib.sha256(args.source.read_bytes()).hexdigest()==manifest['source_sha256']
        manifest.update(success=True,markdown_equal=True,notes=[
            'Versions run sequentially, baseline then candidate then optional Python, not alternating multiple rounds.',
            'Each first call includes lazy initialization; following calls reuse that process engine.',
            'Full output parity and memory must be verified separately; native stage rows may overlap.'])
        save()
    finally:
        if not installed_candidate:
            install('candidate')


if __name__=='__main__':
    main()
