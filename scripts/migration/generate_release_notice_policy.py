"""Record target runtime graphs and explicitly selected license alternatives."""
import json
from pathlib import Path
import re
import subprocess

from license_bundle import TEXT, INDEX, pack, sha
from shared_license_terms import pack_shared
from release_notice_policy import ROOT, POLICY, CARGO_NOTICES


def selected_files(package):
    source=Path(package['manifest_path']).parent
    files={path.name:path.read_bytes() for pattern in ('LICENSE*','COPYING*','NOTICE*','COPYRIGHT*')
           for path in source.glob(pattern) if path.is_file()}
    expression=package.get('license') or ''
    alternatives=expression.split(' OR ')
    simple='MIT' in alternatives and all(re.fullmatch(r'[A-Za-z0-9.+-]+',item) for item in alternatives)
    unicode_choice=expression.replace(' ', '') == '(MITORApache-2.0)ANDUnicode-3.0'
    mit=next((name for name in files if name.upper() in ('LICENSE-MIT','LICENSE-MIT.MD','LICENSE-MIT.TXT')),None)
    selected=set(files)
    if mit and (simple or unicode_choice):
        alternatives={'LICENSE-APACHE','LICENSE-APACHE.MD','LICENSE-APACHE.TXT','LICENSE-ZLIB','LICENSE-ZLIB.MD','LICENSE-0BSD'}
        selected={name for name in files if name.upper() not in alternatives}
    return files, selected


def main():
    meta=json.loads(subprocess.check_output(['cargo','metadata','--locked','--offline','--format-version','1'],cwd=ROOT))
    packages={(p['name']+' '+p['version']):p for p in meta['packages']}
    by_id={p['id']:p for p in meta['packages']}
    nodes={n['id']:n for n in meta['resolve']['nodes']}
    profiles={}
    for label,target in [('windows','x86_64-pc-windows-msvc'),('linux','x86_64-unknown-linux-gnu')]:
        text=subprocess.check_output(['cargo','tree','--locked','--offline','-p','rapiddoc-python','--features','cuda',
            '--edges','normal,no-proc-macro','--target',target,'--prefix','none','--format','{p}'],cwd=ROOT,text=True)
        profiles[label]=sorted({m[1]+' '+m[2] for line in text.splitlines() if (m:=re.match(r'^(\S+) v(\S+)',line))})
    profiles['all']=sorted(set(profiles['windows'])|set(profiles['linux']))
    pending=[p['id'] for p in meta['packages'] if p['name'] in ('ureq','ureq-proto','sha2','parking_lot','parking_lot_core','lock_api','scopeguard')]
    seen=set()
    while pending:
        ident=pending.pop()
        if ident in seen:continue
        seen.add(ident)
        for dep in nodes[ident]['deps']:
            if any(k['kind'] is None for k in dep['dep_kinds']) and not any('proc-macro' in t['kind'] for t in by_id[dep['pkg']]['targets']):
                pending.append(dep['pkg'])
    generated={};rules={};choices={}
    for ident in sorted(seen):
        p=by_id[ident];component=p['name']+' '+p['version']
        if component not in profiles['all'] or not (p.get('source') or '').startswith('registry+'):continue
        files,keep=selected_files(p)
        if not files:raise ValueError('Missing source license: '+component)
        choices[component]=dict(spdx=p['license'],selected_files=sorted(keep),repository=p.get('repository'))
        for name in keep:
            path=f"cargo-runtime/{p['name']}-{p['version']}/{name}"
            generated[path]=files[name]
            rules[path]=dict(component=component,selected=True)
    for domain in ('image','chart'):
        manifest=json.loads((ROOT/f'packaging/office-{domain}-dependency-notices.json').read_text('utf8'))
        for p in manifest['packages']:
            component=p['name']+' '+p['version']
            package=packages.get(component)
            if package is None:raise ValueError('Unrecorded dependency '+component)
            files,keep=selected_files(package)
            choices[component]=dict(spdx=package['license'],selected_files=sorted(keep),repository=p.get('repository'))
            for original in p['notices']:
                path='office/'+original.split('rust/rapiddoc-office/licenses/',1)[1]
                rules[path]=dict(component=component,selected=Path(original).name in keep)
    for name in ('RATIONAL','TRAITS'):
        rules[f'NUM-{name}-LICENSE-APACHE.txt']=dict(component='num-'+name.lower()+' '+('0.4.2' if name=='RATIONAL' else '0.2.19'),selected=False)
    # Only Adobe glyph data and the conversion algorithm are copied, not the
    # fontTools test font collection, cu2qu, or its other optional components.
    import fontTools.agl
    lines=fontTools.agl._aglText.splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith('# Copyright'))
    end=next(i+1 for i,line in enumerate(lines[start:],start) if 'SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.' in line)
    generated['fonttools-glyph-data/ADOBE-AGL.txt']=('\n'.join(line[2:] if line.startswith('# ') else line.removeprefix('#') for line in lines[start:end])+'\n').encode()
    # The reviewed source archive contains third-party attribution only.
    migration=(ROOT/'licenses/MIGRATION-NOTICES.txt').read_bytes()
    migration_index=json.loads((ROOT/'licenses/migration-notices.json').read_text('utf8'))
    retained_migration=[]
    for entry in migration_index['entries']:
        data=migration[entry['start']:entry['end']]
        if sha(data)!=entry['sha256']:raise ValueError('Migration archive changed')
        path='migration/'+entry['name']+'/'+sha(data)[:12]+'.txt'
        generated[path]=data;retained_migration.append(path)
    CARGO_NOTICES.mkdir(exist_ok=True)
    text,index=pack_shared(generated)
    (CARGO_NOTICES/TEXT).write_bytes(text);(CARGO_NOTICES/INDEX).write_bytes(index)
    manifests={ROOT/'Cargo.toml'}
    manifests.update(Path(p['manifest_path']) for p in meta['packages']
                     if p.get('source') is None and p['name']+' '+p['version'] in profiles['all'])
    manifests.update(ROOT/name for name in ('licenses/MIGRATION-NOTICES.txt','licenses/migration-notices.json'))
    manifest_hashes={path.relative_to(ROOT).as_posix():sha(path.read_bytes().replace(b'\r\n',b'\n')) for path in sorted(manifests)}
    policy=dict(schema=1,cargo_lock_sha256=sha((ROOT/'Cargo.lock').read_bytes().replace(b'\r\n',b'\n')),manifest_hashes=manifest_hashes,profiles=profiles,files=rules,license_choices=choices,retained_migration=retained_migration,
                scope='Windows MSVC and Linux glibc runtime (CPU and CUDA); normal edges excluding proc-macro/build/dev',
                source_audit_records='Development provenance and migration records remain in source; Cargo license alternatives are read from locked registry sources')
    POLICY.write_text(json.dumps(policy,indent=2)+'\n','utf8')
    print('Recorded',len(rules),'notice decisions;',len(generated),'selected runtime notice files')


if __name__=='__main__':main()
