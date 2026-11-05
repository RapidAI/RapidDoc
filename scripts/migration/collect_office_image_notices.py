"""Collect published dependency notices for the native Office image engines."""
import json,subprocess
from pathlib import Path
from license_bundle import TEXT, INDEX, pack, read_tree, compact
from shared_license_terms import pack_shared
from release_notice_policy import filter_entries
ROOT=Path(__file__).resolve().parents[2]
def main(roots=('image','tiff','resvg','usvg'), destination=Path('rust/rapiddoc-office/licenses/image-dependencies'), output_path='packaging/office-image-dependency-notices.json'):
    meta=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version','1'],cwd=ROOT,text=True,encoding='utf-8'))
    packages={p['id']:p for p in meta['packages']};nodes={n['id']:n for n in meta['resolve']['nodes']}
    selected=[p for p in meta['packages'] if p['name'] in roots]
    pending=[p['id'] for p in selected];seen=set();items=[];missing=[];collected={}
    while pending:
        ident=pending.pop()
        if ident in seen:continue
        seen.add(ident);package=packages[ident];pending.extend(d['pkg'] for d in nodes[ident]['deps'])
        if not (package.get('source') or '').startswith('registry+'):continue
        source=Path(package['manifest_path']).parent;target=destination/(package['name']+'-'+package['version']);files=[]
        for pattern in ['LICENSE*','COPYING*','NOTICE*','COPYRIGHT*']:
            for path in source.glob(pattern):
                if not path.is_file() or path.name in files:continue
                collected[(target/path.name).relative_to('rust/rapiddoc-office/licenses').as_posix()]=path.read_bytes();files.append(path.name)
        if not files:missing.append(package['name'])
        items.append(dict(name=package['name'],version=package['version'],license=package['license'],repository=package.get('repository'),notices=[(target/file).as_posix() for file in files]))
    folder=ROOT/'rust/rapiddoc-office/licenses'
    prefixes=('image-dependencies/','chart-dependencies/')
    previous=read_tree(folder)
    replace_prefix=destination.relative_to('rust/rapiddoc-office/licenses').as_posix()+'/'
    entries={name:data for name,data in previous.items() if name.startswith(prefixes) and not name.startswith(replace_prefix)}
    entries.update(collected)
    entries,_=filter_entries(entries,'office-source','all')
    text,index=pack_shared(entries)
    (folder/TEXT).write_bytes(text);(folder/INDEX).write_bytes(index)
    compact(folder,selected_prefixes=prefixes,shared_terms=True)
    output=dict(roots=[p['name']+' '+p['version'] for p in selected],packages=sorted(items,key=lambda v:v['name']),missing_top_level_notices=missing,
                notice_archive=dict(text='rust/rapiddoc-office/licenses/'+TEXT,index='rust/rapiddoc-office/licenses/'+INDEX,
                                    paths='notices paths identify original entries in the archive'))
    (ROOT/output_path).write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print('Office dependency notices:',len(items),'packages; missing:',missing)
if __name__=='__main__':main()
