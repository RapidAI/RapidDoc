"""Adjacent unchanged-model control for the observed 03 formula timing swing."""
from pathlib import Path
import hashlib,json,re,sys,time
from loguru import logger
root=Path(__file__).resolve().parents[2]
mode=sys.argv[1]
stem=sys.argv[2]if len(sys.argv)>2 else'03'
if mode=='new':sys.path.insert(0,str(root/'build/api-dev-sdk'))
import rapid_doc_rust as sdk
f=json.loads((root/'build/migration/user-seven-pdfs-20261029.json').read_text(encoding='utf-8'))
case=next(c for c in f['cases']if Path(c['source']).stem==stem)
prior=json.loads((root/'build/migration/user-seven-rust-before-phase-20261029.json').read_text(encoding='utf-8'))
baseline=next(c for c in prior['cases']if c['source']==case['source'])
source=root/case['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==case['sha256']
logs=[];sink=logger.add(lambda record:logs.append(str(record)),level='INFO')
engine=sdk.RapidDoc(ocr_config={'seal_enable':False})
started=time.perf_counter();result=engine(source);elapsed=time.perf_counter()-started;logger.remove(sink)
assert result.markdown==case['expected']['markdown']
assert result.middle_json==baseline['expected']['middle_json']
assert result.content_list_json==case['expected']['content_list_json']
phases={m.group(1):dict(items=int(m.group(2)),seconds=float(m.group(3)))for m in re.finditer(re.escape(stem)+r': ([a-z_]+): (\d+) items in ([0-9.]+)s',''.join(logs))}
out=dict(mode=mode,seconds=elapsed,phases=phases,markdown_sha256=hashlib.sha256(result.markdown.encode()).hexdigest())
suffix=''if stem=='03'else'-'+stem
path=root/'build'/f'recognition-execution-control-{mode}{suffix}.json';assert not path.exists()
path.write_text(json.dumps(out,indent=2),encoding='utf-8');print('Adjacent execution control:',json.dumps(out),flush=True)
