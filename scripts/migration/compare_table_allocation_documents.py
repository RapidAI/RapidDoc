"""Alternating unchanged-document comparison after noisy sequential timings."""
from pathlib import Path
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[2]
names = {'dense': '49－长表格-密集型', 'finance': '比亚迪财报'}
if '--worker' in sys.argv:
    if '--dev' in sys.argv:
        sys.path.insert(0, str(root / 'build/api-dev-sdk'))
    from PIL import Image
    import rapid_doc_rust as sdk
    cases = json.loads((root / 'build/migration/user-seven-pdfs-20261029.json').read_text(encoding='utf-8'))['cases']
    prior = json.loads((root / 'build/migration/user-seven-rust-before-phase-20261029.json').read_text(encoding='utf-8'))['cases']
    baselines = {c['source']: c for c in prior}
    engine = sdk.RapidDoc(ocr_config={'seal_enable': False})
    print('ready', flush=True)
    for line in sys.stdin:
        key = line.strip()
        case = next(c for c in cases if Path(c['source']).stem == names[key])
        source = root / case['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == case['sha256']
        started = time.perf_counter()
        result = engine(source)
        seconds = time.perf_counter() - started
        actual = dict(markdown=result.markdown, middle_json=result.middle_json, content_list_json=result.content_list_json)
        assert actual == baselines[case['source']]['expected']
        assert result.markdown == case['expected']['markdown']
        assert result.content_list_json == case['expected']['content_list_json']
        assets = []
        for name, data in sorted(result.images.items()):
            with Image.open(io.BytesIO(data)) as image:
                assets.append(dict(name=name, mode=image.mode, size=list(image.size),
                                   pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        assert assets == case['assets']
        digest = hashlib.sha256(json.dumps([actual, assets], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        print(json.dumps(dict(seconds=seconds, sha256=digest)), flush=True)
    raise SystemExit

destination = root / 'build/native-table-allocation-document-paired.json'
assert not destination.exists(), destination
workers, logs = {}, []
results = {mode: {key: [] for key in names} for mode in ('old', 'new')}
warmup = {mode: {} for mode in results}
try:
    for mode in results:
        log = (root / f'build/native-table-allocation-document-paired-{mode}.txt').open('x', encoding='utf-8')
        logs.append(log)
        workers[mode] = subprocess.Popen([sys.executable, __file__, '--worker', *(['--dev'] if mode == 'new' else [])],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8',
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    for worker in workers.values():
        assert worker.stdout.readline().strip() == 'ready'
    for key in names:
        expected = None
        for iteration in range(4):
            for mode in (('old', 'new') if iteration % 2 == 0 else ('new', 'old')):
                worker = workers[mode]
                worker.stdin.write(key + '\n')
                worker.stdin.flush()
                raw = worker.stdout.readline()
                assert raw, f'{mode} worker failed; see its log'
                value = json.loads(raw)
                if expected is None:
                    expected = value['sha256']
                assert value['sha256'] == expected
                if iteration:
                    results[mode][key].append(value['seconds'])
                else:
                    warmup[mode][key] = value['seconds']
        print(key, {mode: statistics.median(results[mode][key]) for mode in results}, flush=True)
finally:
    for worker in workers.values():
        worker.stdin.close()
        try:
            code = worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=10)
            raise
        assert code == 0
    for log in logs:
        log.close()
destination.write_text(json.dumps(dict(scope='Separate persistent workers, identical inputs/configuration; alternate order, one warm-up per document and three measured parses; output validation outside parse timer; no concurrent parsing or Cargo', seconds=results, warmup_seconds=warmup), indent=2), encoding='utf-8')
