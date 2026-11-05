"""Adjacent-process checks for table no-op PNGs and borrowed CTC outputs."""
from pathlib import Path
import hashlib
import io
import json
import re
import sys
import time

from loguru import logger
from PIL import Image

root = Path(__file__).resolve().parents[2]
mode = sys.argv[1]
assert mode in ('old', 'new')
if mode == 'new':
    sys.path.insert(0, str(root / 'build/api-dev-sdk'))
import rapid_doc_rust as sdk

namespace = sys.argv[2] if len(sys.argv) > 2 else 'borrowed-outputs'
assert Path(namespace).name == namespace
destination = root / 'build' / f'{namespace}-control-{mode}.json'
assert not destination.exists(), destination
fixtures = json.loads((root / 'build/migration/user-seven-pdfs-20261029.json').read_text(encoding='utf-8'))
prior = json.loads((root / 'build/migration/user-seven-rust-before-phase-20261029.json').read_text(encoding='utf-8'))
baselines = {c['source']: c for c in prior['cases']}
engine = sdk.RapidDoc(ocr_config={'seal_enable': False})
summaries = []
stems = sys.argv[3:] or ('small_ocr', '49－长表格-密集型', '示例7-研究报告')
for stem in stems:
    case = next(c for c in fixtures['cases'] if Path(c['source']).stem == stem)
    source = root / case['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == case['sha256']
    logs = []
    token = logger.add(lambda record: logs.append(str(record)), level='INFO')
    started = time.perf_counter()
    try:
        result = engine(source)
        elapsed = time.perf_counter() - started
    finally:
        logger.remove(token)
    assert result.markdown == case['expected']['markdown']
    assert result.content_list_json == case['expected']['content_list_json']
    assert result.middle_json == baselines[case['source']]['expected']['middle_json']
    assets = []
    for name, png in sorted(result.images.items()):
        with Image.open(io.BytesIO(png)) as image:
            assets.append(dict(name=name, mode=image.mode, size=list(image.size),
                               pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
    assert assets == case['assets']
    phases = {m.group(1): dict(items=int(m.group(2)), seconds=float(m.group(3)))
              for m in re.finditer(re.escape(stem) + r': ([a-z_]+): (\d+) items in ([0-9.]+)s', ''.join(logs))}
    summaries.append(dict(source=case['source'], seconds=elapsed, phases=phases,
                          pages=len(result.middle_json['pdf_info']), assets=len(assets),
                          markdown_sha256=hashlib.sha256(result.markdown.encode()).hexdigest(),
                          matches_python_markdown_content_assets=True,
                          matches_prior_rust_middle=True,
                          matches_python_middle=result.middle_json == case['expected']['middle_json']))
    print('Borrowed output control:', json.dumps(summaries[-1], ensure_ascii=False), flush=True)
destination.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
