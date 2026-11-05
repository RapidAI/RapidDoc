"""Distribution attribution must survive consolidation without text loss."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('notice_sync', ROOT/'scripts/migration/sync_migration_notices.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_source_and_distribution_preserve_every_original_notice():
    assert sync.sync(check=True) == 120
    index = json.loads((ROOT/'licenses/migration-notices.json').read_text(encoding='utf8'))
    versions = [e for e in index['entries'] if e['name'] == 'PDFTEXT-NOTICE.txt']
    assert len(versions) == 2  # differing originals must not overwrite each other
    assert len({v['sha256'] for v in versions}) == 2
    assert (ROOT/'bindings/python/python/rapid_doc_rust/licenses/PDFTEXT.txt').is_file()
    assert (ROOT/'rust/rapiddoc-inference/vendor/x86-simd-sort/LICENSE.md').is_file()


def test_corrupt_or_missing_notice_content_is_rejected():
    raw = (ROOT/'licenses/MIGRATION-NOTICES.txt').read_bytes()
    index = json.loads((ROOT/'licenses/migration-notices.json').read_text(encoding='utf8'))
    assert sync.validate(raw.replace(b'\n', b'\r\n'), index) == 120
    broken = bytearray(raw)
    broken[index['entries'][0]['start']] ^= 1
    with pytest.raises(ValueError, match='content changed'):
        sync.validate(broken, index)
    changed = copy.deepcopy(index)
    changed['entries'][0]['end'] = len(raw)+1
    with pytest.raises(ValueError, match='byte range'):
        sync.validate(raw, changed)
