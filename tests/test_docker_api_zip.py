import importlib.util
import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch):
    path = Path(__file__).parents[1] / 'docker/app.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('zip_api_regression', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'has_vl_env', lambda: False)
    return module


@pytest.mark.parametrize('fields,expected_md,expected_images', [
    ({}, True, True),
    ({'return_md': 'false'}, False, True),
    ({'return_images': 'false'}, True, False),
    ({'return_md': 'true', 'return_images': 'true'}, True, True),
])
def test_zip_defaults_preserve_results_and_explicit_false(api, monkeypatch, tmp_path, fields, expected_md, expected_images):
    seen = []
    async def parse(**kwargs):
        seen.append(kwargs)
        directory = Path(kwargs['output_dir']) / kwargs['pdf_file_names'][0] / kwargs['parse_method']
        (directory/'images').mkdir(parents=True)
        if kwargs['f_dump_md']:
            (directory/'论文.md').write_text('Parsed document', encoding='utf-8')
        (directory/'images'/'figure.png').write_bytes(b'PNG test data')
    monkeypatch.setattr(api, 'aio_do_parse', parse)
    with TestClient(api.app) as client:
        response = client.post('/file_parse', files={'files': ('论文.pdf', b'%PDF-1.4', 'application/pdf')},
                               data={'response_format_zip': 'true', 'output_dir': str(tmp_path), **fields})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [name.replace('\\', '/') for name in archive.namelist()]
        assert ('论文/论文.md' in names) is expected_md
        assert ('论文/images/figure.png' in names) is expected_images
        if expected_md:
            assert archive.read(archive.namelist()[names.index('论文/论文.md')]).decode('utf-8') == 'Parsed document'
    assert seen[0]['formula_enable'] is True
    assert seen[0]['table_enable'] is True
    assert not Path(seen[0]['output_dir']).exists()
