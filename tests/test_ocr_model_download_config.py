import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import rapidocr
import rapidocr.models
from omegaconf import OmegaConf

from rapid_doc.utils import models_download_utils


@pytest.fixture(params=['project', 'docker'])
def downloader(request, monkeypatch):
    if request.param == 'project':
        return models_download_utils
    path = Path(__file__).parents[1] / 'docker/models_download_utils.py'
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location('docker_model_download_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('aliases', [False, True])
def test_model_download_accepts_old_and_new_metadata(downloader, monkeypatch, tmp_path, aliases):
    model = {'model_dir': 'https://example.test/model.onnx', 'SHA256': 'model-hash',
             'dict_url': 'https://example.test/dict.txt', 'license': 'Apache-2.0'}
    version = {'det': {'ch': model}, 'cls': {'ch': {
        'model_dir': 'https://example.test/cls.onnx', 'SHA256': 'cls-hash'}}}
    if aliases:
        # 3.10.0 adds strings alongside det/rec/cls, plus nested model routing.
        version['aliases'] = {'zh': 'ch', 'ja': 'japan'}
    config = {'model_routes': {'PP-OCRv6': {'aliases': {'zh': 'ch'}}},
              'openvino': {'PP-OCRv6': version},
              'fonts': {'ch': {'path': 'https://example.test/font.ttf', 'SHA256': 'font-hash'}}}
    monkeypatch.setattr(downloader, 'read_yaml', lambda _: OmegaConf.create(config))
    calls = []
    monkeypatch.setattr(downloader.DownloadFile, 'run', lambda item: calls.append(item))
    downloader.ocr_download(tmp_path, None, SimpleNamespace(__file__=str(tmp_path/'__init__.py')))
    assert {c.file_url for c in calls} == {
        'https://example.test/model.onnx', 'https://example.test/cls.onnx',
        'https://example.test/dict.txt', 'https://example.test/font.ttf'}
    assert next(c for c in calls if c.file_url.endswith('model.onnx')).sha256 == 'model-hash'
    assert all(c.save_path.parent == tmp_path for c in calls)


def test_real_installed_rapidocr_download_manifest(downloader, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(downloader.DownloadFile, 'run', lambda item: calls.append(item))
    downloader.ocr_download(tmp_path, rapidocr.models, rapidocr)
    assert calls
    assert any(c.file_url.endswith('ch_ppocr_mobile_v2.0_cls_mobile.onnx') for c in calls)
    assert all(c.file_url.startswith('https://') for c in calls)
    assert not any(c.save_path.name in {'ch', 'japan', 'chinese_cht'} for c in calls)


def test_complete_download_entry_accepts_current_configs(downloader, monkeypatch, tmp_path):
    monkeypatch.setenv('RAPID_MODELS_DIR', str(tmp_path))
    calls = []
    monkeypatch.setattr(downloader.DownloadFile, 'run', lambda item: calls.append(item))
    assert downloader.download_pipeline_models() is True
    assert any(c.file_url.endswith('ch_ppocr_mobile_v2.0_cls_mobile.onnx') for c in calls)
    assert any(c.file_url.endswith('unet.onnx') for c in calls)
