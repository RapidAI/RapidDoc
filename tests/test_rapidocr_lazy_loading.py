from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from rapid_doc.model.ocr import rapid_ocr


@pytest.mark.parametrize('lazy', [False, True])
@pytest.mark.parametrize('ops', [False, True])
def test_adapters_are_loaded_before_session_installation(monkeypatch, lazy, ops):
    detector = SimpleNamespace(session=SimpleNamespace(session=Mock()))
    recognizer = SimpleNamespace(session=SimpleNamespace(session=Mock()), rec_batch_num=6)
    for adapter in (detector, recognizer):
        adapter.session.session.get_providers.return_value = ['CPUExecutionProvider']
    engine = SimpleNamespace(
        cfg=OmegaConf.create({'Det': {'engine_type': 'onnxruntime', 'session': None},
                              'Rec': {'engine_type': 'onnxruntime', 'session': None}}),
        text_det=None if lazy else detector, text_rec=None if lazy else recognizer)
    if lazy:
        engine._load_det_model = Mock(side_effect=lambda: setattr(engine, 'text_det', detector))
        engine._load_rec_model = Mock(side_effect=lambda: setattr(engine, 'text_rec', recognizer))
    monkeypatch.setattr(rapid_ocr, 'RapidOCR', lambda **kwargs: engine)
    monkeypatch.setattr(rapid_ocr, 'get_device', lambda: 'cpu')
    monkeypatch.setattr(rapid_ocr, 'check_openvino', lambda: False)
    install = rapid_ocr.install_onnx_ops
    def checked_install(value, settings):
        assert value.text_det is detector and value.text_rec is recognizer
        return install(value, settings)
    monkeypatch.setattr(rapid_ocr, 'install_onnx_ops', checked_install)
    model = rapid_ocr.RapidOcrModel(ocr_config={'onnx_ops': ops})
    assert model.text_detector is detector
    assert model.text_recognizer is recognizer
    assert model.rec_batch_num == 6
    assert model.onnx_ops_reports == {}
    if lazy:
        engine._load_det_model.assert_called_once_with()
        engine._load_rec_model.assert_called_once_with()


def test_model_loading_failure_is_not_silently_skipped(monkeypatch):
    engine = SimpleNamespace(text_det=None, text_rec=None,
                             _load_det_model=Mock(side_effect=RuntimeError('invalid model')))
    monkeypatch.setattr(rapid_ocr, 'RapidOCR', lambda **kwargs: engine)
    monkeypatch.setattr(rapid_ocr, 'get_device', lambda: 'cpu')
    monkeypatch.setattr(rapid_ocr, 'check_openvino', lambda: False)
    install = Mock()
    monkeypatch.setattr(rapid_ocr, 'install_onnx_ops', install)
    with pytest.raises(RuntimeError, match='invalid model'):
        rapid_ocr.RapidOcrModel()
    install.assert_not_called()
