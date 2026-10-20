import numpy as np
import pytest

from rapid_doc.model.ocr.ocr_patch import TextDetector, patch_text_detector


@pytest.mark.parametrize('limit_type', ['min', 'max'])
@pytest.mark.parametrize('call_style', ['no_argument', 'positional', 'keyword'])
def test_detector_preprocessing_accepts_old_and_new_call_signatures(monkeypatch, limit_type, call_style):
    monkeypatch.setattr(TextDetector, 'get_preprocess', TextDetector.get_preprocess)
    patch_text_detector()
    detector = TextDetector.__new__(TextDetector)
    detector.limit_side_len = 64
    detector.limit_type = limit_type
    detector.mean = [0.485, 0.456, 0.406]
    detector.std = [0.229, 0.224, 0.225]
    if call_style == 'no_argument':
        preprocess = detector.get_preprocess()
    elif call_style == 'positional':
        preprocess = detector.get_preprocess(80)
    else:
        preprocess = detector.get_preprocess(max_wh=80)
    image = np.full((40, 80, 3), 128, dtype=np.uint8)
    output = preprocess(image)
    assert output.shape == ((1, 3, 64, 128) if limit_type == 'min' else (1, 3, 32, 64))
    assert np.isfinite(output).all()
