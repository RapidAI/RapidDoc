import cv2
import numpy as np
import pytest
from unittest.mock import Mock

pytest.importorskip('img2table')

from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self import ModelType


def ruled_table():
    image = np.full((300, 500, 3), 255, dtype=np.uint8)
    boxes, texts, scores = [], [], []
    for y in (20, 100, 180, 260):
        cv2.line(image, (20, y), (480, y), (0, 0, 0), 2)
    for x in (20, 250, 480):
        cv2.line(image, (x, 20), (x, 260), (0, 0, 0), 2)
    for row in range(3):
        for col in range(2):
            x, y = 40 + col * 230, 65 + row * 80
            text = f'R{row}C{col}'
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
            boxes.append([[x, y-20], [x+90, y-20], [x+90, y+3], [x, y+3]])
            texts.append(text)
            scores.append(.99)
    return image, [boxes, texts, scores]


def test_real_extraction_reuses_ocr_and_preserves_escaping():
    image, rows = ruled_table()
    rows[1][0] = 'A &amp; B &lt; C'
    model = RapidTableModel(None, {'model_type': 'img2table'})
    html = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), rows)
    assert '<table>' in html
    assert html.count('<tr>') == 3
    assert html.count('<td') == 6
    assert 'A &amp; B &lt; C' in html
    assert '&amp;amp;' not in html
    assert 'R2C1' in html
    assert model.rule_table_class(image) == (None, 1.0)


def test_empty_detection_and_enum_configuration():
    model = RapidTableModel(None, {'model_type': ModelType.IMG2TABLE})
    assert model._structure(np.full((100, 100, 3), 255, dtype=np.uint8), [[], [], []]) is None


def test_img2table_as_fallback_reuses_existing_ocr():
    image, rows = ruled_table()
    model = RapidTableModel.__new__(RapidTableModel)
    model.ocr_engine = None
    model.model_type = ModelType.TABLE_TRANSFORMER
    model.fallback_model_type = 'img2table'
    model.fallback_to_text = False
    model.fallback_text_loss_threshold = .2
    model._fallback_model = None
    model._fallback_config = {'fallback_model_type': 'img2table'}
    model._structure = Mock(return_value='<table><tr><td>R0C0</td></tr></table>')
    assert 'R0C0' in model.predict(image, rows)
    assert model._fallback_model.model_type == ModelType.IMG2TABLE
    assert 'R2C1' in model._fallback_model._structure(image, rows)


def test_real_merged_header():
    image, rows = ruled_table()
    # Remove the first-row internal separator to create a two-column header.
    cv2.line(image, (250, 22), (250, 98), (255, 255, 255), 5)
    model = RapidTableModel(None, {'model_type': 'img2table'})
    html = model._structure(image, rows)
    assert 'colspan="2"' in html
    assert 'R0C0' in html and 'R0C1' in html and 'R2C1' in html
