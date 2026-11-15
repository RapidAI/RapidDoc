"""img2table 2.x structure extraction using RapidDoc's existing OCR results."""
from html import unescape

import cv2


class Img2TableBackend:
    def __init__(self, options=None):
        try:
            from img2table.document import Image
            from img2table.ocr._types import OCRData
        except ImportError as exc:
            raise ImportError('img2table 2.x requires: pip install "rapid-doc[img2table]"') from exc
        self.image_type = Image
        self.ocr_data_type = OCRData
        self.options = {
            'implicit_rows': False,
            'implicit_columns': False,
            'borderless_tables': True,
            'min_confidence': 50,
            **(options or {}),
        }

    def predict(self, bgr_image, ocr_result):
        # Lossless in-memory encoding uses the upstream public Image API. Keep
        # rotation disabled because supplied OCR uses the current crop coordinates.
        ok, encoded = cv2.imencode('.png', bgr_image)
        if not ok:
            raise ValueError('Could not encode table image')
        records = []
        for index, (box, text, score) in enumerate(zip(*ocr_result)):
            records.append({
                'id': f'word_{index}', 'parent': f'word_{index}',
                'value': unescape(str(text)), 'confidence': int(round(float(score) * 100)),
                'x1': int(round(min(p[0] for p in box))),
                'y1': int(round(min(p[1] for p in box))),
                'x2': int(round(max(p[0] for p in box))),
                'y2': int(round(max(p[1] for p in box))),
            })
        document = self.image_type(src=encoded.tobytes(), detect_rotation=False,
                                   ocr_data=self.ocr_data_type(records={0: records}))
        tables = document.extract_tables(**self.options)
        if not tables:
            return None
        # A layout crop represents one table. Avoid concatenating unrelated
        # tables into HTML that the downstream single-table contract cannot use.
        table = max(tables, key=lambda t: (t.bbox.x2 - t.bbox.x1) * (t.bbox.y2 - t.bbox.y1))
        return table.html
