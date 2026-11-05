import os
from pathlib import Path
from rapid_doc.model.orientation.rapid_orientation import RapidOrientation


class RapidOrientationModel:
    def __init__(self):
        rapid_doc_dir = Path(os.path.abspath(__file__)).parent.parent.parent
        self.orientation_engine = RapidOrientation(model_path=str(rapid_doc_dir / 'resources' / 'rapid_orientation.onnx'))

    def predict(self, input_img, det_res=None, *, table_mode=False):
        height, width = input_img.shape[:2]
        if width <= 0 or height / width <= 1.2:
            return '0'
        # Page classification retains its existing model decision. Table mode
        # no longer relies on tall text boxes: single digits are often square.
        if not table_mode:
            return self.orientation_engine(input_img)[0]
        label, score, margin, _ = self.orientation_engine.predict_scores(input_img)
        return label if label in ('90', '270') and score >= .9 and margin >= .2 else '0'
