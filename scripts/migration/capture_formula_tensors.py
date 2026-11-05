"""Capture model inputs, then deliberately stop; this is not a PDF speed test."""
from pathlib import Path
import hashlib
import json
import os
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
mode = sys.argv[1]
assert mode in ('python', 'native')
namespace = sys.argv[2] if len(sys.argv)>2 else 'formula-tensor-audit-20261029'
assert Path(namespace).name == namespace
label = sys.argv[3] if len(sys.argv)>3 else mode
assert Path(label).name == label
directory = root/'build'/namespace/label
directory.mkdir(parents=True, exist_ok=False)
source = root/'demo/pdfs/03.pdf'
class CaptureComplete(Exception):
    pass

if mode == 'python':
    import numpy as np
    from rapid_doc.model.formula.rapid_formula_model import RapidFormulaModel
    def capture(self, images, batch_size):
        loaded = [self.latex_engine.load_img(image) for image in images]
        tensors = self.latex_engine.model_handler.model_processors.preprocess(loaded)
        assert len(tensors) == len(loaded)
        for index, (image, tensor) in enumerate(zip(loaded, tensors)):
            raw = np.asarray(tensor, dtype='<f4').tobytes()
            prefix = directory/f'{index:06}'
            prefix.with_suffix('.f32').write_bytes(raw)
            prefix.with_suffix('.input.json').write_text(json.dumps(dict(
                shape=list(tensor.shape), sha256=hashlib.sha256(raw).hexdigest(),
                sources=[dict(width=image.shape[1], height=image.shape[0],
                    rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    channel_reversed_sha256=hashlib.sha256(image[...,::-1].tobytes()).hexdigest())])), encoding='utf-8')
        raise CaptureComplete('formula tensor capture complete')
    RapidFormulaModel.batch_predict = capture
    import demo
    original_config = demo._build_config
    def config():
        values = list(original_config())
        values[1] = {**values[1], 'seal_enable':False}
        values[2] = {**values[2], 'model_dir_or_path':str(root/'rapid_doc/model/formula/rapid_formula_self/models/pp_formulanet_plus_m.onnx')}
        return tuple(values)
    demo._build_config = config
    call = lambda: demo.do_parse(str(directory/'unused-output'), ['03'], [source.read_bytes()],
        f_draw_layout_bbox=False, f_draw_span_bbox=False, f_dump_orig_pdf=False)
else:
    sys.path.insert(0, str(root/'build/api-dev-sdk'))
    import rapid_doc_rust as sdk
    from rapid_doc_rust._native import PdfEngine
    from rapid_doc_rust._runtime import geos_runtime
    os.environ['RAPIDDOC_DIAGNOSTIC_FORMULA_DIR'] = str(directory)
    data = source.read_bytes()
    library, _ = sdk._prepare_pdf(data, None, 256*1024*1024)
    host = sdk.RapidDoc(ocr_config={'seal_enable':False})
    models = host._get_models()
    def progress(event):
        if json.loads(event)['stage'] == 'ocr_detection':
            raise CaptureComplete('formula tensor capture complete')
    request = dict(docvortex_mode=host._mode, geos_library=geos_runtime(), formula_batch_size=1)
    engine = PdfEngine()
    call = lambda: engine.parse(library, data, json.dumps(request),
        lambda role:models[role].descriptor(), progress)
try:
    call()
except CaptureComplete:
    pass
else:
    raise AssertionError('Expected diagnostic stop before PDF export')
files = list(directory.glob('*.input.json'))
assert len(files) == 74, len(files)
print(f'{mode}: captured {len(files)} exact formula inputs; no PDF result produced', flush=True)
