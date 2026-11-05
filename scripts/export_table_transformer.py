"""Export pinned Microsoft TATR v1.1-all and validate PyTorch / ORT / upstream cells."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
REPO = "microsoft/table-transformer-structure-recognition-v1.1-all"
REVISION = "7587a7ef111d9dcbf8ac695f1376ab7014340a0c"
SOURCE_REVISION = "16d124f616109746b7785f03085100f1f6247575"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, default=ROOT/'tmp/tatr-weights')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'tmp/tatr-source')
    parser.add_argument('--output', type=Path, default=ROOT/'rapid_doc/model/table/rapid_table_self/models/table_transformer_v1_1_all.onnx')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true', help='Validate an existing ONNX without exporting or replacing it')
    parser.add_argument('--images', nargs='+', type=Path, default=[
        ROOT/'tests/fixtures/table_transformer/basic.png', ROOT/'tests/fixtures/table_transformer/complex.png',
        ROOT/'tests/fixtures/table_transformer/dense.png'])
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import requests
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from torchvision import transforms
    from PIL import Image
    from fitz import Rect
    from collections import defaultdict
    from transformers import TableTransformerForObjectDetection
    from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer.main import (
        preprocess, decode, outputs_to_objects, cells_to_result, THRESHOLDS)

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    # Verify the downloaded checkpoint against HF's LFS SHA, not just a local hash.
    metadata = requests.get(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60)
    metadata.raise_for_status()
    files = {item['rfilename']:item for item in metadata.json()['siblings']}
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        path = args.weights_dir/name
        expected = files[name].get('lfs', {}).get('sha256')
        if path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {path}')
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.part')
            with requests.get(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', stream=True, timeout=(30,120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(1024*1024):
                        output.write(chunk)
            if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Checkpoint checksum mismatch: {temporary}')
            temporary.replace(path)
    # Fetch exact upstream files for an independent postprocessing reference.
    sources = {}
    for name in ('src/postprocess.py', 'src/inference.py', 'LICENSE'):
        response = requests.get(f'https://raw.githubusercontent.com/microsoft/table-transformer/{SOURCE_REVISION}/{name}', timeout=60)
        response.raise_for_status()
        sources[name] = response.text
        (args.source_dir/Path(name).name).write_text(response.text, encoding='utf-8')
    upstream_pp = types.ModuleType('tatr_reference_postprocess')
    exec(compile(sources['src/postprocess.py'], 'upstream/postprocess.py', 'exec'), upstream_pp.__dict__)
    namespace = dict(postprocess=upstream_pp, Rect=Rect, defaultdict=defaultdict)
    selected = {'iob','align_headers','refine_table_structure','objects_to_structures','structure_to_cells','MaxResize'}
    tree = ast.parse(sources['src/inference.py'])
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected:
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'upstream/inference.py', 'exec'), namespace)
    reference_transform = transforms.Compose([namespace['MaxResize'](1000), transforms.ToTensor(),
                                               transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    torch.set_num_threads(args.threads)
    model = TableTransformerForObjectDetection.from_pretrained(str(args.weights_dir), local_files_only=True,
                                                               attn_implementation='eager').eval()
    assert model.config.num_queries == 125 and model.config.num_labels == 6

    class Export(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, images):
            result = self.inner(pixel_values=images, pixel_mask=torch.ones_like(images[:, 0], dtype=torch.long))
            return result.logits, result.pred_boxes

    wrapper = Export(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output if args.verify_only else args.output.with_suffix('.unverified.onnx')
    first = cv2.imread(str(args.images[0]))
    if first is None:
        raise FileNotFoundError(args.images[0])
    pixels = torch.from_numpy(preprocess(first))
    print('Verifying existing TATR ONNX...' if args.verify_only else
          'Exporting TATR v1.1-all, FP32, opset17, dynamic H/W...', flush=True)
    if not args.verify_only:
        with torch.no_grad():
            torch.onnx.export(wrapper, (pixels,), str(temporary), input_names=['images'], output_names=['logits','pred_boxes'],
                              dynamic_axes={'images':{2:'height',3:'width'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(str(temporary))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=['CPUExecutionProvider'])
    validation = []
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        pixels = preprocess(image)
        reference_pixels = reference_transform(Image.fromarray(image[:,:,::-1])).numpy()[None]
        np.testing.assert_array_equal(pixels, reference_pixels)
        with torch.no_grad():
            result = model(pixel_values=torch.from_numpy(pixels))
        exported = session.run(None, {'images':pixels})
        errors = []
        for actual, expected in zip(exported, [result.logits.numpy(), result.pred_boxes.numpy()]):
            np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=3e-4)
            errors.append(float(np.max(np.abs(actual-expected))))
        size = (image.shape[1],image.shape[0])
        # Same fixed OCR coordinates for both paths; no runtime OCR difference.
        words_path = path.with_suffix('.words.json')
        tokens = json.loads(words_path.read_text(encoding='utf-8')) if words_path.exists() else []
        objects = outputs_to_objects(result.logits.numpy(),result.pred_boxes.numpy(),size)
        structures = namespace['objects_to_structures'](deepcopy(objects),deepcopy(tokens),THRESHOLDS)
        reference_cells = [namespace['structure_to_cells'](s,deepcopy(tokens))[0]
                           for s in structures if s['rows'] and s['columns']]
        actual = decode(*exported,size,tokens)
        if reference_cells:
            assert any(cells_to_result(cells,size)[0] == actual[0] for cells in reference_cells), 'Upstream cells differ'
        else:
            assert len(actual[1]) == 0
        validation.append(dict(image=str(path),max_errors=errors,preprocess_exact=True,
                               upstream_cells_equal=True,cells=len(actual[1])))
        print(validation[-1],flush=True)
    report = dict(model=REPO,revision=REVISION,source_revision=SOURCE_REVISION,
                  weights_sha256=hashlib.sha256((args.weights_dir/'model.safetensors').read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),validation=validation,
                  torch=torch.__version__,onnxruntime=ort.__version__)
    # Release before publishing so Windows permits replacing an existing file.
    del session
    if not args.verify_only:
        temporary.replace(args.output)
        args.output.with_suffix('.LICENSE').write_text(sources['LICENSE'],encoding='utf-8')
    report_path = args.output.with_suffix('.verification.json' if args.verify_only else '.validation.json')
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Validated model: {args.output} ({args.output.stat().st_size} bytes)',flush=True)


if __name__ == '__main__':
    main()
