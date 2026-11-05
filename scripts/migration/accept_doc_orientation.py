"""Compare whole-page orientation with original Python, in isolated processes.

--label is a controlled classifier response for testing all rotation branches;
without it the original orientation model is used. Original weights are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['python', 'rust'], required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', choices=['0', '90', '180', '270'])
    parser.add_argument('--disabled', action='store_true')
    parser.add_argument('--full-models', action='store_true', help='Also run the default formula/table/seal stages')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['USE_DOC_ORIENTATION_CLASSIFY'] = 'false' if args.disabled else 'true'
    os.environ['RAPIDDOC_PARITY_TRACE'] = str(args.output/'trace')
    labels = []
    if args.engine == 'python':
        from rapid_doc import RapidDoc
        from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
        original = RapidOrientationModel.predict

        def predict(self, image, det_res=None):
            label = args.label if args.label is not None else original(self, image, det_res)
            labels.append(label)
            return label

        RapidOrientationModel.predict = predict
    else:
        from rapid_doc_rust import RapidDoc
        if args.label is not None:
            import onnx
            from onnx import helper, TensorProto
            from rapid_doc_rust._runtime import LazySession
            path = args.output/'controlled-orientation.onnx'
            scores = [float(label == args.label) for label in ['0', '90', '180', '270']]
            nodes = [helper.make_node('ReduceMean', ['image'], ['mean'], keepdims=0),
                     helper.make_node('Mul', ['mean', 'zero'], ['ignored']),
                     helper.make_node('Add', ['ignored', 'fixed'], ['scores'])]
            graph = helper.make_graph(nodes, 'controlled-orientation',
                [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 224, 224])],
                [helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 4])],
                initializer=[helper.make_tensor('zero', TensorProto.FLOAT, [], [0.]),
                             helper.make_tensor('fixed', TensorProto.FLOAT, [1, 4], scores)])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
            helper.set_model_props(model, {'character': '0\n90\n180\n270'})
            onnx.save(model, path)
            descriptor = LazySession.descriptor

            def controlled(self):
                if self.role == 'orientation':
                    self.override = path
                return descriptor(self)

            LazySession.descriptor = controlled
    config = dict(engine_cfg=dict(use_cuda=True))
    engine = RapidDoc(parse_method='ocr', formula_enable=args.full_models, table_enable=args.full_models,
        layout_config=dict(config, docvortex_mode='off'),
        ocr_config={'EngineConfig.onnxruntime.use_cuda': True, 'seal_enable': args.full_models},
        table_config=config)
    started = time.perf_counter()
    result = engine(args.source)
    elapsed = time.perf_counter()-started
    from accept_docx_delivery import snapshot
    value = snapshot(result)
    (args.output/'snapshot.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    (args.output/'document.md').write_text(result.markdown, encoding='utf8')
    if args.engine == 'rust':
        for path in sorted((args.output/'trace').glob('page-*-model.json')):
            dets = json.loads(path.read_text())['layout_dets']
            labels.extend(sorted({item.get('rotate_label', '<disabled>') for item in dets}))
    report = dict(engine=args.engine, enabled=not args.disabled, controlled_label=args.label,
                  labels=labels, seconds=elapsed, markdown_characters=len(result.markdown))
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
