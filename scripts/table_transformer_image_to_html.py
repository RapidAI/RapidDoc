"""Run TATR ONNX on a single table crop, with existing OCR or local PP-OCRv6."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output-table/table-transformer')
    parser.add_argument('--words-json', type=Path, help='Fixed OCR tokens with input-coordinate bbox/text fields')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from rapid_doc.model.table.rapid_table_self import RapidTable, RapidTableInput, ModelType
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    if args.words_json:
        tokens = json.loads(args.words_json.read_text(encoding='utf-8-sig'))
        boxes = [[[t['bbox'][0],t['bbox'][1]],[t['bbox'][2],t['bbox'][1]],
                  [t['bbox'][2],t['bbox'][3]],[t['bbox'][0],t['bbox'][3]]] for t in tokens]
        texts, scores = [t['text'] for t in tokens], [1.0]*len(tokens)
    else:
        from rapidocr import RapidOCR
        resources = ROOT/'rapid_doc/resources'
        files = [resources/'PP-OCRv6_det_small.onnx',resources/'PP-OCRv6_rec_small.onnx',resources/'ppocrv6_dict.txt']
        for path in files:
            if not path.is_file():
                parser.error(f'Missing local OCR file: {path}; alternatively pass --words-json')
        ocr = RapidOCR(params={'Global.use_cls':False,'Det.model_path':str(files[0]),
                              'Rec.model_path':str(files[1]),'Rec.rec_keys_path':str(files[2]),
                              'Det.mean':[.485,.456,.406],'Det.std':[.229,.224,.225],
                              'EngineConfig.onnxruntime.intra_op_num_threads':args.threads})
        recognized = ocr(image)
        boxes = recognized.boxes if recognized.boxes is not None else []
        texts = list(recognized.txts) if recognized.txts is not None else []
        scores = list(recognized.scores) if recognized.scores is not None else [1.0]*len(texts)
        tokens = []
        for idx,(box,text) in enumerate(zip(boxes,texts)):
            points = np.asarray(box)
            tokens.append(dict(bbox=[*points.min(axis=0).tolist(),*points.max(axis=0).tolist()],text=str(text),
                               block_num=0,line_num=idx,span_num=0))
    engine = RapidTable(RapidTableInput(model_type=ModelType.TABLE_TRANSFORMER, use_ocr=False,
                                       model_dir_or_path=args.model,engine_cfg={'intra_op_num_threads':args.threads}))
    start = time.perf_counter()
    result = engine(image,ocr_results=[(boxes,texts,scores)])
    elapsed = time.perf_counter()-start
    args.output_dir.mkdir(parents=True,exist_ok=True)
    stem = args.image.stem
    html_path = args.output_dir/f'{stem}.html'
    html_path.write_text('<!doctype html><meta charset="utf-8"><style>table{border-collapse:collapse}td,th{border:1px solid #888;padding:5px}</style>'+result.pred_htmls[0],encoding='utf-8')
    canvas = Image.fromarray(image[:,:,::-1])
    draw = ImageDraw.Draw(canvas)
    for box in result.cell_bboxes[0]:
        draw.rectangle(box.tolist(),outline='red',width=1)
    canvas.save(args.output_dir/f'{stem}.cells.png')
    (args.output_dir/f'{stem}.words.json').write_text(json.dumps(tokens,ensure_ascii=False,indent=2),encoding='utf-8')
    summary = dict(image=str(args.image.resolve()),html=str(html_path.resolve()),
                   cells=result.cell_bboxes[0].tolist(),logic=result.logic_points[0].tolist(),
                   inference_and_postprocess_seconds=elapsed)
    (args.output_dir/f'{stem}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(html=str(html_path.resolve()),cells=len(result.cell_bboxes[0]),seconds=elapsed),ensure_ascii=False))


if __name__ == '__main__':
    main()
