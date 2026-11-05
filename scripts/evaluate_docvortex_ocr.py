"""Run the optional OCR/raster rule engine independently; no model fallback."""
import argparse
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',nargs='+',type=Path)
    parser.add_argument('--words-json',type=Path,help='Fixed OCR token list for one image; text must be unescaped')
    parser.add_argument('--ocr-model-dir',type=Path,default=ROOT/'rapid_doc/resources')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'tmp/docvortex-ocr-validation')
    parser.add_argument('--word-box',action='store_true',help='Use OCR word/character boxes instead of default text-line boxes')
    args = parser.parse_args()
    if args.words_json and len(args.images) != 1:
        parser.error('--words-json requires one image')
    from PIL import Image
    import numpy as np
    from rapid_doc.model.table.docvortex_ocr import recover_ocr_table,save_ocr_diagnostics
    ocr = None
    summaries = []
    for path in args.images:
        image = np.asarray(Image.open(path).convert('RGB'))
        start = time.perf_counter()
        if args.words_json:
            words = json.loads(args.words_json.read_text(encoding='utf-8'))
            boxes = [np.array(w['bbox']).reshape(-1,2) for w in words]
            texts = [w['text'] for w in words]
            scores = [w.get('score',1.) for w in words]
            source = 'fixed OCR; absent confidence uses 1.0 (not an OCR accuracy measurement)'
        else:
            if ocr is None:
                from rapidocr import RapidOCR
                files=[]
                for preferred,legacy in [('PP-OCRv6_det_small.onnx','det.onnx'),
                                         ('PP-OCRv6_rec_small.onnx','rec.onnx'),('ppocrv6_dict.txt','dict.txt')]:
                    model_file=args.ocr_model_dir/preferred
                    if not model_file.is_file():model_file=args.ocr_model_dir/legacy
                    if not model_file.is_file():
                        parser.error(f'Missing OCR file: {args.ocr_model_dir/preferred}; use --ocr-model-dir')
                    files.append(model_file)
                ocr = RapidOCR(params={'Global.use_cls':False,
                                      'Det.model_path':str(files[0]),
                                      'Rec.model_path':str(files[1]),
                                      'Rec.rec_keys_path':str(files[2]),
                                      'Det.mean':[.485,.456,.406],'Det.std':[.229,.224,.225],
                                      'EngineConfig.onnxruntime.intra_op_num_threads':4})
            output = ocr(image[:,:,::-1].copy(),return_word_box=args.word_box)
            from rapidocr.main import flatten_word_results
            word_items = flatten_word_results(output.word_results) if args.word_box else []
            if word_items:
                texts,scores,boxes = zip(*word_items)
                source = 'fresh OCR word boxes'
            else:
                boxes = output.boxes if output.boxes is not None else []
                texts,scores = output.txts or [],output.scores or []
                source = 'fresh OCR line boxes'
        elapsed_ocr = time.perf_counter()-start
        report = {}
        start = time.perf_counter()
        result = recover_ocr_table(image,(boxes,[html.escape(t) for t in texts],scores),diagnostics=report)
        elapsed_rules = time.perf_counter()-start
        folder = save_ocr_diagnostics(args.output_dir/path.stem,image,report)
        (folder/'ocr.json').write_text(json.dumps([{'bbox':np.asarray(b).tolist(),'text':t,'score':float(s)}
                                                 for b,t,s in zip(boxes,texts,scores)],ensure_ascii=False,indent=2),encoding='utf-8')
        if result:
            (folder/'result.html').write_text('<meta charset="utf-8"><style>td,th{border:1px solid #888;padding:4px}table{border-collapse:collapse}</style>'+result.html,encoding='utf-8')
        summary = {'image':str(path),'accepted':result is not None,'reason':report['reason'],
                   'ocr_source':source,'ocr_tokens':len(texts),'ocr_seconds':elapsed_ocr,
                   'rule_seconds':elapsed_rules,'artifacts':str(folder)}
        if result:
            summary.update(rows=result.rows,cols=result.cols,cells=len(result.cells),source=result.source,confidence=result.confidence)
        summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
