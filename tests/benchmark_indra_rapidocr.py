"""Local CPU OCR comparison; PDF text is reference data, never OCR input."""
from pathlib import Path
import json
import time
import statistics
import unicodedata
import importlib.metadata
import argparse

import cv2
import fitz
import numpy as np
from rapidfuzz.distance import Levenshtein
import indra_ocr
from rapidocr import RapidOCR, EngineType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output-ocr-benchmark'


def norm(s):
    return ''.join(unicodedata.normalize('NFKC', s).split())


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-det', action='store_true', help='Match RapidDoc fixed detection resolution')
    args = parser.parse_args()
    if args.fixed_det:
        OUT = ROOT / 'output-ocr-benchmark-fixed'
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.ch_ppocr_det.utils import DetPreProcess
        TextDetector.get_preprocess = lambda self, max_wh: DetPreProcess(
            self.limit_side_len, self.limit_type, self.mean, self.std)
    OUT.mkdir(exist_ok=True)
    params = {
        'Global.use_cls': False,
        'Det.engine_type': EngineType.ONNXRUNTIME,
        'Rec.engine_type': EngineType.ONNXRUNTIME,
        'Det.model_path': str(ROOT / 'rapid_doc/resources/PP-OCRv6_det_small.onnx'),
        'Rec.model_path': str(ROOT / 'rapid_doc/resources/PP-OCRv6_rec_small.onnx'),
        'EngineConfig.onnxruntime.intra_op_num_threads': 4,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        'EngineConfig.onnxruntime.use_cuda': False,
        'Det.limit_side_len': 960,
        'Det.limit_type': 'max',
        'Det.std': [0.229, 0.224, 0.225],
        'Det.mean': [0.485, 0.456, 0.406],
        'Det.box_thresh': 0.5,
        'Det.use_dilation': True,
        'Det.unclip_ratio': 1.8,
    }
    start = time.perf_counter()
    # Converter stores the native OCR engine; avoid PDF auto/text paths entirely.
    from indra_ocr.pipeline import OrtOCR
    indra = OrtOCR(indra_ocr.model_dir(), threads=4, math=False, furniture=True)
    indra_load = time.perf_counter() - start
    start = time.perf_counter()
    rapid = RapidOCR(params=params)
    rapid_load = time.perf_counter() - start
    samples = [('03.pdf', 0), ('03.pdf', 1), ('small_ocr.pdf', 0),
               ('05.pdf', 0), ('demo1.pdf', 0), ('比亚迪财报.pdf', 2),
               ('49－长表格-密集型.pdf', 0), ('示例7-研究报告.pdf', 0)]
    pages = []
    crops = []
    for name, index in samples:
        with fitz.open(ROOT / 'demo/pdfs' / name) as doc:
            page = doc[index]
            pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append((name, index, gray, bgr))
            # 05.pdf has broken font mappings: retain speed sample, exclude accuracy.
            if name not in {'demo1.pdf', '比亚迪财报.pdf', '示例7-研究报告.pdf'}:
                continue
            lines = []
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    text = ''.join(s['text'] for s in line['spans'])
                    # Visually verified extraction error in this benchmark reference.
                    if name == 'demo1.pdf':
                        text = text.replace('q 2005 Elsevier', '© 2005 Elsevier')
                    if len(norm(text)) >= 12 and '\ufffd' not in text and line['dir'] == (1.0, 0.0):
                        lines.append((line['bbox'], text))
            for pos in np.linspace(0, len(lines)-1, min(20, len(lines)), dtype=int):
                box, text = lines[pos]
                x0,y0,x1,y1 = [v*200/72 for v in box]
                crop = bgr[max(0,int(y0)-3):min(bgr.shape[0],int(y1)+4),
                           max(0,int(x0)-4):min(bgr.shape[1],int(x1)+5)].copy()
                crops.append((name, text, crop))
    # Warm-up is excluded. Engines run serially and alternate order between repeats.
    indra(pages[0][2]); rapid(pages[0][3], use_cls=False)
    records = []
    for name,index,gray,bgr in pages:
        times = {'indra': [], 'rapidocr_small': []}
        outputs = {}
        for repeat in range(3):
            order = ['indra', 'rapidocr_small'] if repeat % 2 == 0 else ['rapidocr_small', 'indra']
            for engine in order:
                start = time.perf_counter()
                if engine == 'indra':
                    result = indra(gray)
                    outputs[engine] = result.text
                    outputs['indra_stages_ms'] = result.timings
                else:
                    result = rapid(bgr, use_cls=False)
                    outputs[engine] = '\n'.join(result.txts or [])
                    outputs['rapid_total_internal_s'] = result.elapse
                times[engine].append(time.perf_counter()-start)
        rec = {'file':name, 'page':index+1, 'image_size':[gray.shape[1],gray.shape[0]],
               'seconds':times, 'median_seconds':{k:statistics.median(v) for k,v in times.items()},
               'outputs':outputs}
        records.append(rec)
        print(name, index+1, rec['median_seconds'], flush=True)
        (OUT/'pages.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    accuracy = []
    for name,ref,bgr in crops:
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        h,w = gray.shape
        ipred = indra.recognize(gray,np.array([[0,0,w,h]],np.float32))[0][0]
        rresult = rapid(bgr,use_det=False,use_cls=False)
        rpred = ''.join(rresult.txts or [])
        row = {'file':name,'reference':ref,'indra':ipred,'rapidocr_small':rpred,
               'chars':len(norm(ref)), 'errors':{k:Levenshtein.distance(norm(ref),norm(v))
               for k,v in [('indra',ipred),('rapidocr_small',rpred)]}}
        accuracy.append(row)
    (OUT/'line_accuracy.json').write_text(json.dumps(accuracy,ensure_ascii=False,indent=2),encoding='utf-8')
    summary = {'versions':{p:importlib.metadata.version(p) for p in ['indra-ocr','rapidocr','onnxruntime']},
               'threads':4,'dpi':200,'fixed_detection_960':args.fixed_det,'repeats':3,'pages':len(pages),'reference_lines':len(crops),
               'reference_chars':sum(r['chars'] for r in accuracy),'math':False,
               'model_load_seconds':{'indra':indra_load,'rapidocr_small':rapid_load},
               'page_latency_seconds':{k:sum(r['median_seconds'][k] for r in records)/len(records)
               for k in ['indra','rapidocr_small']},
               'line_cer':{k:sum(r['errors'][k] for r in accuracy)/sum(r['chars'] for r in accuracy)
               for k in ['indra','rapidocr_small']}}
    summary['speedup'] = summary['page_latency_seconds']['rapidocr_small']/summary['page_latency_seconds']['indra']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
