"""CPU comparison on known synthetic formula crops, including Chinese text."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import json
import time
import statistics
import argparse
import numpy as np
import cv2
import fitz
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from matplotlib.mathtext import math_to_image
from indra_ocr.models import model_dir
from indra_ocr.math import MathReader
from rapid_doc.model.formula.rapid_formula_self import RapidFormula, RapidFormulaInput, ModelType, EngineType

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output-formula-benchmark'
SAMPLES = [
    ('energy', '', r'E=mc^{2}'),
    ('fraction', '', r'\frac{a+b}{c+d}'),
    ('root', '', r'\sqrt{x^{2}+y^{2}}'),
    ('quadratic', '', r'x=\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}'),
    ('sum', '', r'\sum_{i=1}^{n}i=\frac{n(n+1)}{2}'),
    ('integral', '', r'\int_{0}^{1}x^{2}dx=\frac{1}{3}'),
    ('derivative', '', r'\frac{d}{dx}\sin x=\cos x'),
    ('trig', '', r'\sin^{2}x+\cos^{2}x=1'),
    ('zh_area', '面积', r'=\pi r^{2}'),
    ('zh_average', '平均值', r'=\frac{x_{1}+x_{2}+x_{3}}{3}'),
    ('zh_variance', '方差', r'=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\mu)^{2}'),
    ('zh_speed', '速度', r'=\frac{路程}{时间}'),
]


def render_samples():
    OUT.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/simsun.ttc', 44)
    rows = []
    for name,zh,formula in SAMPLES:
        if name == 'zh_speed':
            img = Image.new('RGB',(370,125),'white')
            draw = ImageDraw.Draw(img)
            draw.text((12,40),'速度 =',font=font,fill='black')
            draw.text((220,0),'路程',font=font,fill='black')
            draw.line((210,62,320,62),fill='black',width=2)
            draw.text((220,66),'时间',font=font,fill='black')
            zh = '速度路程时间'
            reference = r'\text{速度}=\frac{\text{路程}}{\text{时间}}'
        else:
            buf = io.BytesIO()
            math_to_image('$'+formula+'$',buf,dpi=160,format='png',color='black')
            m = Image.open(buf).convert('RGB')
            left = int(font.getlength(zh))+8 if zh else 0
            height = max(m.height,54)+24
            img = Image.new('RGB',(m.width+left+24,height),'white')
            img.paste(m,(left+12,(height-m.height)//2))
            if zh:
                draw = ImageDraw.Draw(img)
                draw.text((12,(height-44)//2-6),zh,font=font,fill='black')
            reference = (r'\text{'+zh+'}' if zh else '')+formula
        path = OUT/(name+'.png')
        img.save(path)
        rows.append({'id':name,'chinese':zh,'reference':reference,'path':str(path)})
    with fitz.open(ROOT/'demo/pdfs/03.pdf') as doc:
        page=doc[1]
        # Visually transcribed equation (2-1), excluding the equation number.
        clip=fitz.Rect(page.rect.width*300/715,page.rect.height*120/1055,
                       page.rect.width*433/715,page.rect.height*168/1055)
        path=OUT/'real_vehicle.png'
        page.get_pixmap(matrix=fitz.Matrix(200/72,200/72),clip=clip,alpha=False).save(path)
        rows.append({'id':'real_vehicle','chinese':'',
            'reference':r'\frac{dV}{dt}=\frac{\sum F_{t}-\sum F_{r}}{\delta M}', 'path':str(path)})
    (OUT/'samples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('engine',choices=['indra','S','M'])
    parser.add_argument('--real-only',action='store_true')
    args = parser.parse_args()
    rows = render_samples()
    if args.real_only:
        rows=[r for r in rows if r['id']=='real_vehicle']
    start = time.perf_counter()
    if args.engine == 'indra':
        folder = model_dir(math=True)
        so = ort.SessionOptions()
        so.intra_op_num_threads=4
        so.inter_op_num_threads=1
        so.enable_cpu_mem_arena=False
        engine = MathReader(folder,so,['CPUExecutionProvider'])
        def predict(img):
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            yy,xx = np.where(gray < 200)
            box=[xx.min(),yy.min(),xx.max()+1,yy.max()+1]
            return engine.read(gray,np.array([box],np.float32))[0]
        providers=engine.enc.get_providers()
        cjk_tokens=[t for t in engine.itos if any('\u3400'<=c<='\u9fff' for c in t)]
    else:
        size = args.engine.lower()
        path = Path('D:/CodeProjects/doc/RapidAI/models')/f'pp_formulanet_plus_{size}.onnx'
        cfg = RapidFormulaInput(model_type=getattr(ModelType,f'PP_FORMULANET_PLUS_{args.engine}'),
            model_dir_or_path=path,engine_type=EngineType.ONNXRUNTIME,
            engine_cfg={'intra_op_num_threads':4,'inter_op_num_threads':1,'use_cuda':False,'onnx_ops':False})
        engine=RapidFormula(cfg)
        def predict(img):
            return engine([cv2.cvtColor(img,cv2.COLOR_RGB2BGR)])[0].rec_formula, None
        providers=engine.session.session.get_providers()
        cjk_tokens=None
    load_seconds=time.perf_counter()-start
    imgs=[np.array(Image.open(r['path']).convert('RGB')) for r in rows]
    predict(imgs[0])
    result={'engine':args.engine,'threads':4,'providers':providers,'load_seconds':load_seconds,
            'repeats':3,'chinese_vocab_tokens':len(cjk_tokens) if cjk_tokens is not None else None,'samples':[]}
    if args.real_only:
        previous=OUT/(args.engine+'.json')
        if previous.exists():
            result=json.loads(previous.read_text(encoding='utf-8'))
            result['samples']=[r for r in result['samples'] if r['id']!='real_vehicle']
    for row,img in zip(rows,imgs):
        times=[]
        predictions=[]
        confidence=None
        for repeat in range(3):
            start=time.perf_counter()
            pred,confidence=predict(img)
            times.append(time.perf_counter()-start)
            predictions.append(pred)
        entry={**row,'predictions':predictions,'confidence':confidence,'seconds':times,'median_seconds':statistics.median(times)}
        result['samples'].append(entry)
        (OUT/(args.engine+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(args.engine,row['id'],round(entry['median_seconds'],3),pred,confidence,flush=True)
    result['mean_sample_median_seconds']=statistics.mean(r['median_seconds'] for r in result['samples'])
    (OUT/(args.engine+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',args.engine,result['mean_sample_median_seconds'],flush=True)


if __name__=='__main__':
    main()
