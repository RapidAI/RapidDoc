"""冻结实际旧SLANet matcher、逻辑点、raw HTML与CPU结构网络；不执行最终BeautifulSoup归一化。"""
import copy,json,math,random
import numpy as np
from PIL import Image
from common import ROOT,digest,write_json
from export_table_text import norm

def main():
    target=ROOT/'build/migration/legacy-table-match.json'
    if target.exists():raise FileExistsError(target)
    import rapid_doc.backend.pipeline.analyze_utils
    from rapid_doc.model.table.rapid_table_self.table_matcher.main import TableMatch
    from rapid_doc.model.table.rapid_table_self.utils.utils import format_ocr_results
    from rapid_doc.model.table.rapid_table_self.table_structure.pp_structure.pre_process import TablePreprocess
    from rapid_doc.model.table.rapid_table_self.table_structure.pp_structure.post_process import TableLabelDecode
    from rapid_doc.model.table.rapid_table_self.utils.typings import ModelType
    matcher=TableMatch();geometry=[];assembled=[];runs=[]
    def match(cells,dt):geometry.append(dict(request=dict(cell_bboxes=cells,dt_boxes=dt),expected=norm(matcher.match_result(cells,np.asarray(dt,np.float64)))))
    prior=json.loads((ROOT/'build/migration/table-match.json').read_text(encoding='utf-8'))
    for c in prior['cases']:match(c['cell_bboxes'],c['ocr_boxes'])
    for width in [math.nextafter(1.,0),1.,math.nextafter(1.,2),1.0001,.9999,1e-10]:
        match([[0,0,10000,10000],[0,0,10000,10000]],[[0,0,width,1],[0,0,1,0],[20,20,20,20]])
    rng=random.Random(184)
    for count in [1,2,5,33,257]:
        for _ in range(8):
            cells=[];dt=[]
            for i in range(count):
                x,y=rng.uniform(-100,500),rng.uniform(-100,500);w,h=rng.uniform(0,100),rng.uniform(0,100)
                cells.append([x,y,x+w,y+h]if i%2 else[x,y,x+w,y,x+w,y+h,x,y+h])
            for i in range(40):
                x,y=rng.uniform(-100,600),rng.uniform(-100,600);dt.append([x,y,x+rng.uniform(-5,50),y+rng.uniform(-5,50)])
            match(cells,dt)
    templates=[['<html>','<body>','<table>','<thead>','<tr>','<td></td>','<td></td>','</tr>','</thead>','</table>','</body>','</html>'],['<tr>','<td',' colspan="2"',' rowspan="2"','>','</td>','<td></td>','</tr>','<tr>','<td></td>','</tr>'],['<td>','</td>','<td></td>'],['<td',' colspan="0"','>','</td>','<td></td>'],['<td',' rowspan="-1"','>','</td>'],['<td>literal</td>','<tbody>','</tbody>','<td></td>']]
    contents=[[],[''],[' raw '],['<tbody>'],['<thead>'],['&amp;<escaped>'],['<b>A</b>','B'],['xx<b>A</b>','  B  ',''],['','<b>B</b>'],['\x1c 中 \u3000',' 文 ','$x$'],['<b>','</b>'],['Ａ','ß','🙂'],['a  b',' c d ']]
    for tags in templates:
        for text in contents:
            for mode in [0,1]:
                matched={0:list(range(len(text)))}if mode else{0:[],1:list(range(len(text)))}
                raw,pred=matcher.get_pred_html(tags,matched,[(s,1)for s in text]);logic=matcher.decode_one_logic_points(tags)
                assembled.append(dict(request=dict(tokens=tags,matched=matched,texts=text),expected=dict(raw_html=raw,pred=pred,logic_points=logic)))
    def process(structure,rows,w,h):
        dt,rec=format_ocr_results(copy.deepcopy(rows),h,w);dt,rec=matcher.filter_ocr_result(np.asarray(structure['cell_bboxes']),dt,rec)
        matched=matcher.match_result(structure['cell_bboxes'],dt);raw,pred=matcher.get_pred_html(structure['tokens'],matched,rec)
        return norm(dict(raw_html=raw,pred=pred,matched=matched,dt_boxes=dt.tolist(),logic_points=matcher.decode_one_logic_points(structure['tokens'])))
    def quad(b):return [[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
    for count in [1,2,5,20]:
        for eight in [False,True]:
            for _ in range(12):
                cs=[quad([i*20,10,i*20+20,40])for i in range(count)]
                cells=[sum(c,[])if eight else[c[0][0],c[0][1],c[2][0],c[2][1]]for c in cs]
                struct=dict(tokens=['<table>','<tbody>','<tr>']+['<td></td>']*count+['</tr>','</tbody>','</table>'],cell_bboxes=cells)
                qs=[quad([rng.uniform(-20,count*20+20),rng.uniform(-30,20),rng.uniform(-20,count*20+20),rng.uniform(20,80)])for _ in range(12)]
                rows=[qs,[contents[(i%12)+1][0]for i in range(12)],[.1]*12]
                req=dict(structure=struct,ocr_result=rows,width=max(100,count*20),height=100);runs.append(dict(request=req,expected=process(struct,rows,req['width'],100)))
    import onnxruntime as ort
    from legacy_table_models import create_full_models
    full=json.loads((ROOT/'build/migration/table-native-cpu.json').read_text(encoding='utf-8'));worker,_=create_full_models(full)
    models=json.loads((ROOT/'build/migration/table-structure.json').read_text(encoding='utf-8'))['models'];real=[];pipelines=[]
    pre=TablePreprocess();opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.log_severity_level=3
    for model in models:
        sess=ort.InferenceSession(str(ROOT/model['model']),sess_options=opts,providers=['CPUExecutionProvider'])
        kind=ModelType.SLANETPLUS if model['variant']=='slanet-plus'else ModelType.SLANET1M
        decoder=TableLabelDecode(sess.get_modelmeta().custom_metadata_map['character'].splitlines(),dict(model_type=kind))
        for old in model['png_cases']:
            path=ROOT/old['input'];rgb=np.asarray(Image.open(path).convert('RGB'));h,w=rgb.shape[:2];bgr=rgb[:,:,::-1].copy()
            tensors,shapes=pre([bgr]);pred=sess.run(None,{sess.get_inputs()[0].name:np.asarray(tensors)});ss,bb=decoder(pred[0],pred[1],shapes,[bgr])
            structure=dict(tokens=ss[0][0],confidence=float(ss[0][1]),cell_bboxes=bb[0].tolist())
            rows=[[quad([30,20,130,60]),quad([30,90,140,130]),quad([260,90,350,130]),quad([30,165,140,210]),quad([260,165,350,210])],['Header','中 文','123','Beta','456'],[1.]*5]
            common=dict(source=path.relative_to(ROOT).as_posix(),source_sha256=digest(path.read_bytes()),variant=model['variant'],model=model['model'],sha256=model['sha256'])
            real.append(dict(**common,structure=structure,ocr_result=rows,width=w,height=h,expected=process(structure,rows,w,h)))
            ocr=worker.ocr(bgr)[0]
            from rapid_doc.model.table.utils import normalize_table_ocr_text
            result=[list(v)for v in zip(*[[r[0],normalize_table_ocr_text(r[1][0]),r[1][1]]for r in ocr])]if ocr else[]
            pipelines.append(dict(**common,expected=process(structure,result,w,h)if result else None,structure=structure if result else None,ocr_result=norm(result)))
    refs=['rapid_doc/model/table/rapid_table_self/table_matcher/main.py','rapid_doc/model/table/rapid_table_self/utils/utils.py','rapid_doc/model/table/rapid_table_self/main.py']
    write_json(target,dict(geometry=geometry,assembled=assembled,runs=runs,real=real,pipelines=pipelines,sources={p:digest((ROOT/p).read_bytes())for p in refs},det_model=full['det_model'],det_sha256=full['det_sha256'],rec_model=full['rec_model'],rec_sha256=full['rec_sha256'],scope='Actual legacy TableMatch geometry/raw HTML/logic points and real SLANet-plus/1m CPU models. Supplied-OCR and real OCR composition listed separately; blank OCR skips network. No final BeautifulSoup/CJK normalization, UNET matching, dual-model selection or full parser parity. Flat 4/8-value cell boxes, finite coordinates, bounded integer spans; strict OCR column count is intentional vs old zip truncation.'))
    print(f'Frozen {len(geometry)} geometry/{len(assembled)} HTML+logic/{len(runs)} format→match; {len(real)} real structures/{len(pipelines)} CPU OCR compositions')
if __name__=='__main__':main()
