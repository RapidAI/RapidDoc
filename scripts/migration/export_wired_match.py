"""UNET 专用 OCR 几何、文本分割/归并与边缘裁剪的 Python 合同。"""
import copy,json,re
import numpy as np
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_recover import match_ocr_cell,sorted_ocr_boxes,gather_ocr_list_by_row,plot_html_table
from common import ROOT,write_json

def ccw(b):x,y,r,s=b;return [[x,y],[x,s],[r,s],[r,y]]
def cw(b):x,y,r,s=b;return [[x,y],[r,y],[r,s],[x,s]]
def capture(name,polygons,ocr):
    original=[[line['points'],line['text'],line['confidence'],i]for i,line in enumerate(ocr)]
    matched,unmatched=match_ocr_cell(copy.deepcopy(original),np.array(polygons,np.float64).reshape(-1,4,2))
    cells=[];cell_lines=[]
    for i in range(len(polygons)):
        fragments=[];values=matched.get(i,[])
        for points,text,confidence,index in values:
            fragments.append(dict(points=np.array(points).tolist(),text=text,confidence=confidence,ocr_index=index,split=text!=ocr[index]['text']or np.array(points).tolist()!=ocr[index]['points']))
        cells.append(fragments)
        text_boxes=[[[float(v[0][0][0]),float(v[0][0][1]),float(v[0][2][0]),float(v[0][2][1])],v[1]]for v in values]
        _,indices=sorted_ocr_boxes([v[0]for v in text_boxes],threhold=.3)
        gathered=gather_ocr_list_by_row([copy.deepcopy(text_boxes[j])for j in indices],threhold=.3)
        cell_lines.append([re.sub(r'[\r\n]+','',v[1]).strip()for v in gathered if v[1]and v[1].strip()])
    # 预览保留原 OCR 索引；旧 empty-pred 早退没有返回 unmatched，不能据此静默丢文字。
    unmatched=[v[3]for v in unmatched]if polygons else list(range(len(ocr)))
    return dict(id=name,polygons=polygons,ocr=ocr,expected=dict(cells=cells,cell_lines=cell_lines,unmatched=unmatched))

def main():
    def ocr(b,text):return dict(points=cw(b),text=text,confidence=.9)
    cases=[];two=[ccw([0,0,100,80]),ccw([100,0,200,80])]
    for text in ['123 456','1,234.56 12.5%','2024/01/02 2023-12-31','甲乙丙丁戊己庚辛壬癸','ABCDEFGHIJ','甲乙 12%','Ⅻ²αβ١٢٣','\x1c甲乙\u00a0','a&<b>c','2024/111/01 12.3','', '单']:
        cases.append(capture('cross-'+repr(text),two,[ocr([10,10,190,30],text)]))
    cases.append(capture('projection-six',[ccw([i*100,0,(i+1)*100,80])for i in range(6)],[ocr([0,10,600,30],'100 200 300 400 500 600')]))
    cases.append(capture('dominant-padding',two,[ocr([80,10,190,30],'1234567890')]))
    cases.append(capture('overlapping-ambiguous',[ccw([0,0,150,80]),ccw([50,0,200,80])],[ocr([80,10,120,30],'text')]))
    cases.append(capture('point-zero-area',two,[ocr([10,10,10,10],'A')]))
    cases.append(capture('touch-only',two,[ocr([200,10,220,30],'outside')]))
    cases.append(capture('empty-pred',[],[ocr([0,0,10,10],'kept unmatched')]))
    cases.append(capture('gather',two,[ocr([60,11,80,30],'B'),ocr([10,10,30,30],' A '),ocr([10,45,80,65],'\r\n C \n'),ocr([120,10,150,30],'中文')]))
    rng=np.random.default_rng(731)
    for n in range(40):
        polygons=[ccw([i*100,0,(i+1)*100,80])for i in range(int(rng.integers(2,10)))];lines=[]
        for j in range(12):
            a,b=sorted(rng.uniform(0,len(polygons)*100,2));top=rng.uniform(0,60)
            lines.append(ocr([a,top,b,top+15],['123 456 789','甲乙丙丁戊己庚辛壬癸','ABCD','1,234.00 25%','12/31 abc','αβ²Ⅻ'][j%6]))
        cases.append(capture(f'random-{n}',polygons,lines))
    html=[]
    for name,points,boxes,texts in [
        ('normal-empty-column',[[0,0,0,0],[0,0,1,1]],[[0,0,100,80],[100,0,200,80]],[['A'],[]]),
        ('narrow-empty-column',[[0,0,0,0],[0,0,1,1]],[[0,0,10,80],[10,0,110,80]],[[],['A']]),
        ('wide-empty-column',[[0,0,0,0],[0,0,1,1]],[[0,0,300,80],[300,0,400,80]],[[],['A']]),
        ('visible-noise-preserved',[[0,0,0,0],[0,0,1,1]],[[0,0,10,80],[10,0,110,80]],[['X'],['A']]),
        ('merged-header',[[0,0,0,1],[1,1,0,0],[1,1,1,1]],[[0,0,200,40],[0,40,100,80],[100,40,200,80]],[['Header'],['A','B'],['C']]),
        ('sparse-edge',[[0,0,0,0],[0,0,1,1],[1,1,1,1]],[[0,0,10,40],[10,0,110,40],[10,40,110,80]],[[],['A'],['B']]),
        ('inner-hole',[[0,0,0,0],[0,0,1,1],[0,0,2,2],[1,1,0,0],[1,1,2,2]],[[0,0,100,40],[100,0,200,40],[200,0,300,40],[0,40,100,80],[200,40,300,80]],[['A'],[],['B'],['C'],['D']]),
        ('whitespace',[[0,0,0,0]],[[0,0,100,40]],[['\x1c A\r\nB \u00a0','\u2000']]),
        ('escape',[[0,0,0,0]],[[0,0,100,40]],[['<script>&"\'']]),
        ('empty',[],[],[]),
    ]:
        polygons=[ccw(b)for b in boxes]
        escape=lambda s:s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;').replace("'",'&#39;')
        text_map={i:[escape(v)for v in values]for i,values in enumerate(texts)}
        value=plot_html_table(points,text_map,polygons)
        html.append(dict(id=name,points=points,polygons=polygons,texts=texts,expected=value))
    write_json(ROOT/'build/migration/wired-match.json',dict(cases=cases,html=html))
    print(f'Exported {len(cases)} UNET matcher and {len(html)} HTML/trim contracts')

if __name__=='__main__':main()
