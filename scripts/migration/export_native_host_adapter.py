"""Freeze unchanged host adapter, including object-identity supplement consumption."""
import copy,json,math
from types import SimpleNamespace
from common import ROOT,write_json
from rapid_doc.model.docvortex_adapter import convert_docvortex_layout,docvortex_layout_fallback_reason,merge_docvortex_supplements,_table_text_angle,_LABELS
def main():
    target=ROOT/'build/migration/native-host-adapter.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(op,q,label,actual=None):
        try:
            if op=='convert':out=convert_docvortex_layout(q['blocks'],q['width'],q['height'],set(q.get('ignored_labels',[])))
            elif op=='fallback':out=docvortex_layout_fallback_reason(q['blocks'],q['layout'],q['width'],q['height'],q.get('native_char_count',0))
            elif op=='direction':out=_table_text_angle(SimpleNamespace(geometry=SimpleNamespace(chars=q['chars'])),q['table_bbox'])
            else:out=merge_docvortex_supplements(q['native'],actual if actual is not None else q['supplements'],include_formula=q.get('include_formula',True),include_seal=q.get('include_seal',True))
        except Exception as e:cases.append(dict(label=label,operation=op,request=q,error=type(e).__name__));return
        cases.append(dict(label=label,operation=op,request=q,expected=out))
    def block(label,content='Original text',**kw):return dict(type=label,bbox=[.1,.2,.8,.7],content=content,**kw)
    for label in _LABELS:
        for a in [0,90]:
            for ignored in [[],[label]]:add('convert',dict(blocks=[block(label,angle=a)],width=317,height=593,ignored_labels=ignored),f'{label}/{a}/{ignored}')
    for caption in ['Table 2','tab. 2','表２','表格 \u001c２','Figure 1','FIG.2','FİG.2','fıgure 3','图 ٢','\u001cTable 3','ordinary legend',[dict(content='Ta'),dict(content=['ble ',dict(content='1')])]]:
        for a in [0,90]:add('convert',dict(blocks=[block('caption',caption,angle=a)],width=301,height=607),f'caption/{caption}/{a}')
    for bb in [None,[],[0,0,1],[0,0,0,1],[-1,-2,2,3],[1,0,0,1],['.1','.2','.8','.7'],['nan',0,1,1],[0,0,'nan',1],[False,0,True,1],[None,0,1,1]]:
        add('convert',dict(blocks=[dict(type='text',bbox=bb,content='x')],width=301,height=607),f'bbox/{bb}')
    for content in ['x',[],None,[dict(content=['A',dict(content='Ｂ')],url='keep')]]:
        for angle in ['90','90.0',False,None,270]:add('convert',dict(blocks=[block('text',content,angle=angle)],width=317,height=593),f'rich/{content}/{angle}')
    def layout(cat=1,label='text',b=(10.,10.,90.,30.),score=1.,**kw):
        x,y,r,t=b;return dict(category_id=cat,original_label=label,original_order=0,poly=[x,y,r,y,r,t,x,t],polygon_points=None,score=score,**kw)
    for label in ['header','footer','aside_text','page_number','image','text']:
        for proportion in [.399999,.4,.400001]:
            for count in [79,80]:add('fallback',dict(blocks=[block(label,'A'*80)],layout=[layout(3 if label=='image'else 1 if label=='text'else 2,label,(0.,0.,100.,100.*proportion))],width=100,height=100,native_char_count=count),f'fallback/{label}/{proportion}/{count}')
    for blocks,ls in [([],[]),([block('text')],[]),([],[layout()]),([block('header','a'*79)],[layout(2,'page_number')])]:add('fallback',dict(blocks=blocks,layout=ls,width=100,height=100),f'empty/{len(cases)}')
    base=[layout(1,'text',(0.,0.,100.,20.)),layout(14,'equation',(20.,40.,80.,60.)),layout(1,'text',(0.,80.,100.,100.)),layout(5,'table',(0.,120.,100.,200.)),layout(3,'image',(120.,0.,200.,80.))]
    extras=[layout(14,'isolate_formula',(20.,40.,80.,60.),.9),layout(14,'isolate_formula',(21.,40.,79.,60.),.8),layout(9,'formula_number',(85.,45.,95.,55.)),layout(13,'inline_formula',(30.,130.,60.,150.),.9),layout(3,'seal',(130.,10.,190.,70.),.95),layout(14,'isolate_formula',(150.,90.,190.,110.),.4999)]
    for native in [[],base,base[::-1]]:
        for formula in [False,True]:
            for seal in [False,True]:
                for pp in [[],extras,extras[::-1]]:add('supplements',dict(native=native,supplements=pp,include_formula=formula,include_seal=seal),f'supplements/{len(cases)}')
    for score in [.499999,.5,.500001]:
        for x in [19.9999,20.,20.0001]:add('supplements',dict(native=base,supplements=[layout(14,'equation',(x,40.,80.,60.),score)]),f'score-overlap/{score}/{x}')
    for native in [base,[]]:
        pp=[extras[0],extras[0],copy.deepcopy(extras[0]),extras[2],extras[2]]
        add('supplements',dict(native=native,supplements=pp,supplement_identities=[0,0,2,3,3]),f'aliases/{len(native)}',actual=pp)
    for values in [[0.,math.pi/2],[-math.pi/2,0.],[math.radians(4.9999),math.radians(5.0001)],[math.pi,math.pi/2],[None,'bad'],['nan','inf']]:
        for writing in [False,True]:
            chars=[dict(char='A',bbox=[float(i),1.,i+1.,3.],rotation=a,**(dict(writing_angle=a)if writing else{}))for i,a in enumerate(values)]
            add('direction',dict(chars=chars,table_bbox=[0.,0.,100.,100.]),f'direction/{values}/{writing}')
    acceptance=json.loads((ROOT/'build/migration/native-pdf-acceptance.json').read_text(encoding='utf-8'))
    for c in acceptance['cases']:
        for i,page in enumerate(c['expected']['pages']):add('convert',dict(blocks=page,width=1191,height=1684,ignored_labels=['number','header','footer','aside_text']),f'actual/{c["label"]}/{i}')
    write_json(target,dict(cases=cases));print(len(cases),'original host adapter cases;',sum('error'in c for c in cases),'original exceptions')
if __name__=='__main__':main()
