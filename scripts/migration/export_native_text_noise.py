"""Freeze old candidates, paint requests and complete source removal side effects."""
import copy,json,random,sys
from types import SimpleNamespace
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    edges='--edges'in sys.argv
    target=ROOT/('build/migration/native-text-noise-edges.json'if edges else'build/migration/native-text-noise.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_noise import _nested_prose_numeric_outliers_python as candidates,exclude_faint_native_noise as exclude
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None
    def line(text,b,h,index,font='Body'):
        return dict(text=text,bbox=b,effective_height=h,angle=0,source_index=index,font_signature=[font,0],chars=[dict(char=text,char_idx=index,bbox=b)])
    def nested():return [line('alpha bravo charlie delta echo',[10.,20.,200.,30.],10.,1),line('foxtrot golf hotel india juliet',[10.,40.,200.,50.],10.,2),line('12',[50.,31.,58.,39.],8.,3,'Number')]
    def directory():return [line(str(i+1)+'. directory item',[10.,y,250.,y+30.],30.,i+1)for i,y in enumerate([10.,60.,110.])]+[line('2',[60.,45.,68.,53.],8.,4,'Number')]
    def gutter():return [line('alpha bravo charlie delta echo',[10.,20.+i*15.,200.,30.+i*15.],10.,i+1)for i in range(3)]+[line('4',[100.,100.,108.,108.],8.,4,'Number')]
    cases=[]
    def add(label,rows,visual=(),paint='default',extra=()):
        rows=copy.deepcopy(rows);allchars=copy.deepcopy([c for r in rows for c in r.get('chars',[])]+list(extra))
        if paint=='default':paint={str(r['source_index']):([200,200,200,255]if r.get('font_signature')and r['font_signature'][0]=='Number'else[0,0,0,255])for r in rows}
        request=clean(dict(source=dict(lines=rows,chars=allchars,page_index=7,metadata={'keep':'original'}),visual_bboxes=visual,paint=paint))
        objects=[]
        for r in copy.deepcopy(rows):
            if r.get('font_signature')is not None:r['font_signature']=tuple(r['font_signature'])
            objects.append(_LineItem(**r))
        old=SimpleNamespace(lines=objects,chars=copy.deepcopy(allchars));original={id(l):r for l,r in zip(objects,rows)};positions={id(l):i for i,l in enumerate(objects)}
        cs=candidates(old,visual)
        requested=sorted({c['char_idx']for l,refs in cs for p in [l,*refs]for c in p.chars if c.get('char','').strip()and'char_idx'in c})
        prepared=dict(candidates=[dict(line_index=positions[id(l)],reference_indices=[positions[id(r)]for r in refs])for l,refs in cs],requested_char_indices=requested)
        calls=[]
        def read(indices):calls.append(indices);return {int(k):tuple(v)if v is not None else None for k,v in paint.items()}
        removed=exclude(old,read if paint is not None else None,visual)
        out=copy.deepcopy(request['source']);out['lines']=[original[id(l)]for l in old.lines];out['chars']=old.chars
        cases.append(dict(label=label,request=request,prepared=clean(prepared),expected=clean(dict(source=out,excluded_source_indices=sorted(removed),requested_char_indices=calls[0]if calls else[]))))
    add('nested',nested());add('directory',directory());add('gutter',gutter(),[(0.,50.,90.,200.),(120.,50.,220.,200.)]);add('empty',[])
    for value in [169,170,171,199,200,224,225,226]:
        for channel in range(3):
            color=[200,200,200,255];color[channel]=value
            add(f'color-{value}-{channel}',nested(),paint={'1':[0,0,0,255],'2':[0,0,0,255],'3':color})
    for alpha in [0,1,128,255]:add('alpha-'+str(alpha),nested(),paint={'1':[0,0,0,255],'2':[0,0,0,255],'3':[200,200,200,alpha]})
    for value in [119,120,121,244,245,246,255]:add('reference-'+str(value),nested(),paint={'1':[value]*3+[255],'2':[value]*3+[255],'3':[200]*3+[255]})
    for value in [None,{}, {'3':[200]*3+[255]}, {'1':[0,0,0,0],'2':[0,0,0,0],'3':[200,200,200,255]}, {'1':None,'2':[0,0,0,255],'3':None}]:add('missing-'+str(len(cases)),nested(),paint=value)
    for text in ['1','123','1234','１２','١٢','²','12a',' 12\u3000','\u200b12','-12','1.2','']:
        rows=nested();rows[-1]['text']=text;rows[-1]['chars'][0]['char']=text;add('digits-'+text,rows)
    for field,values in [('angle',[0,90,180,270]),('semantic_type',[None,'text','title']),('effective_height',[0.,6.9999,7.,8.,10.5,10.5001]),('font_signature',[None,[],['Body',0],['Number',0],['Number',None]])]:
        for value in values:
            rows=nested();rows[-1][field]=value;add(f'eligibility-{field}-{value}',rows)
    for count in [1,2,4,5]:
        rows=nested();rows.extend(line('1',[50.,60.+i*10.,58.,68.+i*10.],8.,10+i,'Number')for i in range(count-1));add('same-font-'+str(count),rows)
    for text in ['a b c d e','aa bb cc dd','aa bb cc dd ee','中文中文中文中文中文中','中文中文中文中文中文中文','AA bb 12 cc dd ee']:
        rows=nested();rows[0]['text']=rows[1]['text']=text;add('prose-'+text,rows)
    for b in [[9.,31.,58.,39.],[50.,10.,58.,18.],[50.,31.,201.,39.],[50.,31.,58.,50.],[50.,31.,58.,31.]]:
        rows=nested();rows[-1]['bbox']=b;add('bbox-'+str(b),rows)
    for text in ['1. English','１２、中文','12) word','123. word','１. z','1. 3','1. a',' 2)　中']:
        rows=directory()
        for r in rows[:-1]:r['text']=text
        add('entry-'+text,rows)
    for n in [2,3,4]:
        rows=gutter();rows=rows[:n]+[rows[-1]];add('gutter-prose-'+str(n),rows,[(0.,50.,90.,200.),(120.,50.,220.,200.)])
    for gap in [0.,1.59,1.6,1.61,3.]:add('gutter-gap-'+str(gap),gutter(),[(0.,50.,100.-gap,200.),(108.+gap,50.,220.,200.)])
    for missing in [False,True]:
        rows=nested();rows[-1]['chars'].append(dict(char=' ',bbox=[0.,0.,0.,0.]))
        if missing:rows[-1]['chars'][0].pop('char_idx')
        add('missing-index-'+str(missing),rows,extra=[dict(char='unindexed'),dict(char='other',char_idx=99)])
    rows=nested();rows[0]['source_index']=3;add('duplicate-source-indices',rows)
    rng=random.Random(261029)
    for i in range(100):
        rows=nested()if i%2 else directory();r=rng.choice(rows);r['bbox']=list(r['bbox']);r['bbox'][rng.randrange(4)]+=rng.choice([-.01,.01,-5.,5.]);add('geometry-'+str(i),rows)
    if edges:
        cases=[]
        for bright in [False,True]:
            for count in [7,8,9,10]:
                for alpha in [0,255]:
                    rows=nested();paint={'3':[200,200,200,255]}
                    for j,r in enumerate(rows[:2]):
                        r['chars']=[dict(char='x',char_idx=10+j*5+k)for k in range(5)]
                    for k in range(10):
                        value=(245 if bright else 120)if k<count else 180
                        paint[str(10+k)]=[value]*3+[alpha if k==9 else 255]
                    add(f'ratio-{bright}-{count}-{alpha}',rows,paint=paint)
        for color in [[170,205,200,255],[170,206,200,255],[190,225,210,255],[189,225,210,255]]:
            add('channel-delta-'+str(color),nested(),paint={'1':[0,0,0,255],'2':[0,0,0,255],'3':color})
        for font in [None,[],['Body',False],['Body',0]]:
            rows=gutter()
            for r in rows[:-1]:r['font_signature']=font;r['angle']=90
            add('gutter-font-'+str(font),rows,[(0.,50.,90.,200.),(120.,50.,220.,200.)],paint={str(i):([200,200,200,255]if i==4 else[0,0,0,255])for i in range(1,5)})
        for c in ['\u33ff','\u3400','\u9fff','\ua000']:
            rows=nested()
            for r in rows[:2]:r['text']=c*12
            add('cjk-'+str(ord(c)),rows)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/text_noise.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/models.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases))
    print('Frozen',len(cases),'noise cases;',sum(bool(c['prepared']['candidates'])for c in cases),'candidate-positive;',sum(bool(c['expected']['excluded_source_indices'])for c in cases),'removal-positive')
if __name__=='__main__':main()
