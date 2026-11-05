"""冻结旧banded_numeric候选、数值语法、共同行隙及CPython来源set顺序。"""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-banded-numeric.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import banded_numeric as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    cases=[];gutters=[];rng=random.Random(361833)
    def objects(r):
        v=r['table_input'];q=r['text'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v.get('chars',())),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']));tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height']);return ti,tx
    def add(r,label):
        ti,tx=objects(r);d={};c=old.build_banded_numeric_candidate(ti,tx,d);cases.append(dict(label=label,request=r,expected=dict(candidate=None if c is None else asdict(c),diagnostics=copy.deepcopy(d),html=None if c is None else serialize_candidate_html(c))))
    prior=json.loads((ROOT/'build/migration/native-table-multiline.json').read_text(encoding='utf-8'))
    for c in prior['cases']:add(c['request'],'prior-'+c['label'])
    real=copy.deepcopy(prior['real'])
    def make(cols=4,header=1,body=4,mode='outer',grouped=False):
        width=cols*60.;height=(header+body)*24.+6.;gs=[];rs=[];chars=[];lines=[];rects=[]
        defs=[]
        for row in range(header):
            if grouped and row==0:defs.append([('Metrics',65.,width-10.)])
            elif grouped and row==1:defs.append([('GroupA',65.,width-10.)])
            elif grouped and row==2:defs.append([(f'Col{c}',c*60.+6.,c*60.+22.)for c in range(1,cols)])
            else:defs.append([(f'Head{row}-{c}',c*60.+6.,c*60.+22.)for c in range(cols)])
        defs += [[(f'Row{r}' if c==0 else str(10+r+c),c*60.+6.,c*60.+22.)for c in range(cols)]for r in range(body)]
        for row,tokens in enumerate(defs):
            ids=[];ts=[]
            for s,x0,x1 in tokens:
                i=len(gs);b=[x0,row*24.+6.,x1,row*24.+14.];gs.append(asdict(NativeTableGlyph(i,i,s,tuple(b),row)));ids.append(i);ts.append(asdict(NativeTableToken(s,tuple(b),(i,),(i,))));chars.append(dict(char_idx=i,char=s,bbox=b))
            rs.append(asdict(NativeTableTextRow(row,(0.,row*24.+6.,width,row*24.+14.),tuple(NativeTableToken(**t)for t in ts),tuple(ids))))
        if mode in ('outer','both'):lines=[dict(bbox=[0.,y,width,y+.1],width=.1,orientation='horizontal')for y in (0.,height-.1)]
        if mode in ('fill','both'):rects=[dict(bbox=[60.*c,0.,60.*(c+1),20.],segment_count=5,fill_visible=True,stroke_visible=False)for c in range(cols)]
        if grouped:lines.append(dict(bbox=[60.,45.95,width,46.05],width=.1,orientation='horizontal'))
        return json.loads(json.dumps(dict(table_input=dict(table_bbox=[0.,0.,width,height],page_size=[1000.,1000.],angle=0,chars=chars,drawing_lines=lines,rectangles=rects),text=dict(glyphs=gs,rows=rs,median_glyph_width=8.,median_glyph_height=8.))))
    for cols in (2,3,4,5,7,20,21):
        for head in (0,1,2,3,4):
            for body in (2,3,4):
                for mode in ('none','outer','fill','both'):add(make(cols,head,body,mode),f'grid-{cols}-{head}-{body}-{mode}')
    for cols in (3,4,5,7):
        for mode in ('none','outer','fill','both'):
            base=make(cols,3,4,mode,True);add(base,f'grouped-{cols}-{mode}')
            for change in ('no-rule','two-rules','short-rule','extra-left-title','leaf-short'):
                r=copy.deepcopy(base)
                if change=='no-rule':r['table_input']['drawing_lines']=r['table_input']['drawing_lines'][:-1]
                if change=='two-rules':r['table_input']['drawing_lines'].append(copy.deepcopy(r['table_input']['drawing_lines'][-1]))
                if change=='short-rule':r['table_input']['drawing_lines'][-1]['bbox'][2]=75.
                if change=='extra-left-title':r['text']['glyphs'][0]['bbox'][0]=0.
                if change=='leaf-short':r['text']['rows'][2]['tokens']=r['text']['rows'][2]['tokens'][:-1]
                add(r,f'grouped-{cols}-{mode}-{change}')
    base=make()
    for numeric in ('0','-1','−1','1,234.5','1%','1 k','1M','O','o','✓','✔','✗','✘','×','1/2','1++','1—','１２３','١٢','²','NaN'):
        r=copy.deepcopy(base)
        for row in r['text']['rows'][1:]:
            for token in row['tokens'][1:]:
                token['text']=numeric;i=token['glyph_ids'][0];r['text']['glyphs'][i]['text']=numeric;r['table_input']['chars'][i]['char']=numeric
        add(r,f'numeric-{numeric}')
    for key in ('char_idx','bbox','char'):
        r=copy.deepcopy(base)
        for c in r['table_input']['chars']:c.pop(key)
        add(r,f'raw-missing-{key}')
    for row in (0,1,2,4):
        for dx in (-30.,-.6,0.,.6,30.):
            r=copy.deepcopy(base);i=r['text']['rows'][row]['glyph_ids'][1];r['text']['glyphs'][i]['bbox'][0]+=dx;add(r,f'gutter-glyph-{row}-{dx}')
    for fill in (False,True):
        for delta in (-1.00001,-1.,0.,1.,1.00001):
            r=make(mode='fill');r['table_input']['rectangles'][1]['bbox'][0]+=delta;r['table_input']['rectangles'][0]['fill_visible']=fill;add(r,f'filled-adjacency-{fill}-{delta}')
    for _ in range(120):
        r=make(cols=rng.randrange(2,7),header=rng.randrange(1,4),body=rng.randrange(3,8),mode=rng.choice(['none','outer','fill']))
        for g in r['text']['glyphs']:
            dx=rng.uniform(-3,3);g['bbox'][0]+=dx;g['bbox'][2]+=dx
        add(r,'jitter')
    for angle in (90,180,270,45,-90):r=copy.deepcopy(base);r['table_input']['angle']=angle;add(r,f'angle-{angle}')
    for bbox in ([240.,126.,0.,0.],[0.,0.,0.,126.]):r=copy.deepcopy(base);r['table_input']['table_bbox']=bbox;add(r,'raw-bbox')
    for lower,upper in ((22.,66.),(42.,42.6),(42.,42.64),(66.,22.)):
        for indices in ([],[0],[1,2,3,4]):
            _,tx=objects(base);gutters.append(dict(text=base['text'],indices=indices,lower=lower,upper=upper,expected=old._shared_gutter(tx,set(indices),lower,upper)))
    strings=['','-','–','—','O','o','✓','✔','✗','✘','×','1e3','(1)','−1','1\n',' 1 k ','1%%','1++','1/2','<1','≤1','1 K','1\u001c2']
    for i in range(0x110000):
        ch=chr(i)
        if ch.isdecimal():strings.extend([ch,'-'+ch+'%'])
    for _ in range(300):strings.append(''.join(rng.choices('01a+-–—/ .,<>≤≥%kKMBO✓²\t',k=rng.randrange(1,12))))
    numbers=[dict(text=s,expected=old._numeric_value(s))for s in strings]
    sets=[dict(input=a,expected=list(set(a)))for a in [[],[0,1,2],[7,1,7,-1,-2],[-1,-2,-3,2**32,0],*[[rng.randrange(-10**12,10**12)for _ in range(rng.randrange(1,150))]for _ in range(160)]]]
    src='rapid_doc/_vendor/docvortex_core/';paths=['analyzers/native/pdf/_table_recovery/'+p for p in ('banded_numeric.py','sparse_common.py','candidate.py')]+['analyzers/native/pdf/spatial_text.py'];write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in paths},cases=cases,real=real,gutters=gutters,numbers=numbers,sets=sets));print(f'Frozen {len(cases)} banded cases, {sum(c["expected"]["candidate"] is not None for c in cases)} nonempty; {len(numbers)} numeric, {len(gutters)} gutter, {len(sets)} signed set facts')
if __name__=='__main__':main()
