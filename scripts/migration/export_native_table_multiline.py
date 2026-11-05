"""冻结旧多行稀疏完整主分支与grouped_rule_band回退，不覆写参考。"""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json

def main():
    path=ROOT/'build/migration/native-table-multiline.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import sparse_multiline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    cases=[]
    def add(r,label):
        v=r['table_input'];q=r['text']
        ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v.get('chars',())),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']))
        tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height'])
        d=[];cs=old.build_sparse_multiline_candidates(ti,tx,d)
        cases.append(dict(label=label,request=r,expected=dict(candidates=[asdict(c)for c in cs],diagnostics=copy.deepcopy(d),html=[serialize_candidate_html(c)for c in cs])))
    prior=json.loads((ROOT/'build/migration/native-table-sparse-hybrid.json').read_text(encoding='utf-8'))
    for c in prior['cases']:add(c['request'],'prior-'+c['label'])
    real=copy.deepcopy(prior['real'])
    def make(pattern,cols=4,mode='none',head=1):
        width=cols*60.;height=len(pattern)*24.+6.;gs=[];rs=[];chars=[]
        for row,cs in enumerate(pattern):
            ids=[];ts=[]
            for col in cs:
                i=len(gs);b=[col*60.+6.,row*24.+6.,col*60.+22.,row*24.+14.];s=chr(65+i%26)
                gs.append(asdict(NativeTableGlyph(i,i,s,tuple(b),row)));ids.append(i);ts.append(asdict(NativeTableToken(s,tuple(b),(i,),(i,))));chars.append(dict(char_idx=i,char=s,bbox=b))
            rs.append(asdict(NativeTableTextRow(row,(0.,row*24.+6.,width,row*24.+14.),tuple(NativeTableToken(**t)for t in ts),tuple(ids))))
        lines=[];rects=[]
        if mode in ('header','bands'):lines=[dict(bbox=[0.,head*24.-2.,width,head*24.-1.9],width=.1,orientation='horizontal')]
        if mode=='bands':lines+= [dict(bbox=[0.,y,width,y+.1],width=.1,orientation='horizontal')for y in [0.,height-1.,*range(head*24+47,int(height)-1,48)]]
        if mode=='fill':rects=[dict(bbox=[60.*c,24.*r,60.*(c+1),24.*r+20.],segment_count=5,fill_visible=True,stroke_visible=True)for r in range(len(pattern))for c in range(cols)]
        return json.loads(json.dumps(dict(table_input=dict(table_bbox=[0.,0.,width,height],page_size=[1000.,1000.],angle=0,chars=chars,drawing_lines=lines,rectangles=rects),text=dict(glyphs=gs,rows=rs,median_glyph_width=8.,median_glyph_height=8.))))
    full=[0,1,2,3]
    patterns=[ [full]*6, [full,full,[1,2,3],full,[1,2,3],full,[1,2,3]], [full]+[full,[1,2,3],[1,2,3]]*3, [full]+[full,full,[1,2,3],[1,2,3]]*3, [full]+[[1,2,3],[0],[1,2,3]]*3, [full]+[[0,1],[2,3]]*4, [full]+[[1,2],full,[1,2]]*3, [full]+[full,[3],full,[3],full,[3]], [full]+[full,[1,2],[1,2],full,[1,2],[1,2]] ]
    for mode in ('none','header','bands','fill'):
        for pattern in patterns:add(make(pattern,mode=mode),f'logical-{mode}-{patterns.index(pattern)}')
    rng=random.Random(7237)
    for _ in range(220):
        pattern=[full]+[sorted(rng.sample(full,rng.randrange(1,5)))for _ in range(rng.randrange(3,16))]
        add(make(pattern,mode=rng.choice(['none','header','bands','fill'])),'occupancy-pattern')
    # 真正有多个独立整行标题带，数据行>=3支持空白走廊；原始字形与source_idx共同保存。
    grouped=make([full,[0],full,full,full,[0],full,full,full],mode='none')
    grouped['table_input']['drawing_lines']=[dict(bbox=[0.,y,240.,y+.1],width=.1,orientation='horizontal')for y in [0.,24.,48.,120.,144.,221.9]]
    for text in ('Group','Section A','第一组','A\u00a0B','A—B','ﬁ','',None):
        r=copy.deepcopy(grouped)
        for row in (1,5):
            i=r['text']['rows'][row]['glyph_ids'][0];s=text or 'G';r['text']['glyphs'][i]['text']=s;r['text']['rows'][row]['tokens'][0]['text']=s;r['table_input']['chars'][i]['char']=text
        add(r,f'group-text-{text}')
    for count in (0,1,2,3,4,5,6):
        r=copy.deepcopy(grouped);r['table_input']['drawing_lines']=r['table_input']['drawing_lines'][:count];add(r,f'group-horizontal-{count}')
    for x in (0.,.8,.800001,60.,239.2,240.):
        r=copy.deepcopy(grouped);r['table_input']['drawing_lines'].append(dict(bbox=[x-.05,24.,x+.05,48.],width=.1,orientation='vertical'));add(r,f'group-vertical-{x}')
    for dy in (-3.,-1.,-.8,0.,.8,1.,3.):
        r=copy.deepcopy(grouped);r['text']['rows'][1]['bbox'][1]=24.+dy;r['text']['rows'][1]['bbox'][3]=32.+dy;add(r,f'group-row-boundary-{dy}')
    for c in (0,1,2):
        r=copy.deepcopy(grouped);r['text']['rows'][2]['tokens']=r['text']['rows'][2]['tokens'][:c];add(r,f'group-token-count-{c}')
    # 空白必须左右可见source都在同一跨列格内；点状空格框也保留。
    for space in (' ','\t','\u00a0','\n'):
        r=copy.deepcopy(grouped)
        for g in r['text']['glyphs']:g['source_index']*=3
        for row in r['text']['rows']:
            for t in row['tokens']:t['source_char_indices']=[i*3 for i in t['source_char_indices']]
        for c in r['table_input']['chars']:c['char_idx']*=3
        for row in (1,5):
            i=r['text']['rows'][row]['glyph_ids'][0];a=r['text']['glyphs'][i];a['text']='A B';a['bbox'][2]=26.;r['text']['rows'][row]['tokens'][0]['text']='A B'
            original=r['table_input']['chars'][i];original.update(char='A',bbox=[6.,row*24.+6.,14.,row*24.+14.]);idx=original['char_idx']
            r['table_input']['chars'].extend([dict(char_idx=idx+1,char=space,bbox=[14.,row*24.+6.,14.,row*24.+6.]),dict(char_idx=idx+2,char='B',bbox=[18.,row*24.+6.,26.,row*24.+14.])]);a['source_index']=idx
            # 两个可见源必须真实属于候选格；拆glyph与token引用。
            new=copy.deepcopy(a);new.update(glyph_id=len(r['text']['glyphs']),source_index=idx+2,text='B',bbox=[18.,row*24.+6.,26.,row*24.+14.]);a.update(text='A',bbox=[6.,row*24.+6.,14.,row*24.+14.]);r['text']['glyphs'].append(new);r['text']['rows'][row]['glyph_ids'].append(new['glyph_id']);tok=r['text']['rows'][row]['tokens'][0];tok['glyph_ids'].append(new['glyph_id']);tok['source_char_indices'].append(idx+2)
        add(r,f'group-explicit-space-{space!r}')
    src='rapid_doc/_vendor/docvortex_core/'
    sources=['analyzers/native/pdf/_table_recovery/'+p for p in ('sparse_multiline.py','grouped_rule_band.py','sparse_common.py','candidate.py')]+['analyzers/native/pdf/spatial_text.py','document/pdf/text/spacing.py']
    write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in sources},cases=cases,real=real))
    print(f'Frozen {len(cases)} complete multiline cases, {sum(bool(c["expected"]["candidates"])for c in cases)} nonempty, {sum(bool(c["expected"]["candidates"])and any(d.get("evidence")=="grouped_rule_band" for d in c["expected"]["diagnostics"])for c in cases)} grouped')
if __name__=='__main__':main()
