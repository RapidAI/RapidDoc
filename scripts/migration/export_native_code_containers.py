"""Freeze both complete old code-container builders and immutable real input projections."""
import copy,json,random
from common import ROOT,digest,write_json
from export_native_table_page_commit import typed
from export_native_table_page_policies import empty
from export_native_inline_clusters import clean
def glyphs(s):
    chars=[]
    for l in s['lines']:
        l['chars']=[];b=l['bbox'];step=l.get('median_glyph_width',5.)or 5.
        for i,t in enumerate(l['text']):
            ch=dict(char=t,char_idx=len(chars),bbox=[b[0]+i*step,b[1],b[0]+(i+1)*step,b[3]],font=dict(name=l.get('font_signature',['Courier'])[0]))
            chars.append(ch);l['chars'].append(ch)
    s['chars']=chars
def base(kind):
    s=empty();s['page_size']=[400.,600.];texts=['def function(x):','    if x > 0:','        return x + 1','        return 0','    else:','return value']
    if kind=='algorithm':texts=['input data','for each item','if valid','return result','end for','output result']
    for i,t in enumerate(texts):
        x=40.+[0,10,20,20,10,0][i];s['lines'].append(dict(text=t,bbox=[x,110.+20*i,x+len(t)*5.,120.+20*i],source_index=i,angle=0,effective_height=10.,median_glyph_width=5.,font_signature=['Courier',0],visual_row_id=i))
    if kind=='filled':s['path_infos']=[dict(bbox=[30.,90.,370.,250.],segment_count=5,form_depth=0,source_index=0,fill_visible=True,stroke_visible=False,fill_rgba=[240,240,240,255],rectangle_bboxes=[[30.,90.,370.,250.]])]
    if kind=='ruled':s['drawing_lines']=[dict(bbox=[30.,y,370.,y],orientation='horizontal',width=0.)for y in [100.,240.]]
    if kind=='algorithm':s['lines'].insert(0,dict(text='Algorithm 1: Example',bbox=[40.,75.,300.,85.],source_index=100,angle=0,effective_height=10.,font_signature=['Body',0],median_glyph_width=5.))
    glyphs(s);return dict(source=s,kind='filled'if kind=='filled'else'ruled',excluded_bboxes=[],claimed_line_indices=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import code_blocks as old
    s=typed(q['source']);f=old._build_code_blocks if q['kind']=='filled'else old._build_rule_delimited_code_blocks
    blocks,claimed=f(s,[tuple(b)for b in q.get('excluded_bboxes',[])],set(q.get('claimed_line_indices',[])))
    return clean(dict(code_blocks=blocks,claimed_line_indices=sorted(claimed)))
def main():
    target=ROOT/'build/migration/native-code-containers.json'
    if target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    def add(label,q):
        q=clean(copy.deepcopy(q))
        try:expected=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=expected))
    for kind in ['filled','ruled','algorithm']:
        q=base(kind);add(kind,q);assert cases[-1]['expected']['code_blocks'],kind
        for mode in ['no-glyphs','no-font','no-paths','no-rules','all-excluded','all-claimed','one-claimed','prose','all-one-row','line-number-gutter','irregular-pitch','large-gap','int-coordinates','duplicate-char','blank-row','empty','chars-no-font','metadata']:
            q=base(kind);s=q['source']
            if mode=='no-glyphs':
                for l in s['lines']:l['chars']=[]
                s['chars']=[]
            if mode=='no-font':
                for l in s['lines']:l['font_signature']=['Body',0]
                glyphs(s)
            if mode=='no-paths':s['path_infos']=[]
            if mode=='no-rules':s['drawing_lines']=[]
            if mode=='all-excluded':q['excluded_bboxes']=[[0.,0.,400.,600.]]
            if mode=='all-claimed':q['claimed_line_indices']=[l['source_index']for l in s['lines']]
            if mode=='one-claimed':q['claimed_line_indices']=[0]
            if mode=='prose':
                for l in s['lines']:l['text']='These ordinary words describe a prose paragraph'
                glyphs(s)
            if mode=='all-one-row':
                for l in s['lines']:l['visual_row_id']=0
            if mode=='line-number-gutter':
                for i,l in enumerate(s['lines']):l['visual_row_id']=i
                s['lines'] += [dict(text=str(i),bbox=[32.,l['bbox'][1],37.,l['bbox'][3]],source_index=200+i,angle=0,effective_height=10.,visual_row_id=i,font_signature=['Courier',0],median_glyph_width=5.)for i,l in enumerate(s['lines'])]
                glyphs(s)
            if mode=='irregular-pitch':s['lines'][3]['bbox'][1]+=7.;s['lines'][3]['bbox'][3]+=7.;glyphs(s)
            if mode=='large-gap':
                for l in s['lines'][3:]:l['bbox'][1]+=45.;l['bbox'][3]+=45.
                glyphs(s)
            if mode=='int-coordinates':
                for l in s['lines']:l['bbox']=[int(x)for x in l['bbox']]
                for p in s['path_infos']:p['bbox']=[int(x)for x in p['bbox']]
            if mode=='duplicate-char':
                for l in s['lines']:l['chars']+=copy.deepcopy(l['chars'])
            if mode=='blank-row':
                for l in s['lines'][3:]:l['bbox'][1]+=20.;l['bbox'][3]+=20.
                glyphs(s)
            if mode=='empty':s.update(lines=[],chars=[],drawing_lines=[],path_infos=[])
            if mode=='chars-no-font':
                for l in s['lines']:
                    for ch in l['chars']:ch.pop('font',None)
            if mode=='metadata':s['preserve']={'x':[1,2,'ok']}
            add(kind+'-'+mode,q)
    for color in [None,[255,255,255,255],[251,255,255,255],[250,255,255,255],[240,240,240,0],[0,0,0,1]]:
        q=base('filled');q['source']['path_infos'][0]['fill_rgba']=color;add('fill-'+str(color),q)
    for font in ['ABCDEF+Courier','Code-Font','Consolas','Ｍｏｎｏ','Monotype','Body','StraßeMono','courıer']:
        q=base('filled')
        for l in q['source']['lines']:l['font_signature']=[font,0]
        glyphs(q['source']);add('font-'+font,q)
    for angle in [0,90,180,270]:
        q=base('filled');s=q['source']
        from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
        for l in s['lines']:
            l['angle']=angle;l['bbox']=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(s['page_size']),angle))
            for ch in l['chars']:ch['bbox']=list(_rotate_bbox_from_upright(tuple(ch['bbox']),tuple(s['page_size']),angle))
        for p in s['path_infos']:p['bbox']=list(_rotate_bbox_from_upright(tuple(p['bbox']),tuple(s['page_size']),angle))
        add('rotation-'+str(angle),q)
    for text in ['Algorithm 1: Example','算法 １：例','Algorithm 1 Example','ALGORİTHM ١. Test','Not an algorithm','Algorithm１: Example']:
        q=base('algorithm');q['source']['lines'][0]['text']=text;glyphs(q['source']);add('heading-'+text,q)
    for control in ['IF x','ıf x','ſelse x','1: if x','١. while x','1 if x','require x','ensure x','return_value','endé','if\u0301']:
        q=base('algorithm')
        for l in q['source']['lines'][1:]:l['text']=control
        glyphs(q['source']);add('control-'+repr(control),q)
    for slot in [0.3499999,0.35,0.3500001,0.79999,0.8]:
        q=base('filled')
        for l in q['source']['lines']:l['bbox'][0]+=slot*5.;l['bbox'][2]+=slot*5.
        glyphs(q['source']);add('slot-'+str(slot),q)
    rng=random.Random(271030)
    for i in range(100):
        kind=rng.choice(['filled','ruled','algorithm']);q=base(kind);s=q['source']
        for l in s['lines']:
            shift=rng.choice([0.,0.,0.,.2,2.,8.]);l['bbox'][0]+=shift;l['bbox'][2]+=shift
            if rng.randrange(10)==0:l['font_signature']=['Body',0]
        if rng.randrange(5)==0:q['claimed_line_indices']=[rng.randrange(6)]
        glyphs(s);add('combined-'+str(i)+'-'+kind,q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['code_blocks.py','layout_evidence.py','line_layout.py','spatial_text.py','native_text.py','geometry.py']]
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Actual complete old filled and rule+algorithm code block builders; real glyphs or explicit glyph-free fallback, actual lane evidence, no candidate/projection substitution.',cases=cases,errors=errors));print('Frozen code facts',len(cases),'errors',len(errors),'positive',sum(bool(c['expected']['code_blocks'])for c in cases))
if __name__=='__main__':main()
