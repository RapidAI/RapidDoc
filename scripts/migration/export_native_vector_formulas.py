"""Freeze complete old vector-formula construction and repeated-decoration pass."""
import copy,json,random
from types import SimpleNamespace as NS
from common import ROOT,write_json,digest
from export_form_runtime import typed
from export_native_image_chain import line
from export_native_table_page_policies import empty
from export_native_inline_clusters import clean

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.formulas import _build_vector_formula_blocks
    blocks,claimed=_build_vector_formula_blocks(typed(q['source']),q['container_blocks'],set(q['claimed_line_indices']))
    return clean(dict(vector_formula_blocks=blocks,claimed_line_indices=sorted(claimed)))

def path(b,i,**kw):return dict(bbox=b,source_index=i,segment_count=20,form_depth=0,fill_visible=True,stroke_visible=False,fill_rgba=[0,0,0,255],rectangle_bboxes=[],**kw)
def formula(x=150.,y=200.,n=8,step=9.,h=10.):return [path([x+j*step,y,x+j*step+7.,y+h],j)for j in range(n)]
def base():
    s=empty();s['lines']=[line('Original body paragraph with complete native sentence',[30.,y,470.,y+10.],i,font_coverage=1.)for i,y in enumerate([100.,300.,500.])];s['path_infos']=formula()
    return dict(source=s,container_blocks=[],claimed_line_indices=[])

def main():
    target=ROOT/'build/migration/native-vector-formulas.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[];rng=random.Random(176)
    def add(label,q):cases.append(dict(label=label,request=copy.deepcopy(q),expected=reference(q)))
    for mode in ['core','no-paths','few-lines','claimed-body','form-depth','stroke','unfilled','too-few','simple-paths','large-glyph','margin','collision','same-row-prose','container','thin-rule','right-number','left-number','invalid-number','prose-parentheses','two-numbers','local-path-number','right-path-number','inline','inline-color','inline-short','wordmark','wordmark-publication','wordmark-math','bottom-mark','fraction-four-complex','two-lanes','two-cores-one-number','joined-rows','joined-rows-blocked']:
        q=base();s=q['source']
        if mode=='no-paths':s['path_infos']=[]
        if mode=='few-lines':s['lines']=s['lines'][:2]
        if mode=='claimed-body':q['claimed_line_indices']=[0]
        if mode=='form-depth':
            for p in s['path_infos']:p['form_depth']=1
        if mode=='stroke':
            for p in s['path_infos']:p['stroke_visible']=True
        if mode=='unfilled':
            for p in s['path_infos']:p['fill_visible']=False
        if mode=='too-few':s['path_infos']=s['path_infos'][:4]
        if mode=='simple-paths':
            for p in s['path_infos']:p['segment_count']=7
        if mode=='large-glyph':s['path_infos']=formula(h=31.)
        if mode=='margin':s['path_infos']=formula(y=10.)
        if mode=='collision':s['lines'].append(line('Original body text in the formula area',[145.,200.,300.,210.],10))
        if mode=='same-row-prose':s['lines'].append(line('prose words',[225.,200.,400.,210.],10))
        if mode=='container':q['container_blocks']=[dict(type='image',bbox=[140.,190.,300.,220.])]
        if mode=='thin-rule':s['path_infos'].append(path([145.,212.,230.,213.],99));s['path_infos'][-1]['segment_count']=2
        if mode in ['right-number','invalid-number','prose-parentheses','two-numbers','wordmark-math']:s['lines'].append(line('(2)'if mode!='invalid-number'else'(x)',[450.,200.,465.,210.],10))
        if mode=='prose-parentheses':s['lines'][-1]['text']='(words)'
        if mode=='two-numbers':s['lines'].append(line('(3)',[450.,200.,465.,210.],11))
        if mode=='left-number':s['lines'].append(line('（A.2-3）',[30.,200.,60.,210.],10))
        if mode in ['local-path-number','right-path-number']:
            x=240. if mode=='local-path-number'else 450.
            for j,b in enumerate([[x,200.,x+2.,210.],[x+4.,200.,x+8.,210.],[x+10.,200.,x+12.,210.]]):s['path_infos'].append(path(b,100+j))
        if mode in ['inline','inline-color','inline-short']:
            s['lines']+= [dict(line('These original words',[30.,200.,145.,210.],10),font_signature=['Body',0]),dict(line('continue in native paragraph',[225.,200.,470.,210.],11),font_signature=['Body',0])]
            if mode=='inline-color':
                for p in s['path_infos']:p['fill_rgba']=[255,0,0,255]
            if mode=='inline-short':s['lines'][-1]['text']='word';s['lines'][-2]['text']='word'
        if mode.startswith('wordmark')or mode=='bottom-mark':
            y=35. if mode!='bottom-mark'else 550.;s['path_infos']=formula(y=y,n=4,h=6.)+formula(y=y+8.,n=4,h=6.)
            for i,p in enumerate(s['path_infos']):p['source_index']=i
            if mode=='wordmark-publication':s['lines'].append(line('DOI: original journal',[30.,60.,470.,70.],10))
            if mode=='wordmark-math':s['lines'].append(line('x = y',[200.,60.,250.,70.],11))
        if mode=='fraction-four-complex':
            s['path_infos']=[path([150.,190.,157.,198.],0),path([159.,190.,166.,198.],1),path([150.,204.,157.,212.],2),path([159.,204.,166.,212.],3),path([145.,200.,180.,201.],4)];s['path_infos'][-1]['segment_count']=2
        if mode=='two-lanes':
            s['lines']=[line('Complete original paragraph line',[x,y,x+190.,y+10.],i*2+j)for i,y in enumerate([100.,300.,500.])for j,x in enumerate([30.,280.])];s['path_infos']=formula(x=100.)+formula(x=320.)
            for i,p in enumerate(s['path_infos']):p['source_index']=i
        if mode=='two-cores-one-number':
            s['path_infos']+=formula(x=300.)
            for i,p in enumerate(s['path_infos']):p['source_index']=i
            s['lines'].append(line('(2)',[450.,200.,465.,210.],10))
        if mode in ['joined-rows','joined-rows-blocked']:
            s['path_infos']=formula(x=40.,n=11);s['path_infos'][-1]['bbox']=[139.,204.,147.,206.];s['path_infos']+=formula(x=40.,y=222.,n=37)
            for i,p in enumerate(s['path_infos']):p['source_index']=i
            if mode=='joined-rows-blocked':s['lines'].append(line('barrier',[30.,214.,470.,220.],10))
        add(mode,q)
    for i in range(70):
        q=base();s=q['source'];s['path_infos']=formula(x=rng.uniform(50.,260.),y=rng.uniform(35.,540.),n=rng.randrange(3,15),step=rng.uniform(4.,14.),h=rng.uniform(3.,28.))
        for p in s['path_infos']:p['segment_count']=rng.choice([2,7,8,12,20,30])
        rng.shuffle(s['path_infos']);add(f'geometry/{i}',q)
    existing=json.loads((ROOT/'build/migration/native-document-images.json').read_text(encoding='utf-8'))
    for c in existing['cases']:
        if 'expected'not in c:continue
        for i,p in enumerate(c['expected']['pages']):
            q=dict(source=p['source'],container_blocks=[b for b in p['fixed_blocks']if b['type']in ['table','code','image']],claimed_line_indices=p['claimed_line_indices']);add(f'continuous/{c["label"]}/{i}',q)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.formulas import classify_repeated_vector_decorations
    repeated=[];edge=next(c['expected']['vector_formula_blocks']for c in cases if c['label']=='wordmark')
    for mode in ['same','different-position','different-shape','single-page','normalized-size']:
        blocks=copy.deepcopy(edge);pages=[dict(page_size=[500.,600.],fixed_blocks=copy.deepcopy(blocks))for _ in range(2)]
        if mode=='different-position':
            for b in pages[1]['fixed_blocks']:b['bbox']=[x+20 for x in b['bbox']]
        if mode=='different-shape':
            for b in pages[1]['fixed_blocks']:b['_vector_shape'][0][0]+=1
        if mode=='single-page':pages=pages[:1]
        if mode=='normalized-size':
            pages[1]['page_size']=[1000.,1200.]
            for b in pages[1]['fixed_blocks']:b['bbox']=[2*x for x in b['bbox']]
        raw=copy.deepcopy(pages);objects=[NS(**p)for p in pages]
        # Wire arrays represent the original tuple signatures.
        for p in objects:
            for b in p.fixed_blocks:
                if '_vector_shape'in b:b['_vector_shape']=tuple(tuple(x)for x in b['_vector_shape'])
        classify_repeated_vector_decorations(objects);repeated.append(dict(label=mode,request=dict(pages=raw),expected=clean(dict(pages=[vars(p)for p in objects]))))
    refs={str(p.relative_to(ROOT)):digest(p.read_bytes())for p in [ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/formulas.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/text_roles.py']}
    write_json(target,dict(sources=refs,cases=cases,repeated=repeated));print(len(cases),'complete vector formula scenes;',sum(bool(c['expected']['vector_formula_blocks'])for c in cases),'positive;',len(repeated),'repeated document scenes')
if __name__=='__main__':main()
