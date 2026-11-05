"""Freeze the original post-geometry split/reclaim statements, without rewriting them."""
import ast,copy,json
from dataclasses import fields
from common import ROOT,digest,write_json
from export_form_runtime import typed
from export_native_image_chain import line
from export_native_table_page_policies import empty
from export_native_inline_clusters import clean

def project(l):return clean({f.name:getattr(l,f.name) for f in fields(l) if not f.name.startswith('_')})

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.graphics import _split_parallel_graphic_rule_rows
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py'
    text=path.read_text(encoding='utf-8');tree=ast.parse(text)
    # Execute exactly the contiguous original statements: image_containers through remaining_lines filtering.
    fn=next(n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='image_containers' for t in x.targets) for x in n.body))
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='image_containers' for t in n.targets))
    end=next(i for i in range(start,len(fn.body)) if isinstance(fn.body[i],ast.ImportFrom)and fn.body[i].module=='formulas' and any(x.name=='restore_leading_inline_fractions'for x in fn.body[i].names))
    body=ast.Module(body=copy.deepcopy(fn.body[start:end]),type_ignores=[]);ast.fix_missing_locations(body)
    source=typed(q['source']);ls=typed(dict(q['source'],lines=q['remaining_lines'])).lines
    state={k:copy.deepcopy(q.get(k,[])) for k in ['caption_graphics','form_image_blocks','graphic_blocks','raster_image_blocks']}
    image_boxes=[tuple(b['bbox'])for k in ['form_image_blocks','graphic_blocks','raster_image_blocks'] for b in state[k]]
    ls=_split_parallel_graphic_rule_rows(ls,source.drawing_lines,image_boxes,[tuple(b)for b in q['table_bboxes']],source.page_size,source_index_start=q.get('source_index_start'))
    state.update(source=source,remaining_lines=ls,strong_graphic_bboxes=[tuple(b)for b in q['strong_graphic_bboxes']])
    scope=dict(vars(old));scope.update(state);exec(compile(body,str(path),'exec'),scope)
    return clean(dict(remaining_lines=[project(l)for l in scope['remaining_lines']],image_containers=scope['image_containers'],reclaimed_line_indices=sorted(scope['reclaimed'])))

def base():
    s=empty();s['page_size']=[400.,600.]
    return dict(source=s,remaining_lines=[],caption_graphics=[],form_image_blocks=[],graphic_blocks=[],raster_image_blocks=[],table_bboxes=[],strong_graphic_bboxes=[],source_index_start=100)

def heading(i=0,y=130.):
    text='Left heading Right heading';chars=[]
    for j,c in enumerate(text):
        x=40.+j*8 if j<13 else 215.+(j-13)*8
        chars.append(dict(char=c,char_idx=j,bbox=[x,y,x+8.,y+10.],font=dict(name='Body',flags=0,size=10.)))
    return dict(line(text,[40.,y,319.,y+10.],i),chars=chars,visual_row_id=7,run_index=0)

def main():
    target=ROOT/'build/migration/native-late-image-text.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):
        # Serialize the typed input boundary, including frozen original source_bbox/marker features.
        q=copy.deepcopy(q);q['remaining_lines']=[project(l)for l in typed(dict(q['source'],lines=q['remaining_lines'])).lines]
        cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['split','existing-two-runs','source-start','table-block','wrong-angle','no-row','mixed-semantic','narrow-gap','weak-glyphs','no-rule','no-image','white-space','source-bbox','font-weight','three-images']:
        q=base();s=q['source'];s['drawing_lines']=[dict(bbox=b,orientation='horizontal',width=.5)for b in [[30.,145.,190.,145.],[210.,145.,370.,145.]]]
        q['graphic_blocks']=[dict(type='image',bbox=b,angle=0,content='original')for b in [[30.,155.,190.,350.],[210.,155.,370.,350.]]];l=heading();q['remaining_lines']=[l]
        if mode=='existing-two-runs':
            a=copy.deepcopy(l);b=copy.deepcopy(l);a.update(text='Left heading',bbox=[40.,130.,136.,140.],chars=l['chars'][:13],source_index=9);b.update(text='Right heading',bbox=[215.,130.,319.,140.],chars=l['chars'][13:],source_index=3,run_index=1);q['remaining_lines']=[b,a]
        if mode=='source-start':q['source_index_start']=1000
        if mode=='table-block':q['table_bboxes']=[[20.,125.,380.,146.]]
        if mode=='wrong-angle':l['angle']=90
        if mode=='no-row':l['visual_row_id']=None
        if mode=='mixed-semantic':a=copy.deepcopy(l);a.update(source_index=1,semantic_type='caption');q['remaining_lines'].append(a)
        if mode=='narrow-gap':s['drawing_lines'][1]['bbox'][0]=192.
        if mode=='weak-glyphs':l['chars']=l['chars'][:2]+l['chars'][-2:]
        if mode=='no-rule':s['drawing_lines']=[]
        if mode=='no-image':q['graphic_blocks']=[]
        if mode=='white-space':l['chars'][2]['char']='\u00ad';l['chars'][3]['char']='\t';l['chars'][4]['char']='\r';l['chars'][5]['char']='\u200b'
        if mode=='source-bbox':l['source_bbox']=[1.,2.,3.,4.]
        if mode=='font-weight':
            for j,c in enumerate(l['chars']):c['font'].update(name='Bold'if j<5 else'Body',weight=700 if j<5 else 400)
        if mode=='three-images':
            q['source']['page_size']=[650.,600.];s['drawing_lines'].append(dict(bbox=[390.,145.,550.,145.],orientation='horizontal',width=.5));q['graphic_blocks'].append(dict(type='image',bbox=[390.,155.,550.,350.],angle=0,content='original'));l['text']+=' Third heading'
            for j,c in enumerate(' Third heading'):l['chars'].append(dict(char=c,char_idx=26+j,bbox=[395.+8*j,130.,403.+8*j,140.],font=dict(name='Body',flags=0,size=10.)))
            l['bbox'][2]=507.
        add(mode,q)
    for mode in ['reclaim','smallest-owner','tie-first','semantic','caption','ink','background','form-denied','strong-override','weak-core','numeric','rotated-numeric','original-empty','side-card']:
        q=base();q['graphic_blocks']=[dict(type='image',bbox=[30.,100.,370.,350.],angle=0,content='original')];q['remaining_lines']=[line('native label',[50.,140.,130.,150.],0)]
        if mode in ['smallest-owner','tie-first']:q['form_image_blocks']=[dict(type='image',bbox=[40.,130.,140.,160.]if mode=='smallest-owner'else[30.,100.,370.,350.],angle=0,content='first')]
        if mode=='semantic':q['remaining_lines'][0]['semantic_type']='paragraph_title'
        if mode=='caption':q['remaining_lines'][0]['text']='Figure 3: preserve caption'
        if mode=='ink':q['remaining_lines'][0].update(bbox=[0.,0.,400.,600.],ink_bbox=[50.,140.,130.,150.])
        if mode=='background':q['graphic_blocks'][0]['bbox']=[0.,0.,400.,600.];q['source']['image_bboxes']=[[0.,0.,400.,600.]]
        if mode in ['form-denied','strong-override','weak-core']:
            q['source']['form_member_sources']=[dict(bbox=[30.,100.,370.,350.],source_indices=[999])];q['remaining_lines'][0]['chars']=[dict(char='X',bbox=[50.,140.,60.,150.],char_idx=0)]
            if mode!='form-denied':q['strong_graphic_bboxes']=[[30.,100.,370.,300.]if mode=='strong-override'else[30.,100.,40.,110.]]
        if mode in ['numeric','rotated-numeric']:
            q['remaining_lines']=[line(str(i),[50.+(i%3)*100,140.+(i//3)*30,60.+(i%3)*100,150.+(i//3)*30],i)for i in [5,0,4,1,3,2]]
            if mode=='rotated-numeric':q['remaining_lines'][0]['angle']=90
        if mode=='original-empty':q['graphic_blocks'][0]['content']=''
        if mode=='side-card':q['caption_graphics']=q['graphic_blocks'];q['graphic_blocks']=[];q['caption_graphics'][0]['type']='paragraph'
        add(mode,q)
    existing=json.loads((ROOT/'build/migration/native-document-images.json').read_text(encoding='utf-8'))
    for c in existing['cases']:
        if 'expected'not in c:continue
        for i,page in enumerate(c['expected']['pages']):
            q=base();q.update({k:copy.deepcopy(page[k])for k in ['source','remaining_lines','caption_graphics','form_image_blocks','graphic_blocks','raster_image_blocks','table_bboxes','strong_graphic_bboxes']});add(f'continuous/{c["label"]}/{i}',q)
    refs={str(p.relative_to(ROOT)):digest(p.read_bytes())for p in [ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/graphics.py',ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py']}
    write_json(target,dict(sources=refs,cases=cases));print(len(cases),'original split/reclaim continuations')
if __name__=='__main__':main()
