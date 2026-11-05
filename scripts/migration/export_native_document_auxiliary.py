"""Freeze original contiguous document auxiliary/body sequence, before title inference."""
import ast,copy,json
from dataclasses import asdict
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_image_chain import line
from export_native_inline_clusters import clean

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import pipeline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PreparedPage,_AxisLine
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/pipeline.py'
    fn=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)and n.name=='_classify_document_text')
    stop=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='document_title_profile'for t in n.targets))
    block=ast.Module(body=copy.deepcopy(fn.body[:stop]),type_ignores=[]);ast.fix_missing_locations(block)
    pages=[];raw=copy.deepcopy(q['pages'])
    for p in raw:
        s=p.get('source',p)
        pages.append(_PreparedPage(page_size=tuple(p.get('page_size',s['page_size'])),remaining_lines=lines(p['remaining_lines']),table_bboxes=[tuple(b)for b in p.get('table_bboxes',[])],drawing_lines=[_AxisLine(bbox=tuple(r['bbox']),width=r['width'],orientation=r['orientation'])for r in p.get('drawing_lines',s.get('drawing_lines',[]))],fixed_blocks=copy.deepcopy(p.get('fixed_blocks',[])),page_footnote_groups=[set(g)for g in p.get('page_footnote_groups',[])],table_body_profile_lines=lines(p.get('table_body_profile_lines',[]))))
    scope=dict(vars(old),prepared_pages=pages);exec(compile(block,str(path),'exec'),scope)
    for p,typed in zip(raw,pages):p.update(remaining_lines=[project(l)for l in typed.remaining_lines],fixed_blocks=typed.fixed_blocks)
    def profile(p):
        if p is None:return None
        out=asdict(p);out['regular_fonts']=sorted(out['regular_fonts']);return out
    return clean(dict(pages=raw,body_profile=profile(scope['document_body_profile']),canonical_body_profile=profile(scope['canonical_body_profile'])))

def page(ls=None,**extra):return dict(page_size=[400.,600.],remaining_lines=ls or [],table_bboxes=[],drawing_lines=[],fixed_blocks=[],page_footnote_groups=[],table_body_profile_lines=[],**extra)
def body():return [dict(line('Original body paragraph words',[30.,y,370.,y+12.],i),effective_height=12.,em_height=12.,font_signature=['Body',0],font_coverage=1.,dominant_font_weight=400.)for i,y in enumerate([120.,180.,240.,300.,450.])]
def main():
    target=ROOT/'build/migration/native-document-auxiliary-canonical.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,pages):
        pages=copy.deepcopy(pages)
        for p in pages:
            p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])]
            if 'table_body_profile_lines'in p:p['table_body_profile_lines']=[project(l)for l in lines(p['table_body_profile_lines'])]
        q=dict(pages=pages);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-prepared-pages.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'expected'in c:add('prepared/'+c['label'],c['expected']['pages'])
    add('empty',[]);add('empty-page',[page()])
    for mode in ['positive','caption','large-numbered','large-ordinary','body-context','fixed-image','confirmed-table','rotated']:
        p=page(body()+[dict(line('Original running header',[30.,20.,300.,28.],20),effective_height=8.,em_height=8.)]);p['drawing_lines']=[dict(bbox=[20.,40.,380.,40.],orientation='horizontal',width=.5)]
        if mode=='caption':p['remaining_lines'][-1]['caption_start']=True
        if mode.startswith('large'):p['remaining_lines'][-1].update(effective_height=20.,em_height=20.,numbered_heading_start=mode=='large-numbered')
        if mode=='body-context':p['remaining_lines'][-1]['marginal_body_context']=True
        if mode=='fixed-image':p['fixed_blocks']=[dict(type='image',bbox=[20.,10.,380.,45.])]
        if mode=='confirmed-table':p['table_bboxes']=[[20.,10.,380.,45.]]
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            p['page_size']=[600.,400.]
            for l in p['remaining_lines']:l.update(bbox=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(p['page_size']),90)),angle=90)
            p['drawing_lines']=[dict(bbox=list(_rotate_bbox_from_upright((20.,40.,380.,40.),tuple(p['page_size']),90)),orientation='vertical',width=.5)]
        add('header/'+mode,[p])
    for mode in ['double','single','single-gap','single-large','left-lane','fixed','table','reversed']:
        p=page(body()+[dict(line('Footer original words',[90.,y,310.,y+7.],20+i),effective_height=7.,em_height=7.)for i,y in enumerate([555.,566.])]);p['drawing_lines']=[dict(bbox=[20.,y,380.,y],orientation='horizontal',width=.5)for y in ([540.,585.]if mode in ['double','reversed']else[550.])]
        if mode=='single-gap':p['remaining_lines'][-1]['bbox']=[90.,590.,310.,597.]
        if mode=='single-large':p['remaining_lines'][-1]['effective_height']=12.
        if mode=='left-lane':p['remaining_lines']+= [line('Other column body words',[320.,y,390.,y+12.],40+i)for i,y in enumerate([100.,140.,180.])]
        if mode=='fixed':p['fixed_blocks']=[dict(type='code',bbox=[20.,545.,380.,580.])]
        if mode=='table':p['table_bboxes']=[[20.,545.,380.,580.]]
        if mode=='reversed':p['drawing_lines'].reverse()
        add('footer/'+mode,[p])
    for mode in ['outer','same-row','too-tall','middle','bottom']:
        y=570. if mode=='bottom'else 250. if mode=='middle'else 30.
        p=page(body()+[dict(line('1',[190.,y,200.,y+8.],20),semantic_type='page_number'),line('Companion original',[25.,y if mode!='outer'else 10.,150.,y+8. if mode!='outer'else 18.],21)])
        p['fixed_blocks']=[dict(type='image',bbox=[250.,y,300.,y+(200. if mode=='too-tall'else 8.)])];add('companion/'+mode,[p])
    for mode in ['positive','continuation','crossing','fixed','wide','multiple']:
        p=page(body()+[dict(line('Original page note',[30.,510.,250.,518.],20),semantic_type='page_footnote',effective_height=8.),dict(line('Original trailing footer',[30.,y,180.,y+7.],21),effective_height=7.)]);p['remaining_lines'][-1]['bbox']=[30.,530. if mode=='continuation'else 550.,180.,537. if mode=='continuation'else 557.];p['page_footnote_groups']=[[20]]
        if mode=='crossing':p['remaining_lines'].append(line('Cross column body',[280.,500.,380.,530.],22))
        if mode=='fixed':p['fixed_blocks']=[dict(type='anything',bbox=[20.,540.,200.,580.])]
        if mode=='wide':p['remaining_lines'][-1]['bbox'][2]=350.
        if mode=='multiple':p['remaining_lines']+=[dict(line('Original footer piece',[30.,560.+i*8.,180.,567.+i*8.],22+i),effective_height=7.)for i in range(3)]
        add('tail/'+mode,[p])
    for label,t in [('copyright','© Original Publisher'),('capital','COPYRIGHT Original PUBLISHER'),('unicode','copyrıght Original publıſher'),('newline','© Original\nPublisher'),('no-boundary','copyright xpublisher'),('combining','copyright a\u0301publish')]:
        p=page(body()+[dict(line(t,[110.,560.,290.,568.],20),effective_height=8.)]);add('isolated/'+label,[p,page(body())])
    for mode in ['unique','repeat','short','footer','small-background']:
        p=page([dict(line('Unique background native text'if mode!='short'else'abc',[30.,30.,370.,42.],0),semantic_type='footer'if mode=='footer'else'header')]);p['fixed_blocks']=[dict(type='image',bbox=[0.,0.,400.,600. if mode!='small-background'else 300.])];add('background/'+mode,[p,p]if mode=='repeat'else[p])
    for mode in ['connected','detached','barrier','narrow','one']:
        p=page([dict(line('Original table metadata',[30.,y,370. if mode!='narrow'else 100.,y+10.],i),semantic_type='header',effective_height=10.)for i,y in enumerate([65.,80.]if mode!='one'else[80.])]);p['table_bboxes']=[[20.,100.,380.,500.]]
        if mode=='detached':p['remaining_lines'][0]['bbox']=[30.,10.,370.,20.]
        if mode=='barrier':p['drawing_lines']=[dict(bbox=[20.,94.,380.,94.],orientation='horizontal',width=.5)]
        add('preamble/'+mode,[p])
    for mode in ['cross-page','bold','italic','repair','sparse-table','three-rows','canonical','ties']:
        ps=[page(body())for _ in range(3)]
        if mode in ['bold','italic']:
            for p in ps:
                for l in p['remaining_lines']:l.update(font_signature=['Body',64 if mode=='italic'else 0],dominant_font_weight=700. if mode=='bold'else 400.)
        if mode=='repair':ps[0]['remaining_lines'][0].update(style_scale_repaired=True,em_height=8.)
        if mode=='sparse-table':
            for p in ps:p['table_body_profile_lines']=p['remaining_lines'];p['remaining_lines']=[]
        if mode=='three-rows':ps=ps[:1];ps[0]['remaining_lines']+=[dict(line('Large title',[20.,20.+i*40.,380.,40.+i*40.],20+i),effective_height=20.,em_height=20.)for i in range(2)]
        if mode=='canonical':
            for p in ps:
                for l in p['remaining_lines']:l['em_height']=9.
        if mode=='ties':
            for p in ps:
                for l in p['remaining_lines'][::2]:l['effective_height']=6.;l['em_height']=6.
        add('profile/'+mode,ps)
    write_json(target,dict(cases=cases));print(len(cases),'original contiguous document auxiliary/body sequences;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
