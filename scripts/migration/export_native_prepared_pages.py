"""Execute the original complete per-page preparation, retaining source aliases."""
import copy,json
from common import ROOT,write_json
from export_native_document_pages import reference
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_table_page_policies import empty

def main():
    p=ROOT/'build/migration/native-prepared-pages.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['prepare_page']=True;c=dict(label=label,request=q)
        try:c['expected']=reference(q,post_text=True,footnotes=True,full=True)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-footnote-document.json').read_text(encoding='utf-8'))
    for c in f['cases']:add(c['label'],c['request'])
    def wrap(ls,**extra):
        ls=[project(l)for l in lines(ls)];s=dict(empty(),page_size=[400.,600.],lines=ls,**extra)
        return dict(pages=[dict(source=s,geometry=dict(chars=[],tight_bboxes=[],origins=[],loose_bboxes=[]))])
    f=json.loads((ROOT/'build/migration/native-page-merging.json').read_text(encoding='utf-8'))
    for c in f['cases'][:22]:
        q=wrap(c['request']['lines']);q['pages'][0]['source']['page_size']=c['request']['page_size'];add('merge/'+c['label'],q)
    from export_native_image_chain import line
    def glyphs(t,x,y,weight=400.,flags=0,size=10.):return [dict(char=c,char_idx=i,bbox=[x+i*5.,y,x+i*5.+5.,y+size],tight_bbox=[x+i*5.,y,x+i*5.+5.,y+size],origin=[x+i*5.,y+size],font=dict(name='Original',size=size,flags=flags,weight=weight))for i,c in enumerate(t)]
    for mode in ['wrapped-title','regular-tail','short-body','footnote-superscript','inline-math-word','cjk-style','unmapped-ink','rotated-aside']:
        t='Label. Original body text has enough words';chars=glyphs(t,30.,100.)
        if mode in ['wrapped-title','regular-tail','short-body']:
            for c in chars[:6]:c['font']['weight']=700.
            if mode=='regular-tail':chars[0]['font']['weight']=400.
            if mode=='short-body':t='Label. Short body';chars=chars[:len(t)]
        if mode=='footnote-superscript':
            t='1Original native body words';chars=glyphs(t,30.,100.);chars[0].update(bbox=[30.,98.,34.,104.],origin=[30.,104.]);chars[0]['font']['size']=6.
        if mode=='inline-math-word':t='abc and xyz original';chars=glyphs(t,30.,100.,flags=64)
        if mode=='cjk-style':t='原始中文样式';chars=glyphs(t,30.,100.)
        ls=[dict(line(t,[30.,100.,280.,110.],0),chars=chars)]
        extra={}
        if mode=='unmapped-ink':extra['chars']=[dict(char='\x00',char_idx=99,bbox=[300.,100.,310.,130.],tight_bbox=[300.,100.,310.,130.],text_object_id=1,text_render_mode=0,font=dict(size=10.))]
        if mode=='rotated-aside':
            ls=[line('Original complete body line with enough words',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,130.,160.,190.,220.])]+[dict(line('Original side',[5.,100.,15.,220.],10),angle=90)]
        add('compact/'+mode,wrap(ls,**extra))
    runtime=json.loads((ROOT/'build/migration/native-footnote-runtime.json').read_text(encoding='utf-8'));add('live-footnote-page',runtime['request'])
    write_json(p,dict(cases=cases));print(len(cases),'complete original per-page preparations;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
