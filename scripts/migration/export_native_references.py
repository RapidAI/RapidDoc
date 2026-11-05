"""Freeze original document reference regions, numbered grouping and compact titles."""
import copy,json,sys
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def typed(p):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _PreparedPage
    return _PreparedPage(page_size=tuple(p.get('page_size',p.get('source',{}).get('page_size'))),remaining_lines=lines(p['remaining_lines']),table_bboxes=[],drawing_lines=[],fixed_blocks=copy.deepcopy(p.get('fixed_blocks',[])),page_footnote_groups=[set(g)for g in p.get('page_footnote_groups',[])],reference_regions=[tuple(b)for b in p.get('reference_regions',[])],numbered_references=p.get('numbered_references',False),bracketed_reference_numbers=p.get('bracketed_reference_numbers',False))
def dump(raw,p):
    out=copy.deepcopy(raw);out['remaining_lines']=[project(l)for l in p.remaining_lines]
    for key in ['reference_regions','numbered_references','bracketed_reference_numbers','page_footnote_groups']:
        value=getattr(p,key)
        if key in out or value:out[key]=clean(value)
    return out
def reference(q,group_only=False):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import mark_document_reference_regions,group_reference_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.compact_titles import _classify_small_emphasized_titles
    if group_only:
        p=typed(q['page']);group_reference_lines(p.remaining_lines,p);return dump(q['page'],p)
    pages=[typed(p)for p in q['pages']];mark_document_reference_regions(pages)
    for i,p in enumerate(pages):
        if p.numbered_references:group_reference_lines(p.remaining_lines,p)
        _classify_small_emphasized_titles(p,i)
    return clean(dict(pages=[dump(p,t)for p,t in zip(q['pages'],pages)]))
def page(ls,**kw):return dict(dict(page_size=[400.,600.],remaining_lines=ls,fixed_blocks=[],page_footnote_groups=[],reference_regions=[],numbered_references=False,bracketed_reference_numbers=False),**kw)
def numbered(count=4,bracket=True,start=1,x=30.,y=160.,step=30.):
    return [line(f'[{i+start}] Original reference author and source'if bracket else f'{i+start}. Original reference author and source',[x,y+i*step,x+150.,y+i*step+10.],i)for i in range(count)]
def main():
    if '--canonical-document'in sys.argv:
        target=ROOT/'build/migration/native-reference-document-canonical.json'
        if target.exists():raise FileExistsError(target)
        frozen=json.loads((ROOT/'build/migration/native-reference-document.json').read_text(encoding='utf-8'))
        from loguru import logger
        logger.remove()
        for c in frozen['cases']:
            if c['label']=='sequence/two-columns':
                p=c['request']['pages'][1]
                for i,l in enumerate(p['remaining_lines']):l['source_index']=i
            c['expected']=reference(c['request'])
        write_json(target,frozen);print(len(frozen['cases']),'canonical original document reference sequences');return
    targets=[ROOT/f'build/migration/{n}.json'for n in ['native-reference-document','native-reference-groups','native-reference-text']]
    if any(p.exists()for p in targets):raise FileExistsError('immutable reference fixtures exist')
    from loguru import logger
    logger.remove();cases=[];group_cases=[]
    def add(label,q,group=False):
        q=copy.deepcopy(q)
        for p in ([q['page']]if group else q['pages']):p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])]
        c=dict(label=label,request=q)
        try:c['expected']=reference(q,group)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        (group_cases if group else cases).append(c)
    f=json.loads((ROOT/'build/migration/native-document-titles.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'expected'in c:add('titles/'+c['label'],dict(pages=c['expected']['pages']))
    for mode in ['bracket','dot','heading-dot','heading-single','only-two','gap-numbers','same-number','unicode-digit','fullwidth','footer-number','page-note','heading-in-note','two-columns','appendix','structural-end','tight-structural','translated-end','preceding-continuation','long-numeric','pure-label']:
        ls=numbered()
        if mode in ['dot','heading-dot']:ls=numbered(bracket=False)
        if mode.startswith('heading')or mode=='heading-in-note':ls.insert(0,line('References',[30.,130.,180.,140.],20))
        if mode=='heading-single':ls=ls[:2]
        if mode=='only-two':ls=ls[:2]
        if mode=='gap-numbers':ls[-1]['text']='[9] Original reference'
        if mode=='same-number':
            for l in ls:l['text']='[1] Original reference'
        if mode=='unicode-digit':
            for i,l in enumerate(ls):l['text']=f'[{chr(0x661+i)}] Original reference'
        if mode=='fullwidth':
            for i,l in enumerate(ls):l['text']=f'［{i+1}］ Original reference'
        if mode=='footer-number':
            for l in ls:l['semantic_type']='footer'
        if mode=='page-note':
            for l in ls:l['semantic_type']='page_footnote'
        if mode=='heading-in-note':ls[0]['semantic_type']='page_footnote'
        if mode=='two-columns':ls+=numbered(count=4,start=5,x=220.,y=160.);[l.update(source_index=10+i)for i,l in enumerate(ls[4:])]
        if mode in ['appendix','structural-end','tight-structural']:ls.append(line('Appendix A'if mode=='appendix'else'Original next section',[30.,281. if mode=='tight-structural'else 320.,180.,291. if mode=='tight-structural'else 330.],20,semantic_type='paragraph_title',structural_title=True))
        if mode=='translated-end':ls+=[line('Original translated scientific document title',[40.,320.,360.,340.],20),line('Abstract: original',[30.,380.,370.,390.],21)]
        if mode=='preceding-continuation':ls+=[line('Original continuation above second page',[40.,130.,180.,140.],20)]
        if mode=='long-numeric':ls[-1]['text']='[12345] Original reference'
        if mode=='pure-label':
            for i,l in enumerate(ls):l['text']=f'[{i+1}]'
        p=page(ls)
        if mode in ['heading-in-note','page-note']:p['page_footnote_groups']=[[l['source_index']for l in ls]]
        ps=[p]
        if mode=='preceding-continuation':ps=[page(numbered(start=1)),page(numbered(count=2,start=5)+[line('Original continuation above next item',[40.,130.,180.,140.],20)])]
        add('numbered/'+mode,dict(pages=ps))
    for mode in ['carry','numbered-stop','author-stop','appendix-same-page','appendix-next-page','empty-page','two-columns','inline-translation']:
        first=page([line('References',[30.,100.,180.,110.],20)]+numbered())
        second=page(numbered(count=2,start=5))
        if mode in ['author-stop','appendix-same-page']:first=page([line('References',[30.,100.,180.,110.],20)]+[line('Original author bibliography text',[30.,y,370.,y+10.],i)for i,y in enumerate([130.,145.,160.,175.])])
        if mode in ['numbered-stop','author-stop']:second=page([line('Original ordinary body text',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,115.,130.,145.])])
        if mode=='appendix-same-page':first['remaining_lines'].append(line('Appendix A',[30.,250.,180.,260.],30,semantic_type='paragraph_title',structural_title=True))
        if mode=='appendix-next-page':second['remaining_lines'].append(line('Appendix A',[30.,250.,180.,260.],30))
        if mode=='two-columns':second['remaining_lines']+=[dict(l,source_index=20+i)for i,l in enumerate(numbered(count=2,start=7,x=220.))]
        if mode=='inline-translation':second['remaining_lines']+=[line('Original translated scientific title here',[40.,300.,360.,310.],40),line('摘要：原始摘要',[30.,350.,370.,360.],41)]
        ps=[first,second]
        if mode=='empty-page':ps.insert(1,page([]))
        add('sequence/'+mode,dict(pages=ps))
    for mode in ['positive','no-indent','no-years','no-authors','fourteen','container','heading','appendix']:
        ls=[]
        for i in range(5):
            ls+=[line(f'Original, A, B. 202{i}. Native source',[30.,100.+i*60.,370.,110.+i*60.],i*3),line('Original hanging continuation',[40.,115.+i*60.,370.,125.+i*60.],i*3+1),line('Original hanging final continuation',[40.,130.+i*60.,370.,140.+i*60.],i*3+2)]
        if mode=='no-indent':
            for l in ls:l['bbox'][0]=30.
        if mode=='no-years':
            for l in ls:l['text']=l['text'].replace('202','year')
        if mode=='no-authors':
            for l in ls:l['text']=l['text'].replace('Original, A, B.','Original source')
        if mode=='fourteen':ls=ls[:-1]
        if mode=='heading':ls.insert(0,line('参 考 文 献',[30.,60.,180.,70.],20))
        if mode=='appendix':ls.append(line('Appendix A',[30.,420.,180.,430.],20))
        p=page(ls,fixed_blocks=[dict(type='image',bbox=[20.,80.,380.,410.])]if mode=='container'else[]);add('author-year/'+mode,dict(pages=[p]));p['reference_regions']=[[0.,80.,400.,410.]];add('author-group/'+mode,dict(page=p),True)
    for mode in ['bracket','dot','continued-dotted-page','biography','publisher','author-info','editor','punctuation','wide','same-row','overlap','existing-group','page-footnote','duplicate-region','empty','null-effective']:
        ls=numbered(bracket=mode!='dot');ls+=[line('Original hanging continuation',[40.,175.,180.,185.],20),line('Original hanging continuation',[40.,205.,180.,215.],21)]
        p=page(ls,numbered_references=True,bracketed_reference_numbers=mode!='dot',reference_regions=[[20.,130.,200.,400.]],page_footnote_groups=[[0,1,20,99]])
        if mode=='continued-dotted-page':ls[4]['text']='123. Original pages continuation'
        if mode in ['biography','publisher','author-info','editor']:ls[2]['text']={'biography':'Original (1990—) 博士研究','publisher':'Publisher’s note: Original','author-info':'Author information: Original','editor':'［责任编辑：原始'}[mode]
        if mode=='punctuation':ls[-1]['text']='---'
        if mode=='wide':ls[-1]['bbox'][2]=390.
        if mode=='same-row':ls[-1]['bbox']=[100.,175.,180.,185.]
        if mode=='overlap':p['reference_regions']=[[20.,130.,200.,220.],[20.,190.,200.,400.]]
        if mode=='existing-group':ls[0]['paragraph_group']=99
        if mode=='page-footnote':ls[0]['semantic_type']='page_footnote'
        if mode=='duplicate-region':p['reference_regions']*=2
        if mode=='empty':p['remaining_lines']=[]
        if mode=='null-effective':ls[0]['effective_height']=None
        add('group/'+mode,dict(page=p),True)
    for mode in ['positive','first-page','regular-weight','tiny','large','caption','container','no-body','grouped','far','top']:
        ls=[line('Original preceding body words',[30.,150.,370.,160.],0,h=12.,weight=400.)]+[line('Original small heading',[30.,174.,180.,183.],1,h=9.,font='Heading',weight=700.)]+[line('Original stable body paragraph',[30.,y,370.,y+12.],2+i,h=12.,weight=400.)for i,y in enumerate([190.,208.,226.,244.])]
        if mode=='regular-weight':ls[1]['dominant_font_weight']=400.
        if mode=='tiny':ls[1]['effective_height']=6.
        if mode=='large':ls[1]['effective_height']=12.
        if mode=='caption':ls[1]['caption_start']=True
        if mode=='grouped':ls[1]['paragraph_group']=1
        if mode=='far':ls[1]['bbox']=[30.,165.,180.,174.]
        if mode=='no-body':ls=ls[:2]
        if mode=='top':ls=ls[1:]
        p=page(ls,fixed_blocks=[dict(type='image',bbox=[20.,170.,200.,185.])]if mode=='container'else[])
        add('compact/'+mode,dict(pages=[p]if mode=='first-page'else[page([]),p]))
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import _REFERENCE_NUMBER,_is_reference_heading
    strings=['References','参 考 文 献','［References］','参考文献(references)','reference','BIBLIOGRAPHY','Referenceſ','[1] text','[1]','[1] ','[1]1','[1] 1','1. text','1234. text','［١٢］ 原始','[１２３４] text','[12345] text','1.2 text','0. text','[0] text','\u001c[1] native','[1］ text']
    text_cases=[]
    for t in strings:
        m=_REFERENCE_NUMBER.match(t);text_cases.append(dict(text=t,heading=_is_reference_heading(t),number=None if m is None else [int(m.group(1)or m.group(2)),bool(m.group(1))]))
    for dest,data in zip(targets,[dict(cases=cases),dict(cases=group_cases),dict(cases=text_cases)]):write_json(dest,data)
    pos=next(c for c in cases if c['label']=='numbered/bracket');assert pos['expected']['pages'][0]['numbered_references']
    pos=next(c for c in cases if c['label']=='compact/positive');assert pos['expected']['pages'][1]['remaining_lines'][1]['semantic_type']=='paragraph_title'
    print(len(cases),'document reference/compact sequences;',len(group_cases),'group cases;',len(text_cases),'text contracts')
if __name__=='__main__':main()
