"""Freeze original post-container classifiers, retaining reference object aliases."""
import copy,json
from common import ROOT,write_json,digest
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_image_chain import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _mark_native_caption_starts
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.native_boundaries import classify_bilingual_term_bands,classify_appendix_title_bands
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.title_analysis.structural import _mark_emphasized_quote_prose,_classify_native_display_resets
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly.continuity import group_native_inline_formula_prose
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_cjk_bands import restore_native_cjk_bands
    ls=lines(q['lines']);refs=lines(q['reference_lines'])
    for a in q.get('reference_aliases',[]):refs[a['reference_index']]=ls[a['line_index']]
    _mark_native_caption_starts(ls);classify_bilingual_term_bands(ls);classify_appendix_title_bands(ls,tuple(q['page_size']));_mark_emphasized_quote_prose(ls)
    group_native_inline_formula_prose(ls,[(tuple(r['bbox']),r['source_indices'])for r in q['inline_regions']])
    _classify_native_display_resets(ls,tuple(q['page_size']),[tuple(b)for b in q['visual_bboxes']],[tuple(b)for b in q['table_bboxes']],page_index=q['page_index'],reference_lines=refs)
    restore_native_cjk_bands(ls,tuple(q['page_size']),[tuple(b)for b in q['visual_bboxes']+q['table_bboxes']],q['page_index'])
    return [project(l)for l in ls]

def scenes():
    out=[]
    def item(t,b,i=0,h=10.,font='Body',weight=400.,flags=0,**kw):
        l=line(t,list(b),i,font_coverage=1.,dominant_font_weight=weight,**kw);l.update(effective_height=h,font_signature=[font,flags]);return l
    def add(label,ls,**kw):
        ls=[project(l)for l in lines(ls)];q=dict(lines=ls,page_size=[400.,600.],reference_lines=[],reference_aliases=[],page_index=0,visual_bboxes=[],table_bboxes=[],inline_regions=[]);q.update(kw)
        if 'reference_lines'not in kw:q['reference_lines']=copy.deepcopy(ls);q['reference_aliases']=[dict(reference_index=i,line_index=i)for i in range(len(ls))]
        q['reference_lines']=[project(l)for l in lines(q['reference_lines'])];out.append(dict(label=label,request=q))
    for t in ['Fig. 1: Original caption','Figure 1 shows the result','图１ 测试标题','Table A.2 Original','表 ３、测试','Fig. 2 (a), (b)','Listing 12 Code','Algorithm IV: Title','Chart 1 illustrates data','Figure 2 and 3','Ｆｉｇｕｒｅ １: Title','Diagram 2 Title','Figure 2. This title','Figure 2 lower case','图 一 中文标题','Fig 1\nOriginal','Fig 1\u001cTitle']:
        add('caption/'+t,[item(t,[30.,100.,280.,110.])])
    for mode in ['positive','no-definition','no-english','no-chinese','punctuation','next-number','preceding','rotated','semantic','split','digit-unicode','long-title']:
        ls=[item('2.1',[30.,100.,45.,110.]),item('中文术语 Original terminology',[50.,100.,280.,110.],1),item('这是完整的中文术语定义内容',[30.,120.,300.,130.],2)]
        if mode=='no-definition':ls=ls[:2]
        if mode=='no-english':ls[1]['text']='中文术语'
        if mode=='no-chinese':ls[1]['text']='Original terminology'
        if mode=='punctuation':ls[1]['text']+='：'
        if mode=='next-number':ls += [item('2.2 Next item',[30.,140.,280.,150.],3),item('后续条目原文应当保持不被抑制',[30.,160.,300.,170.],4)]
        if mode=='preceding':ls.insert(0,item('前一条目的末行',[30.,80.,280.,90.],3))
        if mode=='rotated':ls[0]['angle']=90
        if mode=='semantic':ls[0]['semantic_type']='text'
        if mode=='split':ls[1]['text']='中文术语';ls.insert(2,item('Original terminology',[130.,100.,280.,110.],3))
        if mode=='digit-unicode':ls[0]['text']='٣.١'
        if mode=='long-title':ls[1]['text']='中文术语 '+'Original terminology '*12
        add('bilingual/'+mode,ls)
    for mode in ['positive','no-section','one-member','offset','wide','rotated','caption','empty-members-error','non-ascii-digit']:
        ls=[item('附 录 A',[170.,100.,230.,110.]),item('（规范性附录）',[160.,120.,240.,130.],1),item('附录的原始名称',[160.,140.,240.,150.],2),item('A.1 原始章节',[30.,170.,270.,180.],3)]
        if mode=='no-section':ls=ls[:-1]
        if mode=='one-member':ls=[ls[0],ls[-1]]
        if mode=='offset':ls[1]['bbox'][0]-=50.;ls[1]['bbox'][2]-=50.;ls[2]['bbox'][0]-=50.;ls[2]['bbox'][2]-=50.
        if mode=='wide':ls[0]['bbox']=[80.,100.,320.,110.];ls=ls[:1]+[ls[-1]]
        if mode=='rotated':ls[0]['angle']=90
        if mode=='caption':ls[1]['semantic_type']='caption'
        if mode=='empty-members-error':ls[0]['bbox']=[0.,100.,400.,110.];ls=ls[:1]+[ls[-1]]
        if mode=='non-ascii-digit':ls[-1]['text']='A.١ 原始章节'
        add('appendix/'+mode,ls)
    for mode in ['italic','bold-six','bold-three','no-opening','too-few-words','math','font-break','gap','intro','short-intro','row-fragments','existing-group','caption','rotated','terminal-newline']:
        n=6 if mode=='bold-six'else 3;ls=[]
        for i in range(n):ls.append(item(('“'if i==0 else'')+'These original natural language quotation words remain native text'+('”'if i==n-1 else''),[30.,100.+i*16.,330.,110.+i*16.],i,font='Quote',flags=0 if mode.startswith('bold')else 64,weight=700. if mode.startswith('bold')else 400.,visual_row_id=i))
        if mode=='no-opening':ls[0]['text']=ls[0]['text'][1:]
        if mode=='too-few-words':
            for l in ls:l['text']='“short quote”'
        if mode=='math':ls[1]['text']+=' = x'
        if mode=='font-break':ls[1]['font_signature']=['Other',64]
        if mode=='gap':ls[1]['bbox'][1]+=20.;ls[1]['bbox'][3]+=20.
        if mode in ['intro','short-intro']:ls.insert(0,item('These six original introductory words explain the quotation:'if mode=='intro'else'Quote:',[30.,80.,330.,90.],10))
        if mode=='row-fragments':
            for i,l in list(enumerate(ls)):
                l['bbox'][2]=180.;ls.append(item('additional quoted original native words',[185.,l['bbox'][1],330.,l['bbox'][3]],i+10,font='Quote',flags=64,visual_row_id=i))
        if mode=='existing-group':ls.append(item('previous text',[30.,40.,330.,50.],10,paragraph_group=77))
        if mode=='caption':ls[1]['caption_start']=True
        if mode=='rotated':ls[1]['angle']=90
        if mode=='terminal-newline':ls[-1]['text']+='\n'
        add('quote/'+mode,ls)
    for mode in ['positive','one-region','two-rows','one-hit','font-break','gap','short','grouped','same-row','slot-width','outside-source']:
        ls=[item('These original native prose words have complete useful meaning',[30.,100.+i*16.,330.,110.+i*16.],i,visual_row_id=i)for i in range(3)];regions=[dict(bbox=[160.,100.,180.,110.],source_indices=[0]),dict(bbox=[160.,116.,180.,126.],source_indices=[1])]
        if mode=='one-region':regions=regions[:1]
        if mode=='two-rows':ls=ls[:2]
        if mode=='one-hit':regions[-1]['source_indices']=[0]
        if mode=='font-break':ls[1]['font_signature']=['Other',0]
        if mode=='gap':ls[1]['bbox'][1]+=30.;ls[1]['bbox'][3]+=30.
        if mode=='short':
            for l in ls:l['text']='short text'
        if mode=='grouped':ls[1]['paragraph_group']=7
        if mode=='same-row':ls[1]['visual_row_id']=0
        if mode=='slot-width':ls[0]['bbox'][2]=130.;ls[1]['bbox'][0]=190.;regions[0]['bbox']=[125.,100.,330.,110.];regions[1]['bbox']=[30.,116.,195.,126.]
        if mode=='outside-source':regions[-1]['source_indices']=[99]
        add('inline/'+mode,ls,inline_regions=regions)
    def body():return [item('Original complete native body paragraph has enough words',[30.,y,340.,y+10.],100+i)for i,y in enumerate([140.,156.,172.,230.,246.,262.])]
    for mode in ['light','one-light','light-used-in-body','no-followers','above-image','table-title','table-offset','large-repeated','one-band','mixed-case','titlecase','chinese-page-zero','chinese-later','colon-labels','two-colon-labels','alias','detached']:
        ls=body();kw={}
        if mode in ['table-title','table-offset']:
            ls += [item('Original table title',[125.,100.,245.,115.],0,h=15.,font='Heading',weight=700.)];kw['table_bboxes']=[[20.,130.,360.,300.]]
            if mode=='table-offset':ls[-1]['bbox'][0]-=50.;ls[-1]['bbox'][2]-=50.
        elif mode in ['large-repeated','one-band','mixed-case','titlecase','chinese-page-zero','chinese-later']:
            ls += [item(t,[30.,y,250.,y+16.],i,h=16.,font='Heading',weight=700.)for i,(t,y)in enumerate([('ORIGINAL LARGE',50.),('TITLE BAND',70.),('SECOND LARGE',300.),('DISPLAY BAND',320.)])]
            if mode=='one-band':ls=ls[:-2]
            if mode=='mixed-case':ls[-1]['text']='Display band'
            if mode=='titlecase':ls[-1]['text']='DISPLAY ǅ BAND'
            if mode.startswith('chinese'):
                ls=ls[:-4]+[item('中文章节：',[30.,50.,250.,66.],0,h=16.,font='Heading',weight=400.),item('原始跨行名称',[30.,70.,250.,86.],1,h=16.,font='Heading',weight=400.)];kw['page_index']=1 if mode=='chinese-later'else 0
        elif mode in ['colon-labels','two-colon-labels']:
            ls=[]
            for i in range(2 if mode=='two-colon-labels'else 3):ls += [item('Original repeated label:',[30.,60.+i*100,230.,70.+i*100],i,font='Label',weight=500.),item('First original body paragraph has enough natural words',[30.,80.+i*100,340.,90.+i*100],100+i*2),item('Second original body paragraph has enough natural words',[30.,96.+i*100,340.,106.+i*100],101+i*2)]
        else:
            ls += [item('Original light heading',[30.,110.,230.,120.],0,font='Light',weight=300.),item('Second light heading',[30.,200.,230.,210.],1,font='Light',weight=300.)]
            if mode=='one-light':ls=ls[:-1]
            if mode=='light-used-in-body':ls[0]['text']='This original native body paragraph has at least twelve complete natural language words';ls[0]['font_signature']=['Light',0]
            if mode=='no-followers':
                for l in ls[:6]:l['bbox'][0]+=40.
            if mode=='above-image':kw['visual_bboxes']=[[20.,125.,280.,190.]];ls[-2]['bbox']=[50.,110.,250.,120.]
            if mode in ['alias','detached']:kw['reference_lines']=copy.deepcopy(ls);kw['reference_aliases']=[dict(reference_index=i,line_index=i)for i in range(len(ls))]if mode=='alias'else[]
        add('display/'+mode,ls,**kw)
    for mode in ['body','number-heading','display-heading','no-tight','healthy','few-chars','font-break','short-tail','wide-gap','container','later-doc-title']:
        ls=[];text='原始中文正文用于验证真实字形行距与自然段持续边界'
        for i in range(4):
            t='短末行'if mode=='short-tail'and i==3 else text;y=150.+i*20.;chars=[dict(char=c,char_idx=i*100+j,bbox=[30.+j*10.,y-10.,40.+j*10.,y+20.],tight_bbox=[30.+j*10.,y,40.+j*10.,y+10.],font=dict(name='Body',size=10.,flags=0,weight=400.))for j,c in enumerate(t)]
            ls.append(item(t,[30.,y-10.,30.+len(t)*10.,y+20.],i,chars_override=chars))
            ls[-1]['chars']=ls[-1].pop('chars_override')
        if mode in ['number-heading','display-heading','later-doc-title']:ls.insert(0,item('一、原始章节标题'if mode=='number-heading'else'原始中文跨行章节标题原始中文跨行章节标题原始中文跨行章节标题',[30.,100.,250.,120.],10,font='Heading',weight=700. if mode=='display-heading'else 400.,**dict(semantic_type='doc_title')if mode=='later-doc-title'else{}))
        if mode=='no-tight':
            for l in ls:
                for c in l['chars']:c.pop('tight_bbox',None)
        if mode=='healthy':
            for l in ls:
                for c in l['chars']:c['bbox'][1]+=10.;c['bbox'][3]-=10.
        if mode=='few-chars':
            for l in ls:l['chars']=l['chars'][:4]
        if mode=='font-break':ls[1]['font_signature']=['Other',0]
        if mode=='wide-gap':ls[1]['bbox'][1]+=40.;ls[1]['bbox'][3]+=40.;ls[1]['chars']=[]
        add('cjk/'+mode,ls,page_index=1 if mode=='later-doc-title'else 0,visual_bboxes=[[20.,90.,300.,140.]]if mode=='container'else [])
    return out

def main():
    p=ROOT/'build/migration/native-text-roles.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=scenes()
    for c in cases:
        try:c['expected']=reference(c['request'])
        except ValueError as e:c.update(error='invalid_input',old_exception=type(e).__name__)
    write_json(p,dict(cases=cases));print(len(cases),'post-container classification scenes;',sum(c.get('expected')!=c['request']['lines']for c in cases if 'expected'in c),'mutating scenes;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
