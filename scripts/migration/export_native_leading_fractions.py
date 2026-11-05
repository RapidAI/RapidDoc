"""Original leading fraction restoration with complete line/character state."""
import copy,json
from common import ROOT,write_json,digest
from export_form_runtime import typed
from export_native_late_image_text import project
from export_native_image_chain import line
from export_native_table_page_policies import empty

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.formulas import restore_leading_inline_fractions
    s=typed(dict(empty(),page_size=q['page_size'],lines=q['lines'],drawing_lines=q['drawing_lines']))
    return [project(l)for l in restore_leading_inline_fractions(s.lines,s.drawing_lines,s.page_size)]
def base():
    preceding=line('An original preceding paragraph with enough words',[50.,184.,350.,194.],0)
    numerator=line('β',[50.,200.,58.,208.],1);numerator['effective_height']=8.
    denominator=line('α表示原生文本的变量',[50.,210.,200.,220.],2);denominator['chars']=[dict(char=c,char_idx=i,bbox=[50.+i*8,210.,58.+i*8,220.],tight_bbox=[50.+i*8,210.,58.+i*8,220.],font=dict(name='Body',size=10.,flags=0))for i,c in enumerate(denominator['text'])]
    return dict(lines=[preceding,numerator,denominator],drawing_lines=[dict(bbox=[49.,209.,59.,209.],orientation='horizontal',width=.5)],page_size=[400.,600.])
def main():
    target=ROOT/'build/migration/native-leading-fractions.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['lines']=[project(l)for l in typed(dict(empty(),page_size=q['page_size'],lines=q['lines'])).lines];cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['restore','no-rules','no-preceding','short-preceding','far-preceding','wrong-indent','semantic-numerator','semantic-denominator','rotated','no-greek','mixed-numerator','large-numerator','few-cjk','wrong-prefix','no-chars','tight-box','loose-fallback','rule-too-wide','rule-too-short','vertical-rule','rule-above','rule-below','two-denominators','source-order','inline-regions','structural-membership','char-before-space']:
        q=base();p,n,d=q['lines']
        if mode=='no-rules':q['drawing_lines']=[]
        if mode=='no-preceding':q['lines']=q['lines'][1:]
        if mode=='short-preceding':p['text']='short'
        if mode=='far-preceding':p['bbox']=[50.,150.,350.,160.]
        if mode=='wrong-indent':p['bbox']=[80.,184.,350.,194.]
        if mode=='semantic-numerator':n['semantic_type']='caption'
        if mode=='semantic-denominator':d['semantic_type']='text'
        if mode=='rotated':n['angle']=90
        if mode=='no-greek':n['text']='2'
        if mode=='mixed-numerator':n['text']='β+２'
        if mode=='large-numerator':n['text']='β'*9
        if mode=='few-cjk':d['text']='α中文'
        if mode=='wrong-prefix':d['text']='正文α表示原生文本的变量'
        if mode=='no-chars':d['chars']=[]
        if mode=='tight-box':d['chars'][0]['tight_bbox']=[90.,210.,98.,220.]
        if mode=='loose-fallback':d['chars'][0]['tight_bbox']=None
        if mode=='rule-too-wide':q['drawing_lines'][0]['bbox'][2]=90.
        if mode=='rule-too-short':q['drawing_lines'][0]['bbox'][2]=52.
        if mode=='vertical-rule':q['drawing_lines'][0]['orientation']='vertical'
        if mode=='rule-above':q['drawing_lines'][0]['bbox'][1]=199.
        if mode=='rule-below':q['drawing_lines'][0]['bbox'][1]=220.
        if mode=='two-denominators':a=copy.deepcopy(d);a.update(text='γ另一原生文本变量',source_index=3);q['lines'].insert(2,a)
        if mode=='source-order':n['source_index']=10;d['source_index']=1;q['lines'].reverse()
        if mode=='inline-regions':n['inline_math_regions']=[[50.,200.,58.,208.]];d['inline_math_regions']=[[50.,210.,58.,220.]];d['native_math_words']=['β','α']
        if mode=='structural-membership':
            for l in [n,d]:l.update(paragraph_group=3,title_band_id=4,structural_title=True,explicit_section_title=True,title_suppressed=True)
        if mode=='char-before-space':d['chars'].insert(0,dict(char=' ',bbox=[0.,0.,1.,1.],char_idx=200))
        add(mode,q)
    previous=json.loads((ROOT/'build/migration/native-late-image-text.json').read_text(encoding='utf-8'))
    for c in previous['cases']:add('continuous/'+c['label'],dict(lines=c['expected']['remaining_lines'],drawing_lines=c['request']['source']['drawing_lines'],page_size=c['request']['source']['page_size']))
    write_json(target,dict(sources={'formulas.py':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/formulas.py').read_bytes())},cases=cases));print(len(cases),'original leading-fraction scenes;',sum(len(c['expected'])<len(c['request']['lines'])for c in cases),'positive')
if __name__=='__main__':main()
