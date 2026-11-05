"""Freeze the complete original formula recognizer and its component checkpoints."""
import copy,json,random
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import formulas as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    ls=lines(q['lines']);tables=[tuple(b)for b in q['table_bboxes']];size=tuple(q['page_size']);rules=[_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in q['drawing_lines']]
    kind=q.get('kind','full')
    if kind=='full':
        blocks,remaining=old._build_formula_like_blocks(ls,tables,size,drawing_lines=rules)
        return clean(dict(formula_blocks=blocks,remaining_lines=[project(l)for l in remaining],lines=[project(l)for l in ls]))
    if kind=='detached':blocks,claimed=old._recover_detached_display_components(ls,tables,size,drawing_lines=rules)
    elif kind=='mixed':blocks,claimed=old._build_mixed_body_display_formulas(ls,tables,size)
    elif kind=='bands':blocks,claimed=old._build_spatial_numbered_bands(ls,tables,size,rules)
    else:raise ValueError(kind)
    return clean(dict(formula_blocks=blocks,lines=[project(l)for l in ls]))

def main():
    target=ROOT/'build/migration/native-formula-full.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('table_bboxes',[]);q.setdefault('drawing_lines',[]);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    prefix=json.loads((ROOT/'build/migration/native-formula-components.json').read_text(encoding='utf-8'))
    for c in prefix['cases']:add(c['label'],c['request'])
    def base():return dict(lines=[line('Original complete body paragraph',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])])
    def mathline(t,b,i,**kw):return dict(line(t,b,i,font='Math',**kw),font_coverage=.5)
    seeds=[]
    for mode in ['detached','ink','calculus','native-word','native-word-barrier','mixed','fraction','stacked','band','band-fallback-marker','band-Unicode','band-ambiguity','band-body','band-style','band-nonword','band-three','band-compact','band-table','inline-host','anchored','extra-ink','adjacent-equations','whole-overlap','margin','paragraph','candidate','rotated','duplicate-values']:
        q=base();q['lines'] += [mathline('x = y + z',[130.,190.,220.,200.],20),mathline('a',[155.,177.,180.,185.],21,h=8.),mathline('b',[155.,204.,180.,212.],22,h=8.)]
        if mode in ['ink','calculus','native-word','native-word-barrier']:
            for l in q['lines'][6:]:l['ink_bbox']=l['bbox'][:]
        if mode=='calculus':q['lines'][6]['text']='∫ x dx'
        if mode.startswith('native-word'):q['lines'][6].update(text='abc = x + y',native_math_words=['abc']if mode=='native-word'else[])
        if mode=='mixed':
            for l in q['lines'][:6]:l['font_coverage']=.5
            q['lines'] += [dict(line('Original complete body paragraph',[30.,y,370.,y+10.],i),font_coverage=.5)for i,y in [(30,170.),(31,230.)]]
        if mode=='fraction':q['drawing_lines']=[dict(bbox=[152.,201.,184.,201.],orientation='horizontal',width=.5)]
        if mode.startswith('band'):
            q['lines'] += [line('(1)',[350.,209.,370.,219.],23)]
            if mode=='band-fallback-marker':q['lines'][-1]['text']='[1]'
            if mode=='band-Unicode':q['lines'][-1]['text']='（١）'
            if mode=='band-ambiguity':q['lines'] += [mathline('z = x',[240.,190.,290.,200.],24)]
            if mode=='band-body':q['lines'].append(line('where ordinary',[30.,190.,120.,200.],24))
            if mode in ['band-style','band-three','band-compact','band-nonword']:
                q['lines'][6]['text']='a b c'if mode=='band-style'else 'abc'
                if mode=='band-style':q['lines'][6]['font_signature']=['Math',64]
                if mode=='band-compact':q['lines'][6]['compact_formula_cluster']=True
                if mode=='band-nonword':q['lines'][6]['text']='abc_1 a b c'
            if mode=='band-table':q['table_bboxes']=[[125.,170.,230.,220.]]
        if mode=='inline-host':q['lines'].append(dict(line('Original ordinary words',[225.,190.,355.,200.],24),ink_bbox=[225.,190.,355.,200.]))
        if mode=='anchored':q['lines']=q['lines'][:6]+[mathline('x = y + z',[130.,190.,220.,200.],20),line('… (1)',[280.,190.,310.,200.],21)]
        if mode=='extra-ink':q['lines'] += [dict(line('a',[180.,214.,185.,220.],25,h=6.),ink_bbox=[180.,213.,185.,220.])]
        if mode=='adjacent-equations':q['lines'] += [mathline('z = a + b',[130.,217.,220.,227.],25)]
        if mode=='whole-overlap':q['lines'] += [mathline('w = x + y',[210.,204.,300.,214.],25),line('(1)',[350.,190.,370.,200.],26)]
        if mode=='margin':
            for l in q['lines'][6:]:l['bbox'][1]-=170.;l['bbox'][3]-=170.
        if mode=='paragraph':q['lines'][6]['paragraph_group']=7
        if mode=='candidate':q['lines'][6]['formula_candidate_only']=True
        if mode=='duplicate-values':q['lines'].append(copy.deepcopy(q['lines'][-1]))
        if mode=='rotated':
            from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
            q['page_size']=[600.,400.]
            for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        seeds.append((mode,q));add('continuation/'+mode,q)
        for kind in ['mixed','detached','bands']:add(kind+'/'+mode,dict(q,kind=kind))
    rng=random.Random(1028)
    for i in range(160):
        mode,q=copy.deepcopy(rng.choice(seeds));q['lines']=q['lines'][:6]+[dict(l,bbox=[l['bbox'][0]+rng.choice([-10.,0.,10.]),l['bbox'][1]+rng.choice([-4.,0.,4.]),l['bbox'][2]+rng.choice([-5.,0.,5.]),l['bbox'][3]+rng.choice([-2.,0.,2.])])for l in q['lines'][6:]]
        if i%5==0:q['lines'][6]['text']=rng.choice(['sample-١','abc = x','lim(x) = y','α ± β','x = y (2)'])
        add('variation/'+str(i),q)
    positive={k:sum(bool(c.get('expected',{}).get('formula_blocks'))for c in cases if c['request'].get('kind','full')==k)for k in ['full','mixed','detached','bands']}
    assert all(positive.values()),positive
    write_json(target,dict(cases=cases));print(len(cases),'original complete formula cases;',positive,'positive;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
