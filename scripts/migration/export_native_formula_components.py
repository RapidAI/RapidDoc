"""Execute the exact old formula-like-block prefix through lane spatial recovery."""
import ast,copy,json,random
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import formulas as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _AxisLine
    path=ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/formulas.py';tree=ast.parse(path.read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_build_formula_like_blocks');end=next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='recovered'for target in n.targets for t in ast.walk(target)))
    body=ast.Module(body=copy.deepcopy(fn.body[:end]),type_ignores=[]);ast.fix_missing_locations(body)
    ls=lines(q['lines']);scope=dict(vars(old),lines=ls,table_bboxes=[tuple(b)for b in q['table_bboxes']],page_size=tuple(q['page_size']),drawing_lines=[_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in q['drawing_lines']]);exec(compile(body,str(path),'exec'),scope)
    return clean(dict(formula_blocks=scope['blocks'],claimed_line_indices=sorted(scope['claimed_source_indices']),paragraph_lines=[project(l)for l in scope['paragraph_lines']],lines=[project(l)for l in ls]))
def main():
    target=ROOT/'build/migration/native-formula-components.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q.setdefault('page_size',[400.,600.]);q.setdefault('table_bboxes',[]);q.setdefault('drawing_lines',[]);q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-document-titles.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'expected'not in c:continue
        for i,p in enumerate(c['expected']['pages']):
            if not p['remaining_lines']:continue
            add('prepared/'+c['label']+'/'+str(i),dict(lines=p['remaining_lines'],page_size=p.get('page_size',p.get('source',{}).get('page_size')),table_bboxes=p.get('table_bboxes',[]),drawing_lines=p.get('drawing_lines',p.get('source',{}).get('drawing_lines',[]))))
    def base():return dict(lines=[line('Original complete body paragraph',[30.,y,370.,y+10.],i)for i,y in enumerate([100.,120.,140.,260.,280.,300.])])
    def tagged():return dict(line('x = y + z (1)',[120.,190.,340.,202.],20,h=10.,font='Math'),style_scale_repaired=True)
    for mode in ['single','single-not-repaired','single-tight','single-prefix-shrink','single-width','single-top','single-empty-prefix','single-greek','single-unicode-number','single-function','split','split-table','split-near-fragment','split-one-font','split-rotated','split-margin','spatial','spatial-scripts','spatial-repeat','spatial-prose','spatial-left-prose','spatial-table','spatial-fraction','spatial-title','isolated-compact','isolated-no-body','isolated-left','isolated-unnumbered','isolated-number','hanging','punctuated-anchor','two-anchors','same-height-anchors','equal-source-values','no-lines','metadata-word']:
        q=base();q['lines'].append(tagged())
        if mode=='single-not-repaired':q['lines'][-1]['style_scale_repaired']=False
        if mode=='single-tight':q['lines'][-1]['ink_bbox']=[122.,192.,337.,201.]
        if mode=='single-prefix-shrink':q['lines'][-1]['text']='\\[x = y\\] （Ａ１）'
        if mode=='single-width':q['lines'][-1]['bbox']=[30.,190.,375.,202.]
        if mode=='single-top':q['lines'][-1]['bbox']=[120.,3.,340.,15.]
        if mode=='single-empty-prefix':q['lines'][-1]['text']='(1)'
        if mode=='single-greek':q['lines'][-1]['text']='α = β × γ (2)'
        if mode=='single-unicode-number':q['lines'][-1]['text']='x = y (١.٢)'
        if mode=='single-function':q['lines'][-1]['text']='sin(x) = cos(y) (1)'
        if mode.startswith('split'):
            q['lines']=q['lines'][:-1]+[line('x =',[120.,190.,150.,200.],20,font='Math',visual_row_id=7,split_from_row=True),line('y + z',[160.,190.,200.,200.],21,font='Other',visual_row_id=7,split_from_row=True),line('(1)',[345.,190.,365.,200.],22,font='Body',visual_row_id=7,split_from_row=True)]
            if mode=='split-table':q['table_bboxes']=[[100.,180.,375.,210.]]
            if mode=='split-near-fragment':q['lines'].append(line('a',[90.,184.,115.,194.],23,font='Math'))
            if mode=='split-one-font':
                for l in q['lines'][-3:]:l['font_signature']=['Body',0]
            if mode=='split-margin':
                for l in q['lines'][-3:]:l['bbox'][1]-=180.;l['bbox'][3]-=180.
            if mode=='split-rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                q['page_size']=[600.,400.]
                for l in q['lines']:l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(q['page_size']),90))
        if mode.startswith('spatial')or mode in ['two-anchors','same-height-anchors','metadata-word']:
            q['lines']=q['lines'][:-1]+[line('x = y',[130.,190.,215.,200.],20,font='Math'),line('(1)',[350.,190.,370.,200.],21)]
            if mode=='spatial-scripts':q['lines']+=[line('a',[200.,179.,210.,187.],22,h=8.,font='Math'),line('b',[200.,202.,210.,210.],23,h=8.,font='Math')]
            if mode in ['spatial-repeat','two-anchors','same-height-anchors']:q['lines']+=[line('z = a',[130.,220.,215.,230.],22,font='Math'),line('(2)',[350.,190. if mode=='same-height-anchors'else 220.,370.,200. if mode=='same-height-anchors'else 230.],23)]
            if mode=='spatial-prose':q['lines'][-2]['text']='Original body words are ordinary prose = x'
            if mode=='spatial-left-prose':q['lines'][-2].update(text='Original body words = x',bbox=[40.,190.,180.,200.]);q['lines'].append(line('Original',[30.,190.,39.,200.],22,font='Math'))
            if mode=='spatial-table':q['table_bboxes']=[[220.,185.,240.,205.]]
            if mode=='spatial-fraction':q['lines']+=[line('a + b',[130.,180.,210.,188.],22,h=8.,font='Math'),line('c',[130.,205.,210.,213.],23,h=8.,font='Math')];q['drawing_lines']=[dict(bbox=[130.,200.,210.,200.],width=.5,orientation='horizontal')]
            if mode=='spatial-title':q['lines'].append(line('Original section title',[30.,205.,190.,217.],22,h=12.,font='Heading'))
            if mode=='metadata-word':q['lines'][-1]['text']='Review';q['lines'][-2]['text']='123'
        if mode.startswith('isolated'):
            q['lines'][-1]=line('x = y + z',[130.,190.,240.,205.],20,font='Math');q['lines'][-1]['compact_formula_cluster']=mode!='isolated-unnumbered';q['lines'][-1]['font_coverage']=.5
            if mode=='isolated-no-body':q['lines']=q['lines'][-1:]
            if mode=='isolated-left':q['lines'][-1]['bbox']=[45.,190.,155.,205.]
            if mode=='isolated-number':q['lines'].append(line('(1)',[350.,190.,370.,200.],21))
        if mode=='hanging':q['lines'][-1].update(bbox=[45.,245.,340.,255.],font_signature=['Body',0]);q['lines']+=[line('Original long previous reference tail',[45.,230.,370.,240.],21),line('Original following full width reference',[30.,260.,370.,270.],22)]
        if mode=='punctuated-anchor':q['lines'][-1].update(text='x = y',compact_formula_cluster=True);q['lines'].append(line(', (1)',[345.,190.,370.,200.],21))
        if mode=='equal-source-values':q['lines'].append(dict(tagged(),source_index=21))
        if mode=='no-lines':q['lines']=[]
        add(mode,q)
    rng=random.Random(801)
    for i in range(70):
        q=base();q['lines']+=[dict(tagged(),bbox=[rng.choice([40.,100.,130.]),190.,rng.choice([220.,300.,360.]),rng.choice([200.,210.,225.])],font_coverage=rng.choice([.5,.75,1.]),em_height=rng.choice([8.,10.,14.]))]
        add('mixed/'+str(i),q)
    positives=sum(bool(c.get('expected',{}).get('formula_blocks'))for c in cases);assert positives>=10
    write_json(target,dict(cases=cases));print(len(cases),'original formula spatial-prefix cases;',positives,'positive;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
