"""Freeze document preparation, sequential terminal grids, vector repeat and auxiliary passes."""
import copy,json
from common import ROOT,write_json
from export_native_document_pages import reference as prepared
from export_native_document_auxiliary import reference as auxiliary,page,body
from export_native_inline_clusters import clean
from export_native_late_image_text import project
from export_native_geometry import lines

def reference(q):
    from types import SimpleNamespace as NS
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.formulas import classify_repeated_vector_decorations
    out=prepared(q,post_text=True,footnotes=True,full=True,chain=True)
    ps=[NS(page_size=tuple(p['source']['page_size']),fixed_blocks=p['fixed_blocks'])for p in out['pages']]
    for p in ps:
        for b in p.fixed_blocks:
            if '_vector_shape'in b:b['_vector_shape']=tuple(tuple(x)for x in b['_vector_shape'])
    classify_repeated_vector_decorations(ps)
    result=auxiliary(dict(pages=out['pages']));out.update(result);return clean(out)

def main():
    target=ROOT/'build/migration/native-document-auxiliary-chain.json';proof=ROOT/'build/migration/native-document-auxiliary-proofs.json'
    if target.exists()or proof.exists():raise FileExistsError('immutable document auxiliary chain exists')
    from loguru import logger
    logger.remove();cases=[];proofs=[]
    def add(label,q):
        q=copy.deepcopy(q);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError)as e:c.update(error='invalid_input',old_exception=type(e).__name__,old_message=str(e))
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-prepared-pages.json').read_text(encoding='utf-8'))
    for i,c in enumerate(f['cases']):
        if i%29==0 or c['label'].startswith('live-')or c['label'].startswith('compact/'):add('prepared/'+c['label'],c['request'])
    from export_closed_rule_pipeline import base
    from export_native_table_page_policies import empty
    def wrap(s):
        s=dict(empty(),**s)
        for i,l in enumerate(s['lines']):
            x,y=l['bbox'][:2]
            if l['text']=='cell':l['text']=str(100+i)
            l['chars']=[dict(char=c,char_idx=i*100+j,bbox=[x+j*5.,y,x+(j+1)*5.,y+10.],font=dict(name='Body',size=10.,flags=0,weight=400.))for j,c in enumerate(l['text'])]
        s['chars']=[c for l in s['lines']for c in l['chars']];s['lines']=[project(l)for l in lines(s['lines'])]
        return dict(source=s,geometry=dict(chars=s['chars'],tight_bboxes=[],origins=[],loose_bboxes=[]))
    for mode in ['positive','empty-gap','above-body','wrong-columns','extra-table','stale-input']:
        first=base()['source'];first['lines']=first['lines'][:6]
        # Put a complete terminal table low on page 1; continuation starts page 2.
        for l in first['lines']:l['bbox'][1]+=300.;l['bbox'][3]+=300.
        for r in first['drawing_lines']:r['bbox'][1]+=300.;r['bbox'][3]+=300.
        second=base()['source'];second['lines']=[l for l in second['lines'][:6]if l['bbox'][0]<100.];second['drawing_lines']=[second['drawing_lines'][0],second['drawing_lines'][3],*second['drawing_lines'][4:]]
        if mode=='above-body':second['lines'].append(dict(text='Original prose above continuation',bbox=[30.,40.,300.,50.],source_index=30,angle=0,chars=[],effective_height=10.))
        if mode=='wrong-columns':
            for r in second['drawing_lines']:
                if r['orientation']=='vertical'and r['bbox'][0]==140.:r['bbox'][0]=r['bbox'][2]=120.
        if mode=='extra-table':first['lines'].append(dict(text='Body below terminal table',bbox=[30.,550.,300.,560.],source_index=30,angle=0,chars=[],effective_height=10.))
        if mode=='stale-input':first['previous_table_grid']=dict(page_index=-1,angle=0,positions=[.2,.3,.4],tolerance=.01);second['previous_table_grid']=dict(page_index=99,angle=0,positions=[.2,.3,.4],tolerance=.01)
        ps=[wrap(first),wrap(second)]
        if mode=='empty-gap':ps.insert(1,wrap(dict(empty(),lines=[])))
        add('grid/'+mode,dict(pages=ps))
    # Directed proofs for positive visual headers, compound split rows and title protection.
    from export_native_image_chain import line
    for mode in ['visual-positive','visual-only-two','visual-first-excluded','visual-gap','visual-size','split-header','split-footer','split-body-protection','compound-positive','compound-big','large-numbered']:
        ps=[page(body())]
        if mode.startswith('visual'):
            ps=[page(body())for _ in range(4)]
            for p in ps:p['fixed_blocks']=[dict(type='image',bbox=[20.,10.,60.,25.])]
            if mode=='visual-only-two':ps=ps[:3]
            if mode=='visual-first-excluded':ps[2]['fixed_blocks']=[]
            if mode=='visual-gap':ps.insert(2,page(body()));ps.insert(2,page(body()))
            if mode=='visual-size':ps[-1]['fixed_blocks'][0]['bbox'][2]=70.
        elif mode.startswith('split'):
            y=580. if mode=='split-footer'else 10.;target_type='footer'if mode=='split-footer'else'header'
            ps[0]['remaining_lines']+=[dict(line('Original marginal label',[30.,y,160.,y+8.],20),semantic_type=target_type,visual_row_id=9,split_from_row=True),dict(line('Original companion',[180.,y,300.,y+8.],21),visual_row_id=9,split_from_row=True,marginal_body_context=mode=='split-body-protection')]
        elif mode.startswith('compound'):
            h=12. if mode=='compound-big'else 8.
            ps[0]['remaining_lines']+=[dict(line('Original running title',[30.,10.,290.,10.+h],20),effective_height=h,visual_row_id=9,split_from_row=True),dict(line('Sidecar',[340.,10.,370.,10.+h],21),effective_height=h,visual_row_id=9,split_from_row=True)]
        else:
            ps[0]['remaining_lines'].append(dict(line('1. Original numbered heading',[30.,10.,300.,30.],20),effective_height=20.,em_height=20.));ps[0]['drawing_lines']=[dict(bbox=[20.,40.,380.,40.],orientation='horizontal',width=.5)]
        for p in ps:p['remaining_lines']=[project(l)for l in lines(p['remaining_lines'])]
        q=dict(pages=ps);proofs.append(dict(label=mode,request=q,expected=auxiliary(q)))
    positive=next(c for c in cases if c['label']=='grid/positive')
    if 'expected'in positive:
        ps=positive['expected']['pages'];assert ps[0]['next_table_grid']is not None
        assert ps[1]['source']['previous_table_grid']==ps[0]['next_table_grid']
        assert any(b['type']=='table'for b in ps[1]['fixed_blocks'])
    else:raise AssertionError(positive)
    v=next(c for c in proofs if c['label']=='visual-positive');assert [p['fixed_blocks'][0]['type']for p in v['expected']['pages']]==['image','header','header','header']
    v=next(c for c in proofs if c['label']=='compound-positive');assert all(l['semantic_type']=='header'for l in v['expected']['pages'][0]['remaining_lines'][-2:])
    write_json(target,dict(cases=cases));write_json(proof,dict(cases=proofs));print(len(cases),'full document chains;',len(proofs),'directed proofs;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
