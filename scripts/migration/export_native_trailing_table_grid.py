"""Freeze old terminal-band continuation proof from real prepared block/line projections."""
import copy
from types import SimpleNamespace as NS
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_table_page_policies import source,empty
from export_native_bar_graphics import objects
from export_supplemental_rule_tables import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_rules import _trailing_table_continuation_grid
    s=source(q['source']);p=q['prepared'];result=_trailing_table_continuation_grid(s,NS(fixed_blocks=p['fixed_blocks'],remaining_lines=objects(p.get('remaining_lines',[]))))
    return clean(asdict(result))if result is not None else None
def base():
    s=empty();s['page_index']=2;s['lines']=[line('text',[50.,110.,80.,120.],0)];s['drawing_lines']=[dict(bbox=[40.,y,240.,y],width=0.,orientation='horizontal')for y in [100.,140.,180.,220.]]+[dict(bbox=[x,100.,x,220.],width=0.,orientation='vertical')for x in [40.,140.,240.]]
    return dict(source=s,prepared=dict(fixed_blocks=[dict(type='table',bbox=[40.,100.,240.,220.])],remaining_lines=[]))
def main():
    target=ROOT/'build/migration/native-trailing-table-grid.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('positive',base())
    for mode in ['no-page','no-table','rotated','merged-header','short-last-band','last-column-stops','last-column-gap','no-top','edge-mismatch','footer','body-after','remaining-after','remaining-footer','tied-tables','second-table','all-angles-em','no-lines','thin-em']:
        q=base();s=q['source'];p=q['prepared']
        if mode=='no-page':s['page_index']=None
        if mode=='no-table':p['fixed_blocks']=[]
        if mode=='rotated':p['fixed_blocks'][0]['angle']=90
        if mode=='merged-header':s['drawing_lines'][-2]['bbox'][1]=180.
        if mode=='short-last-band':s['drawing_lines'][2]['bbox']=[40.,217.,240.,217.]
        if mode=='last-column-stops':s['drawing_lines'][-2]['bbox'][3]=210.
        if mode=='last-column-gap':s['drawing_lines'][-2]['bbox'][3]=217.5
        if mode=='no-top':s['drawing_lines']=s['drawing_lines'][1:]
        if mode=='edge-mismatch':p['fixed_blocks'][0]['bbox'][2]=250.
        if mode in ['footer','body-after']:p['fixed_blocks'].append(dict(type='footer'if mode=='footer'else'text',bbox=[0.,230.,100.,240.]))
        if mode in ['remaining-after','remaining-footer']:p['remaining_lines']=[dict(line('Below',[0.,230.,100.,240.],2),semantic_type='footer'if mode=='remaining-footer'else None)]
        if mode=='tied-tables':p['fixed_blocks'].append(dict(type='table',bbox=[40.,100.,250.,220.]))
        if mode=='second-table':p['fixed_blocks'].append(dict(type='table',bbox=[40.,100.,250.,230.]))
        if mode=='all-angles-em':s['lines'].append(dict(line('Vertical',[0.,0.,10.,80.],3),angle=90,effective_height=80.))
        if mode=='no-lines':s['lines']=[]
        if mode=='thin-em':s['lines'][0]['effective_height']=0.01
        add(mode,q)
    for value in [215.99999,216.,216.00001,217.49999,217.5,217.50001,220.]:
        q=base();q['source']['drawing_lines'][-2]['bbox'][3]=value;add('bottom-'+str(value),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/'+p for p in ['table_rules.py','geometry.py','models.py']]
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},scope='Prepared fixed blocks and remaining lines; derive last physical band tracks, never inject tracks.',cases=cases));print('Frozen trailing grids',len(cases))
if __name__=='__main__':main()
