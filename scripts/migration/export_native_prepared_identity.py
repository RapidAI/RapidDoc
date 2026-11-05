"""Whole remaining-page sequence across source aliases, resplits and frozen evidence."""
import copy,json
from common import ROOT,write_json
from export_native_geometry_pages import reference
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_table_page_policies import empty
def main():
    target=ROOT/'build/migration/native-prepared-identity.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['prepare_page']=True;c=dict(label=label,request=q)
        try:c['expected']=reference(q,post_text=True,full=True)
        except (ValueError,TypeError,AttributeError,IndexError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-text-role-aliases-canonical.json').read_text(encoding='utf-8'))
    for c in f['pages']:add(c['label'],c['request'])
    f=json.loads((ROOT/'build/migration/native-prepared-pages.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if c['label']in ['compact/wrapped-title','compact/footnote-superscript','compact/inline-math-word']:
            s=copy.deepcopy(c['request']['pages'][0]['source']);p=dict(source=s,remaining_lines=copy.deepcopy(s['lines']),table_bboxes=[],strong_graphic_bboxes=[],claimed_line_indices=[])
            add(c['label']+'/canonical',dict(prepared=p,plan=dict(document_style_anomaly=True),page_index=0,style_lines=[],link_lines=[]))
    write_json(target,dict(cases=cases));print(len(cases),'complete source-identity/canonical preparations;',sum('error'in c for c in cases),'old errors')
if __name__=='__main__':main()
