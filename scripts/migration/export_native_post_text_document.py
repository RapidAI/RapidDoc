"""Freeze original full document-to-caption/title/prose classification sequences."""
import copy,json
from common import ROOT,write_json
from export_native_document_pages import reference
from export_native_text_roles import scenes
from export_native_table_page_policies import empty

def main():
    p=ROOT/'build/migration/native-post-text-document.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['include_post_text_roles']=True;c=dict(label=label,request=q)
        try:c['expected']=reference(q,post_text=True)
        except ValueError as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-document-pages-canonical.json').read_text(encoding='utf-8'))
    for c in f['cases']:add(c['label'],c['request'])
    for c in scenes():
        q=c['request'];s=dict(empty(),page_size=q['page_size'],lines=q['lines'],image_bboxes=q['visual_bboxes']);page=dict(source=s,geometry=dict(chars=[],tight_bboxes=[],origins=[],loose_bboxes=[]))
        pages=[page]
        if q['page_index']>0:pages.insert(0,dict(source=dict(empty(),page_size=q['page_size']),geometry=dict(chars=[],tight_bboxes=[],origins=[],loose_bboxes=[])))
        add('post/'+c['label'],dict(pages=pages))
    runtime=json.loads((ROOT/'build/migration/native-document-pages-runtime.json').read_text(encoding='utf-8'));add('live-marginal-pages',runtime['request'])
    write_json(p,dict(cases=cases));print(len(cases),'document-to-post-text sequences;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
