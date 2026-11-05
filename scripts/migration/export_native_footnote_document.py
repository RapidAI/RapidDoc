"""Original raw-document pipeline through early image and page footnotes."""
import json,copy
from common import ROOT,write_json
from export_native_document_pages import reference
from export_native_table_page_policies import empty

def main():
    p=ROOT/'build/migration/native-footnote-document.json'
    if p.exists():raise FileExistsError(p)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q);q['include_post_footnotes']=True;c=dict(label=label,request=q)
        try:c['expected']=reference(q,post_text=True,footnotes=True)
        except ValueError as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    f=json.loads((ROOT/'build/migration/native-post-text-document.json').read_text(encoding='utf-8'))
    for c in f['cases']:add(c['label'],c['request'])
    for name in ['native-early-footnotes-canonical','native-early-footnote-edges']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            r=copy.deepcopy(c['request']);s=dict(empty(),lines=r['lines'],page_size=r['page_size'],image_bboxes=r['visual_bboxes'],drawing_lines=r['drawing_lines'])
            if r['reference_lines']and s['lines']:s['lines'][0]['chars']=r['reference_lines'][0]['chars']
            add(name+'/'+c['label'],dict(pages=[dict(source=s,geometry=dict(chars=[],tight_bboxes=[],origins=[],loose_bboxes=[]))]))
    write_json(p,dict(cases=cases));print(len(cases),'original document-to-footnote sequences;',sum('error'in c for c in cases),'old exceptions')
if __name__=='__main__':main()
