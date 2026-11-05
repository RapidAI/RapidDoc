"""Original continuous final-page prefix through _build_text_blocks, plus raw documents."""
import copy,json
from common import ROOT,write_json
from export_native_text_blocks import reference as blocks
def page(q,state):
    p=q['page'];q=copy.deepcopy(q);sz=p.get('page_size',p.get('source',{}).get('page_size'));out=copy.deepcopy(state);out['text_blocks']=blocks(dict(lines=out['semantic_lines']+out['remaining_lines'],table_bboxes=p.get('table_bboxes',[]),page_size=sz,drawing_lines=p.get('drawing_lines',p.get('source',{}).get('drawing_lines',[])),page_footnote_groups=out['page_footnote_groups'],page_index=q.get('page_index',0),visual_bboxes=[b['bbox']for b in p.get('fixed_blocks',[])if b.get('type')=='image']));return out
def main():
    target=ROOT/'build/migration/native-final-text-blocks.json';chain=ROOT/'build/migration/native-final-text-blocks-chain.json'
    if target.exists()or chain.exists():raise FileExistsError('immutable block chain references exist')
    from loguru import logger
    logger.remove();f=json.loads((ROOT/'build/migration/native-final-text-lines.json').read_text(encoding='utf-8'));g=json.loads((ROOT/'build/migration/native-final-text-lines-chain-canonical.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        if 'expected'in c:c['expected']=page(c['request'],c['expected'])
    for c in g['cases']:
        if 'expected'not in c:continue
        e=c['expected'];e['block_pages']=[page(dict(page=p,page_index=i),s)for i,(p,s)in enumerate(zip(e['pages'],e['text_pages']))];del e['text_pages']
    write_json(target,f);write_json(chain,g);print(len(f['cases']),'continuous text-block pages;',len(g['cases']),'raw-source document chains')
if __name__=='__main__':main()
