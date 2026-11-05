"""冻结旧 sort_blocks_by_bbox→para_split 串联结果（含实际 CPU OCR 行）。"""
import copy,importlib,json
from common import ROOT,digest,write_json


def main():
    target=ROOT/'build/migration/legacy-sort-paragraphs.json'
    if target.exists():raise FileExistsError('Frozen baseline exists; review before generating another version')
    sort=importlib.import_module('rapid_doc.utils.block_sort').sort_blocks_by_bbox
    split=importlib.import_module('rapid_doc.backend.pipeline.para_split').para_split
    source=json.loads((ROOT/'build/migration/legacy-block-sort.json').read_text());cases=[]
    def add(name,request):
        r=copy.deepcopy(request)
        blocks=sort(r['blocks'],r['page_width'],r['page_height'],r['footnote_blocks'],None)
        pages=[dict(page_idx=0,page_size=[r['page_width'],r['page_height']],preproc_blocks=blocks)]
        split(pages)
        cases.append(dict(name=name,request=request,expected=pages))
    for c in source['cases']:
        if c['name'].startswith('random-'):add(c['name'],c['request'])
    boundaries=json.loads((ROOT/'build/migration/legacy-paragraph-boundaries.json').read_text())
    for c in boundaries['cases']:
        if not c['name'].startswith('cpu-ocr-'):continue
        p=c['pages'][0];blocks=copy.deepcopy(p['preproc_blocks'])
        for i,b in enumerate(blocks):b['original_order']=i
        add(c['name'],dict(blocks=blocks,page_width=p['page_size'][0],page_height=p['page_size'][1],footnote_blocks=[]))
    paths=['rapid_doc/utils/block_sort.py','rapid_doc/backend/pipeline/para_split.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in paths},cases=cases,
        profile='旧模型顺序排序与段落规则完整串联 JSON；300 组多块输入及 8 组实际 CPU OCR；不等于整文档解析'))
    print(f'Frozen {len(cases)} complete legacy sort→para_split outputs')


if __name__=='__main__':main()
