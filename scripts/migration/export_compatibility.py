"""运行未修改的 Python 段落器/模型顺序分支，冻结实验策略审查依据。"""
import copy,json
from common import ROOT,write_json,digest
from rapid_doc.backend.pipeline.para_split import para_split
from rapid_doc.utils.block_sort import sort_blocks_by_xycut_plus


def legacy_blocks(page):
    output=[]
    for bi,b in enumerate(page['blocks']):
        kind='title'if b['region']['label']in ('doc_title','paragraph_title')else 'text'
        lines=[]
        for li,line in enumerate(b.get('lines',[])):
            points=line['points'];xs,ys=zip(*points);box=[min(xs),min(ys),max(xs),max(ys)]
            lines.append(dict(bbox=box,spans=[dict(type='text',content=line['text'],bbox=box,score=1.,audit_source=[bi,li])]))
        output.append(dict(type=kind,bbox=b['region']['coordinate'],lines=lines,original_order=bi))
    return output


def main():
    cases=[];skipped=[]
    for source in ('paragraphs','list-paragraphs','line-order'):
        data=json.loads((ROOT/f'build/migration/{source}.json').read_text())
        for c in data.get('page_cases',data.get('cases',[])):
            page=c['page'];name=source+'/'+c['name']
            if any(b['kind']!='text'or not b.get('lines')or b['region']['label']in ('algorithm','vertical_text')for b in page['blocks']):
                skipped.append(dict(name=name,reason='adapter_only_covers_nonempty_horizontal_text_and_titles'));continue
            pages=[dict(preproc_blocks=legacy_blocks(page),page_idx=0,page_size=[page['width'],page['height']])]
            para_split(pages)
            normalized=[]
            for b in pages[0]['para_blocks']:
                sources=[s['audit_source']for l in b['lines']for s in l['spans']]
                if sources:normalized.append(dict(kind=b['type'],sources=sources))
            options=dict(split_columns=c.get('split_columns',True))
            if source=='line-order':options['line_order']='geometry'
            cases.append(dict(name=name,page=page,options=options,legacy=normalized))
    blocks=[]
    for c in json.loads((ROOT/'build/migration/block-order.json').read_text())['cases']:
        if not c['page']['blocks']:continue
        inputs=legacy_blocks(c['page']);actual=sort_blocks_by_xycut_plus(copy.deepcopy(inputs),None)
        order=sorted(range(len(actual)),key=lambda i:actual[i]['index'])
        blocks.append(dict(name=c['name'],page=c['page'],legacy=order))
    sources=['rapid_doc/backend/pipeline/para_split.py','rapid_doc/utils/block_sort.py']
    write_json(ROOT/'build/migration/compatibility.json',dict(paragraphs=cases,block_order=blocks,skipped=skipped,
        sources={p:digest((ROOT/p).read_bytes())for p in sources},
        scope='适配后的几何/行来源合同与实际模型顺序优先分支；不是整文档质量评价。block order 使用输入数组顺序作为已知模型 original_order。'))
    print(f'Frozen old Python: {len(cases)} paragraph contracts, {len(blocks)} model-order priorities; {len(skipped)} explicitly outside adapter scope')


if __name__=='__main__':main()
