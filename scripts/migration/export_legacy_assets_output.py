"""冻结含真实 writer 路径的旧页面→编号→段落→输出参考，不覆盖旧文件。"""
import copy,importlib,json
from common import ROOT,digest,write_json


def main():
    target=ROOT/'build/migration/legacy-assets-output.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    assert (render.inline_left_delimiter,render.inline_right_delimiter,render.display_left_delimiter,render.display_right_delimiter)==('$','$','$$','$$')
    source=json.loads((ROOT/'build/migration/legacy-assets.json').read_text());cases=[]
    for c in source['cases']:
        page=copy.deepcopy(c['expected'])
        if page is None:
            r=c['request'];page=old.make_page_info_dict([],r['page_index'],int(r['page_width']),int(r['page_height']),[])
        pages=[page];old._optimize_formula_number_blocks(pages);old.para_split(pages)
        cases.append(dict(name=c['name'],pages=pages,outputs={mode:render.union_make(copy.deepcopy(pages),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}))
    paths=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in paths},profile='真实旧 writer 返回路径已写入的页面→公式编号→段落→默认 Markdown/content_list；不运行模型推理/内联资产/跨页表格合并'))
    print(f'Frozen {len(cases)} asset-page→paragraph chains and {len(cases)*3} old outputs')


if __name__=='__main__':main()
