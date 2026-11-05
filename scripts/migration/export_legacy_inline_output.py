"""冻结真实内联图片 writer 页面到旧编号、段落和三种输出的串联参考。"""
import copy,importlib,json
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-inline-output.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    old=importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    render=importlib.import_module('rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent')
    assert (render.inline_left_delimiter,render.inline_right_delimiter,render.display_left_delimiter,render.display_right_delimiter)==('$','$','$$','$$')
    source=ROOT/'build/migration/legacy-inline-assets.json';cases=[]
    for c in json.loads(source.read_text())['pages']:
        page=copy.deepcopy(c['expected'])
        if page is None:
            r=c['request'];page=old.make_page_info_dict([],r['page_index'],int(r['page_width']),int(r['page_height']),[])
        pages=[page];old._optimize_formula_number_blocks(pages);old.para_split(pages)
        cases.append(dict(pages=pages,outputs={mode:render.union_make(copy.deepcopy(pages),mode,'images')for mode in ['mm_markdown','nlp_markdown','content_list']}))
    paths=['rapid_doc/backend/pipeline/model_json_to_middle_json.py','rapid_doc/backend/pipeline/para_split.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py']
    write_json(target,dict(cases=cases,source_sha256=digest(source.read_bytes()),sources={p:digest((ROOT/p).read_bytes())for p in paths}))
    print(f'Frozen {len(cases)} inline page chains and {len(cases)*3} old outputs')

if __name__=='__main__':main()
