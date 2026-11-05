"""补充旧段落浮点/Unicode/真实 OCR 合同；独立生成旧输出，不覆盖基线。"""
import copy,importlib,json,re
from common import ROOT,write_json,digest


def main():
    target=ROOT/'build/migration/legacy-paragraph-boundaries.json'
    if target.exists():raise FileExistsError('Frozen baseline exists; review before generating another version')
    legacy=importlib.import_module('rapid_doc.backend.pipeline.para_split').para_split
    initial=json.loads((ROOT/'build/migration/legacy-paragraphs.json').read_text());cases=[]
    def add(name,pages):
        expected=copy.deepcopy(pages);legacy(expected);cases.append(dict(name=name,pages=pages,expected=expected))
    samples=initial['cases'][5:35]+initial['cases'][355:375]
    for index,c in enumerate(samples):
        for scale in (.5,1.25,1.333333333,0.0001):
            pages=copy.deepcopy(c['pages'])
            for page in pages:
                page['page_size']=[v*scale for v in page['page_size']]
                for b in page['preproc_blocks']:
                    if 'bbox'in b:b['bbox']=[v*scale+(.25 if i%2==0 else -.75)for i,v in enumerate(b['bbox'])]
                    for line in b.get('lines',[]):line['bbox']=[v*scale+(.25 if i%2==0 else -.75)for i,v in enumerate(line['bbox'])]
            add(f'float-{index}-{scale}',pages)
    # 固定属性表只选择测试字符；结果由未修改的旧 Python 整链计算。
    chars=set()
    source=(ROOT/'rust/rapiddoc-native/src/text_unicode_data.rs').read_text()
    for a,b,flags in re.findall(r'\(0x([0-9A-F]+),\s*0x([0-9A-F]+),\s*(\d+)\)',source):
        if int(flags)&256:chars.update((int(a,16),int(b,16)))
    template=next(c['pages']for c in initial['cases']if c['name']=='merge-97-')
    values=sorted(chars)
    for start in range(0,len(values),150):
        blocks=[]
        for code in values[start:start+150]:
            left,right=copy.deepcopy(template[0]['preproc_blocks'][0]),copy.deepcopy(template[1]['preproc_blocks'][0])
            right['lines'][0]['spans'][0]['content']=chr(code)+'continued';right['unicode_codepoint']=code
            blocks.extend([left,right,dict(type='title',unicode_separator=code)])
        add(f'unicode-upper-{start}',[dict(page_idx=0,page_size=[600,800],preproc_blocks=blocks)])
    for name in ('native-fusion','list-paragraphs'):
        data=json.loads((ROOT/f'build/migration/{name}.json').read_text())
        examples=data['cases']if name=='native-fusion'else [data['runtime']]
        for i,c in enumerate(examples):
            lines=[]
            for j,l in enumerate(c['ocr']):
                x,y=zip(*l['points']);bbox=[min(x),min(y),max(x),max(y)]
                lines.append(dict(bbox=bbox,spans=[dict(type='text',content=l['text'],score=l['confidence'],source_line=j)]))
            add(f'cpu-ocr-{name}-{i}',[dict(page_idx=c.get('page_index',0),page_size=[c['width'],c['height']],preproc_blocks=[dict(type='text',bbox=[0,0,c['width'],c['height']],lines=lines)])])
    write_json(target,dict(cases=cases,unicode_upper_characters=len(values),source=initial['source'],source_sha256=initial['source_sha256'],
        reference='完整旧 para_split，包含 FP64 bbox/元数据与独立 CPU OCR 行；不是完整文档解析器对照'))
    print(f'Frozen {len(cases)} float/Unicode/OCR contracts ({len(values)} uppercase boundary scalars)')


if __name__=='__main__':main()
