"""Freeze original Python Magika language labels and previously excluded outputs."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from rapid_doc.utils.guess_suffix_or_lang import guess_language_by_text,_get_magika
    from rapid_doc.model.magika.types import Seekable
    from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    examples=['','a','1234567','12345678',' '*5000,'\t\r\n','中文标题与示例代码',
        'def hello(name):\n    print("Hello", name)\n',
        'fn main() { println!("hello"); }',
        '#include <stdio.h>\nint main(void) { puts("hello"); }',
        'SELECT id, name FROM users WHERE active = 1;',
        '<html><body><h1>Hello</h1></body></html>',
        '{"name": "测试", "items": [1, 2, 3]}',
        'Algorithm 1: initialize x; repeat until convergence.']
    for name in ['demo/RapidDoc_end2end.py','rust/rapiddoc-inference/src/magika.rs',
                 'rust/rapiddoc-native/native/clipper6/clipper.cpp']:
        path=ROOT/name
        if path.exists():examples.append(path.read_text(encoding='utf8'))
    for n in [0,1,7,8,1023,1024,1025,4095,4096,4097]:
        for body in ['print("hello")\n','λ = "中文"\n','{}[]<>;']:
            examples.append(' '*n+body+'\n'*n)
    original=_get_magika()
    languages=[]
    for text in examples:
        cfg=original._model_config
        features=original._extract_features_from_seekable(Seekable(io.BytesIO(text.encode('utf8'))),
            cfg.beg_size,cfg.mid_size,cfg.end_size,cfg.padding_token,cfg.block_size,cfg.use_inputs_at_offsets)
        languages.append(dict(text=text,label=guess_language_by_text(text),features=features.beg+features.end))
    cases=[]
    historical=ROOT/'build/migration/legacy-page-chain.json'
    chain=json.loads(historical.read_text()) if historical.exists() else dict(cases=[])
    def algorithm(value):
        if isinstance(value,dict):return value.get('original_label')=='algorithm' or any(algorithm(v) for v in value.values())
        if isinstance(value,list):return any(algorithm(v) for v in value)
        return False
    def add(name,pages):
        for mode in ['mm_markdown','nlp_markdown','content_list']:
            cases.append(dict(name=name+'/'+mode,pages=pages,mode=mode,
                expected=union_make(copy.deepcopy(pages),mode,'images')))
    for case in chain['cases']:
        if algorithm(case['expected']):add(case['name'],case['expected'])
    for i,text in enumerate(examples):
        lines=[dict(bbox=[0,j*12,300,j*12+10],spans=[dict(type='text',content=line)])
               for j,line in enumerate(text.split('\n'))]
        for by_span in [False,True]:
            block=dict(type='text',bbox=[0,0,400,700],lines=copy.deepcopy(lines))
            if by_span:block['lines'][0]['spans'][0]['original_label']='algorithm'
            else:block['original_label']='algorithm'
            add(f'code/{i}/span={by_span}',[dict(page_idx=0,page_size=[600,800],para_blocks=[block],discarded_blocks=[])])
        plain=dict(type='text',bbox=[0,0,400,700],lines=copy.deepcopy(lines))
        add(f'ordinary-text/{i}',[dict(page_idx=0,page_size=[600,800],para_blocks=[plain],discarded_blocks=[])])
    model=ROOT/'rapid_doc/model/magika/models/standard_v3_3/model.onnx'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        languages=languages,cases=cases),ensure_ascii=False),encoding='utf8')
    print('Frozen',len(languages),'original Magika language/features cases and',len(cases),'algorithm output cases')


if __name__=='__main__':main()
