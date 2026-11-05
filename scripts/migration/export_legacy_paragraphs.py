"""冻结未修改的 para_split 完整 JSON 输出和执行覆盖；不导入 Rust，不覆盖旧基线。"""
import argparse,copy,importlib,inspect,json,random,sys
from common import ROOT,digest,write_json
from export_compatibility import legacy_blocks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='build/migration/legacy-paragraphs.json');args=parser.parse_args()
    target=ROOT/args.output
    if target.exists():parser.error('Baseline already exists; use a new --output and review changes')
    module=importlib.import_module('rapid_doc.backend.pipeline.para_split');cases=[];coverage={};filename=module.para_split.__code__.co_filename
    def trace(frame,event,arg):
        if frame.f_code.co_filename!=filename:return None
        if event=='line':coverage.setdefault(frame.f_code.co_name,set()).add(frame.f_lineno)
        return trace
    def span(text,kind='text'):return dict(type=kind,content=text,custom=dict(preserved=[1,False,None]))
    def line(x,y,r,text,extra=False):
        spans=[span(text)]if not extra else [span(' '+text+' '),span('x²','inline_equation'),span(' tail')]
        return dict(bbox=[x,y,r,y+20],spans=spans,is_list_start_line=False,custom='line')
    def block(lines,kind='text',width=200):return dict(type=kind,bbox=[0,0,width,500],lines=lines,custom=['block',1],page_num=-1,page_size='overwritten')
    def page(blocks,index=0,width=600):return dict(page_idx=index,page_size=[width,800],preproc_blocks=blocks,para_blocks=['overwritten'],custom=dict(preserved=True))
    def add(name,pages):
        before=copy.deepcopy(pages);expected=copy.deepcopy(pages)
        sys.settrace(trace)
        try:module.para_split(expected)
        finally:sys.settrace(None)
        assert pages==before
        cases.append(dict(name=name,pages=pages,expected=expected))
    add('empty',[]);add('empty-page',[page([])]);add('empty-text',[page([block([])])])
    add('opaque-barriers',[page([dict(type='image',custom='opaque'),block([line(0,0,200,'body')]),dict(type='title',custom='opaque title')])])
    add('centered-list',[page([block([line(0,0,200,'wide'),line(50,30,150,'center'),line(50,60,150,'center'),line(50,90,150,'center')])])])
    texts=['item','1. item','item ²','中文段落','日本語','한국어','中文English','中文ABC','\x1c a \x1c','', 'a.', 'a；','1. a.', 'text tail']
    rng=random.Random(2846)
    for i in range(350):
        rows=[]
        for j in range(rng.randint(1,12)):
            x=rng.choice([0,0,0,10,14,20,40,50]);r=rng.choice([80,100,130,160,180,200])
            rows.append(line(x,j*30,r,rng.choice(texts),extra=rng.random()<.12))
        add(f'classifier-{i}',[page([block(rows)],width=rng.choice([0,300,400,600,1000]))])
    # 所有旧结束标识、首字符及严格边界；上下页几何保留旧规则实际比较方式。
    for char in ['a','A','é','É','Ⅻ','ǅ','Ǆ','²','９','\x1c','']:
        for ending in ['', '.', '!', '?','。','！','？',')','）','"','”',':','：',';','；','word. ']:
            left=block([line(0,j*30,200,'body')for j in range(4)]);left['lines'][-1]['spans'][0]['content']='tail'+ending
            right=block([line(0,0,200,char+'continued'),line(0,30,200,'more')])
            add(f'merge-{ord(char)if char else 0}-{ending}',[page([left]),page([right],index=7)])
    for i in range(70):
        pages=[]
        for pi in range(2):
            blocks=[]
            for bi in range(rng.randint(1,4)):
                if rng.random()<.15:blocks.append(dict(type=rng.choice(['title','table','image','interline_equation']),custom='barrier'));continue
                rows=[line(rng.choice([0,0,40]),j*30,rng.choice([100,160,200]),rng.choice(texts))for j in range(rng.randint(1,7))]
                blocks.append(block(rows))
            pages.append(page(blocks,index=pi*3))
        add(f'groups-{i}',pages)
    old=json.loads((ROOT/'build/migration/compatibility.json').read_text())
    for c in old['paragraphs']:
        p=c['page'];add('audit/'+c['name'],[dict(page_idx=0,page_size=[p['width'],p['height']],preproc_blocks=legacy_blocks(p))])
    covered={name:sorted(lines)for name,lines in coverage.items()}
    source='rapid_doc/backend/pipeline/para_split.py'
    executable={name:len(inspect.getsourcelines(getattr(module,name))[0])for name in ('__is_list_or_index_block','__merge_2_text_blocks','__merge_2_list_blocks','__para_merge_page','para_split')}
    write_json(target,dict(source=source,source_sha256=digest((ROOT/source).read_bytes()),cases=cases,executed_lines=covered,function_source_lines=executable,
        profile='完整 para_split JSON 合同；有效 Unicode、正文字几何、唯一整数页号；非文字块未知字段原样保留。'))
    print(f'Frozen {len(cases)} complete legacy para_split outputs; executed lines per function: '+str({k:len(v)for k,v in covered.items()}))


if __name__=='__main__':main()
