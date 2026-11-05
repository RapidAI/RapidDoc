"""冻结真实旧公式修复/tokenizer及M模型CPU输出；ftfy未迁移字符明确列为unsupported。"""
import copy,json,random
from pathlib import Path
from html.entities import html5
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,digest,write_json

def safe(s):
    if not all(' '<=c<='~' or c in '\n\t' or '\u4e00'<=c<='\u9fff'for c in s):return False
    for i,c in enumerate(s):
        if c!='&':continue
        tail=s[i+1:]
        if tail.startswith('#'):return False
        if any(tail.startswith(key)for key in html5):return False
    return True

def main():
    target=ROOT/'build/migration/formula-decode.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.post_process import UniMERNetDecode
    import onnx,gc
    proto=onnx.load(ROOT/'build/migration/models/pp_formulanet_plus_m.onnx',load_external_data=False)
    metadata=json.loads(next(p.value for p in proto.metadata_props if p.key=='character'))
    del proto;gc.collect()
    character=ROOT/'build/migration/formula-character.json'
    if character.exists():assert json.loads(character.read_text(encoding='utf-8'))==metadata
    else:character.write_text(json.dumps(metadata,ensure_ascii=False),encoding='utf-8')
    decoder=UniMERNetDecode(copy.deepcopy(metadata));rng=random.Random(331)
    texts=['','x+y','中文公式',r'\left( x \right)',r'\left( x',r'\lefteqn x \right)',r'{\left(x} \right)',r'\text{ 中文 }',r'\text{a中文b }','"x"',r'\uparrow \upalpha \uplus \upsilon \updownarrow',r'\ensuremath{x} \textsubscript{y}',r'\begin{align*}x',r'\end{align*}',r'\begin{alignnn}x',r'\end{alig}',r'a&b\\c&d','a&amp;b','<x &amp; y>','é','αβ','Ｘ＋Ｙ','e\u0301','\r\n','\x1b[31mx','A\x1c中文','\text\x1c{中文}',r'\text{a{中文}}',r'\left\{ x \right\}']
    envs=['array','matrix','pmatrix','bmatrix','vmatrix','Bmatrix','Vmatrix','cases','aligned','gathered','align','align*']
    for env in envs:
        for n in range(1,5):
            texts.extend([rf'\begin{{{env}}}'*n+'x',rf'\end{{{env}}}'*n,rf'\begin{{{env}}}{{rr}}x'+rf'\end{{{env}}}'*(n+1)])
    atoms=['{','}','\\','\\\\',r'\left(',r'\right)',r'\left.',r'\right.',r'\lefteqn',r'\rightarrow',r'\text{中文}',r'\upalpha',r'\null','x','中文',' ',r'\begin{array}',r'\end{matrix}']
    for _ in range(500):texts.append(''.join(rng.choices(atoms,k=rng.randrange(1,25))))
    repairs=[];cases=[]
    for text in texts:
        repaired=decoder.fix_latex(decoder.remove_chinese_text_wrapping(text))
        final=decoder.post_process(text)
        if safe(repaired):assert final==repaired,(text,repaired,final)
        repairs.append(dict(text=text,expected=repaired))
        ids=decoder.tokenizer.encode(text,add_special_tokens=False).ids
        for row in [ids, [0]+ids+[2]+[15,16,17]]:
            decoded=decoder.tokenizer.decode(row[:row.index(2)+1]if 2 in row else row,skip_special_tokens=True)
            repaired=decoder.fix_latex(decoder.remove_chinese_text_wrapping(decoded))
            cases.append(dict(tokens=row,expected=decoder.token2str([np.array(row,np.int64)])[0],supported=safe(repaired)))
    for row in [[],[0,1,2,3],[999999],[2,2,2],[3],[0,1]]:
        cases.append(dict(tokens=row,expected=decoder.token2str([np.array(row,np.int64)])[0],supported=True))
    # 实际旧RapidFormula全链，CPU使用原图，默认优化在非CUDA路径不改写图。
    from rapid_doc.model.formula.rapid_formula_self import RapidFormula
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import RapidFormulaInput
    engine=RapidFormula(RapidFormulaInput(model_dir_or_path=ROOT/'build/migration/models/pp_formulanet_plus_m.onnx',engine_cfg=dict(intra_op_num_threads=1,inter_op_num_threads=1,use_cuda=False,use_dml=False,use_cann=False)))
    folder=ROOT/'build/migration/formula-real';folder.mkdir(parents=True,exist_ok=True)
    images=[]
    font=ImageFont.truetype('C:/Windows/Fonts/times.ttf',36)
    for i,text in enumerate(['x + y = z','a = b + c','']):
        im=Image.new('RGB',(280,72),'white');ImageDraw.Draw(im).text((10,8),text,font=font,fill='black');path=folder/f'{i}.png';im.save(path);images.append(path)
    real=[]
    for indices,batch in [([0],1),([1],1),([2],1),([0,1],2)]:
        inputs=[np.asarray(Image.open(images[i]).convert('RGB'))for i in indices]
        output=engine(inputs,batch_size=batch)
        real.append(dict(sources=[str(images[i].relative_to(ROOT)).replace('\\','/')for i in indices],batch_size=batch,expected=[o.rec_formula for o in output]))
        print('Actual CPU formula',indices,[o.rec_formula for o in output],flush=True)
    refs=['rapid_doc/model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/post_process.py','rapid_doc/model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/utils.py','rapid_doc/model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/main.py','rapid_doc/backend/onnx_ops/formula.py']
    import tokenizers,ftfy
    write_json(target,dict(schema=1,metadata_sha256=digest((ROOT/'build/migration/formula-character.json').read_bytes()),versions=dict(tokenizers=tokenizers.__version__,ftfy=ftfy.__version__),references={p:digest((ROOT/p).read_bytes())for p in refs},repairs=repairs,cases=cases,real=real,model=dict(path='build/migration/models/pp_formulanet_plus_m.onnx',sha256='52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d')))
    print(f'Frozen {len(repairs)} repairs, {len(cases)} tokenizer cases, {len(real)} actual CPU formula batches')

if __name__=='__main__':main()
