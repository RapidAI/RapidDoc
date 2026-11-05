"""冻结实际旧公式post_process(LaTeX→ftfy)，并补充Unicode公式M模型CPU样本。"""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/formula-unicode.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.post_process import UniMERNetDecode
    old=UniMERNetDecode.__new__(UniMERNetDecode)
    source=ROOT/'build/migration/unicode-fix.json';f=json.loads(source.read_text(encoding='utf-8'))
    expected=[old.post_process(c['text'])for c in f['cases']]
    from rapid_doc.model.formula.rapid_formula_self import RapidFormula
    from rapid_doc.model.formula.rapid_formula_self.utils.typings import RapidFormulaInput
    engine=RapidFormula(RapidFormulaInput(model_dir_or_path=ROOT/'build/migration/models/pp_formulanet_plus_m.onnx',engine_cfg=dict(intra_op_num_threads=1,inter_op_num_threads=1,use_cuda=False,use_dml=False,use_cann=False)))
    folder=ROOT/'build/migration/formula-unicode';folder.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/simsun.ttc',32);images=[]
    for i,text in enumerate(['α + β = γ','Ａ＋Ｂ＝Ｃ','面积 S = a × b']):
        im=Image.new('RGB',(420,72),'white');ImageDraw.Draw(im).text((10,8),text,fill='black',font=font);p=folder/f'{i}.png';im.save(p);images.append(p)
    real=[]
    for indices,batch in [([0,1],2),([2],1)]:
        output=engine([np.asarray(Image.open(images[i]).convert('RGB'))for i in indices],batch_size=batch)
        values=[x.rec_formula for x in output]
        real.append(dict(sources=[str(images[i].relative_to(ROOT)).replace('\\','/')for i in indices],batch_size=batch,expected=values));print('Actual Unicode M CPU',values,flush=True)
    path='rapid_doc/model/formula/rapid_formula_self/model_handler/pp_formulanet_plus/post_process.py'
    write_json(target,dict(source_sha256=digest(source.read_bytes()),reference=dict(path=path,sha256=digest((ROOT/path).read_bytes())),expected=expected,real=real));print(f'Frozen {len(expected)} actual old post_process cases and {len(real)} real Unicode M CPU batches')
if __name__=='__main__':main()
