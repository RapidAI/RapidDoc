"""冻结实际旧公式tokenizer的额外token配置分支。"""
import copy,json
import numpy as np
from common import ROOT,write_json
def main():
    target=ROOT/'build/migration/formula-token-configs-v2.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.post_process import UniMERNetDecode
    metadata=json.loads((ROOT/'build/migration/formula-character.json').read_text(encoding='utf-8'));cases=[]
    default=metadata['tokenizer_config_file'];ordinary=copy.deepcopy(default)
    for token in ordinary['added_tokens_decoder'].values():token['special']=False
    added=copy.deepcopy(default)
    for i,text in enumerate(['formula-new-token','<formula-special>']):
        added['added_tokens_decoder'][str(50000+i)]=dict(content=text,lstrip=False,rstrip=False,normalized=False,single_word=False,special=bool(i))
    for config in [None,{},ordinary,added]:
        initial=copy.deepcopy(config) # 旧构造器会pop掉传入config的added_tokens_decoder。
        data=copy.deepcopy(metadata);data['tokenizer_config_file']=config
        d=UniMERNetDecode(data)
        rows=[[0,1,3,4,5,10,11,12,13,14,15,16,17,18,19],[0,50000,50001],[0,2,50000],[2],[999999]]
        expected=d.token2str([np.array(row,np.int64)for row in rows])
        cases.append(dict(config=initial,rows=rows,expected=expected))
    write_json(target,dict(cases=cases));print(f'Frozen {len(cases)} actual old added-token configurations')
if __name__=='__main__':main()
