"""仅为迁移对照获取旧配置固定SHA的M模型；运行库不会自动下载。"""
import hashlib,time
from pathlib import Path
import requests,yaml
from common import ROOT

def main():
    cfg=yaml.safe_load((ROOT/'rapid_doc/model/formula/rapid_formula_self/configs/default_models.yaml').read_text())['onnxruntime']['pp_formulanet_plus_m']
    target=ROOT/'build/migration/models/pp_formulanet_plus_m.onnx'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert hashlib.file_digest(target.open('rb'),'sha256').hexdigest()==cfg['SHA256'];print(target);return
    temporary=target.with_suffix('.onnx.part');sha=hashlib.sha256();count=0;last=time.monotonic()
    with requests.get(cfg['model_dir_or_path'],stream=True,timeout=(20,60))as response:
        response.raise_for_status()
        with temporary.open('wb')as output:
            for chunk in response.iter_content(1024*1024):
                count+=len(chunk)
                if count>1_000_000_000:raise ValueError('formula model exceeds download budget')
                output.write(chunk);sha.update(chunk)
                if time.monotonic()-last>20:print(f'Downloaded {count//1024//1024} MiB',flush=True);last=time.monotonic()
    assert sha.hexdigest()==cfg['SHA256'],(count,sha.hexdigest())
    temporary.replace(target);print(f'Verified {count} bytes: {target}',flush=True)

if __name__=='__main__':main()
