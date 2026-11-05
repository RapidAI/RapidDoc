"""The evaluation runner preserves successful predictions and records failures."""
from pathlib import Path
import subprocess
import sys


def test_file_logs_failures_and_resume_without_inference(tmp_path):
    script=Path(__file__).resolve().parents[2]/'demo/RapidDoc_rust_end2end.py'
    code='''import pathlib,runpy,sys,json
from types import SimpleNamespace
from rapid_doc_rust._native import RapidDocError
scope=runpy.run_path(sys.argv[1]);main=scope['main'];state=main.__globals__
import os
assert os.environ['USE_DOC_ORIENTATION_CLASSIFY']=='true'
assert state['OUTPUT_DIR'].name.endswith('_orientation_on_run2')
root=pathlib.Path(sys.argv[2]);inputs=root/'inputs';inputs.mkdir();out=root/'predictions';out.mkdir()
for name in ['cached','good','bad']:(inputs/(name+'.pdf')).write_bytes(b'%PDF-stub')
(out/'cached.md').write_text('keep',encoding='utf8')
class Engine:
 def __init__(self,**kw):pass
 def __call__(self,source,**kw):
  assert source.stem!='cached'
  if source.stem=='bad':raise RapidDocError('controlled test failure')
  return SimpleNamespace(markdown='ok',images={'asset.png':b'asset'})
state.update(FILES_DIR=inputs,OUTPUT_DIR=out,RapidDoc=Engine)
try:main()
except SystemExit as error:assert error.code==1
else:raise AssertionError('Failure was silently marked successful')
assert (out/'cached.md').read_text()=='keep'
assert (out/'good.md').read_text()=='ok'
assert not (out/'bad.md').exists()
assert not list(out.glob('*.part'))
logs=root/'predictions_logs'
failures=list(logs.glob('*.failures.jsonl'));assert len(failures)==1
record=json.loads(failures[0].read_text(encoding='utf8'));assert record['file'].endswith('bad.pdf')
assert record['error_type']=='RapidDocError'
assert 'controlled test failure' in next(logs.glob('*.errors.log')).read_text(encoding='utf8')
assert any('Done: completed=1, skipped=1, failed=1' in p.read_text(encoding='utf8') for p in logs.glob('*.log'))
'''
    subprocess.run([sys.executable,'-c',code,str(script),str(tmp_path)],check=True,capture_output=True,text=True)
