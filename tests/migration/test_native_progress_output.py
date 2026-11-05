"""Native stage bars work with IDE pipes and stay readable in redirected logs."""
import os
import subprocess
import sys


def capture(mode=None, ide=False):
    env=dict(os.environ)
    env.pop('RAPIDDOC_PROGRESS',None)
    env.pop('PYCHARM_HOSTED',None)
    if mode is not None:env['RAPIDDOC_PROGRESS']=mode
    if ide:env['PYCHARM_HOSTED']='1'
    code='''from rapid_doc_rust._native import ParseProgress
p=ParseProgress('document',1000)
p.update('layout',999)
p.advance('layout',0,5)
p.advance('layout',3,5)
p.clear()
p.advance('layout',5,5)
p.advance('layout',5,5)
p.reset()
p.advance('layout',0,2)
p.advance('layout',2,2)
p.advance('formula',0,10)
p.advance('formula',3,10)
p.clear()
'''
    return subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,check=True).stderr.decode('utf8')


def test_redirected_progress_uses_real_stage_counts_and_keeps_completions():
    text=capture()
    assert '\r'not in text
    assert text.count('5/5')==1
    assert text.count('2/2')==1
    assert '1000/1000'not in text
    assert '10/10'not in text  # An interrupted phase must not be marked complete.
    assert 'Layout Predict: 100%'in text
    assert '<00:00,'in text


def test_pycharm_and_forced_dynamic_progress_use_carriage_returns():
    for text in (capture(ide=True),capture('always')):
        assert '\r'in text
        assert 'Layout Predict: 100%'in text
        assert '5/5'in text and '2/2'in text


def test_progress_can_be_disabled():
    assert capture('off')==''


def test_measured_time_excludes_other_stages_and_initialization():
    env=dict(os.environ,RAPIDDOC_PROGRESS='plain')
    code='''from rapid_doc_rust._native import ParseProgress
p=ParseProgress('document',14)
p.advance('orientation',0,14,0.0)
p.advance('layout',1,1,100.0)
p.advance('orientation',14,14,0.35)
for invalid in [-1.,float('nan'),float('inf')]:
 try:p.advance('orientation',0,14,invalid)
 except ValueError:pass
 else:raise AssertionError('Invalid duration accepted')
'''
    text=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,check=True).stderr.decode('utf8')
    line=next(line for line in text.splitlines()if'Orientation Predict: 100%'in line)
    assert '14/14 [00:00<00:00, 40.00it/s]'in line
