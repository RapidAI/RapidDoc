"""The public API uses wall time for bars and sums measured work separately."""
import subprocess
import sys


def test_public_progress_uses_wall_clock_and_aggregates_windows():
    code='''import json
from types import SimpleNamespace
import rapid_doc_rust as package
import rapid_doc_rust._native as native
import rapid_doc_rust._runtime as runtime
import rapid_doc_rust._compat as compat
from loguru import logger
package.parse_pdf_text=lambda *a,**k:SimpleNamespace(document={'source_page_count':3})
package._prepare_pdf=lambda *a,**k:('stub',None)
runtime.geos_runtime=lambda:'stub'
runtime.reference_crop_sampling=lambda:'opencv4'
calls=[]
class View:
 def __init__(self,*a):pass
 def reset(self):pass
 def clear(self):pass
 def update(self,stage,*a):return stage
 def advance(self,stage,done,total):calls.append((stage,done,total))
class Engine:
 def __init__(self,*args):pass
 def set_resources(self,*args):pass
 def set_config(self,*args):pass
 def parse(self,data,options,emit,device):
  emit({'stage':'document_opened','total':3})
  for count,orientation,layout in [(2,.25,10.),(1,.125,3.)]:
   emit({'stage':'batch','detail':'test batch'})
   for stage,seconds in [('orientation',orientation),('layout',layout)]:
    emit({'stage':stage,'completed':0,'total':count,'elapsed_seconds':0.})
    emit({'stage':stage,'completed':count,'total':count,'elapsed_seconds':seconds})
  return json.dumps({'middle_json':{'pdf_info':[]},'metadata':{},'diagnostics':[],'markdown':'','content_list_json':[]}),[],[]
native.ParseProgress=View;native.DocumentEngine=Engine
messages=[];sink=logger.add(lambda m:messages.append(str(m)),format='{message}')
try:
 engine=compat.RapidDoc(formula_enable=False,table_enable=False)
 engine._models={'table_variant':'unet-slanet-plus','fallback_table_variant':None}
 engine(b'%PDF-stub')
finally:logger.remove(sink)
assert len(calls)==8
line=next(m for m in messages if'processing time ('in m)
assert 'orientation=0.375s'in line and'layout=13.000s'in line,line
'''
    subprocess.run([sys.executable,'-c',code],capture_output=True,check=True)
