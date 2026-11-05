"""Reproduce/verify the immutable Unicode16 East Asian width resource."""
import json,unicodedata
from common import ROOT
target=ROOT/'rust/rapiddoc/src/processing/document/code_containers/wide.json'
ranges=[]
for i in range(0x110000):
    if unicodedata.east_asian_width(chr(i))in {'W','F'}:
        if ranges and ranges[-1][1]+1==i:ranges[-1][1]=i
        else:ranges.append([i,i])
value=dict(unicode_version=unicodedata.unidata_version,ranges=ranges)
if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==value
else:target.write_text(json.dumps(value,separators=(',',':'))+'\n',encoding='utf-8')
print('Verified EAW',value['unicode_version'],len(ranges),'ranges')
