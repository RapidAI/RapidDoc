"""Freeze the current OMML dictionaries and per-character pylatexenc behavior.

The Python implementation encodes each scalar separately; therefore dictionary
keys plus every scalar whose NFC form changes completely cover non-identity
encoder inputs. Unknown scalars remain unchanged, just as in the source.
"""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    from pylatexenc.latexencode import get_builtin_conversion_rules, RULE_DICT
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.equation import omml,latex_dict
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    assert unicodedata.unidata_version=='16.0.0'
    assert importlib.metadata.version('pylatexenc')=='2.10'
    keys=set(range(128))
    for rule in get_builtin_conversion_rules('defaults'):
        assert rule.rule_type==RULE_DICT,'New regex/callable rules require an explicit migration'
        keys.update(rule.rule)
    keys.update(ord(c) for c in latex_dict.T)
    keys.update(ord(c) for c in latex_dict.CHARS)
    for code in range(0x110000):
        if 0xD800<=code<=0xDFFF:continue
        char=chr(code)
        if unicodedata.normalize('NFC',char)!=char:keys.add(code)
    converter=omml.oMath2Latex.__new__(omml.oMath2Latex)
    unicode={chr(code):converter.process_unicode(chr(code)) for code in sorted(keys)}
    unicode={char:value for char,value in unicode.items() if char!=value}
    maps={name:value for name,value in vars(latex_dict).items() if name.isupper() and isinstance(value,dict)}
    maps['SCR_TO_LATEX']=omml.SCR_TO_LATEX
    templates={name:value for name,value in vars(latex_dict).items() if name.isupper() and isinstance(value,str)}
    payload=(json.dumps(dict(unicode=unicode,maps=maps,templates=templates),ensure_ascii=False,separators=(',',':'))+'\n').encode()
    target=ROOT/'rust/rapiddoc-office/src/equation/omml_data.json'
    if args.check:
        assert target.read_bytes()==payload,'OMML data is stale'
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
    print('OMML data:',len(unicode),'non-identity scalars;',len(payload),'bytes')


if __name__=='__main__':main()
