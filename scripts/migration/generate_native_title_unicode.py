"""Freeze Python titlecase categories for exact str.isupper display-band checks."""
import unicodedata
from common import ROOT,write_json

def main():
    write_json(ROOT/'rust/rapiddoc/src/processing/text/native_text_roles/unicode.json',dict(unicode=unicodedata.unidata_version,titlecase=[i for i in range(0x110000)if unicodedata.category(chr(i))=='Lt']))
if __name__=='__main__':main()
