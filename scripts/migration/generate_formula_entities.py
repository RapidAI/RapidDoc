"""导出/校验Python html5实体名：用于保守拒绝尚未实现ftfy的公式输入，不替代实体解码。"""
import argparse
from html.entities import html5
from common import ROOT
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=ROOT/'rust/rapiddoc-inference/src/formula_html_entities.txt';text='\n'.join(sorted(html5))+'\n'
    if args.check:assert path.read_text(encoding='utf-8')==text
    else:path.write_text(text,encoding='utf-8')
    print(f'HTML entity gate: {len(html5)} Python names')
if __name__=='__main__':main()
