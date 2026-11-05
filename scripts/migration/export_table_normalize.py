"""冻结旧表格最终CJK/BeautifulSoup归一化与解析数据；不覆写已有参考。"""
import html,html.entities,json,random
from importlib.metadata import version
from bs4 import BeautifulSoup
from bs4.dammit import EntitySubstitution,UnicodeDammit
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/table-normalize.json';rules=ROOT/'rust/rapiddoc/src/processing/table/table_html_rules.json'
    if target.exists():raise FileExistsError('frozen table normalization reference exists')
    from rapid_doc.model.table.utils import normalize_table_cell_text,normalize_table_html_cell_text
    builder=BeautifulSoup('','html.parser').builder
    rule_data=dict(named=EntitySubstitution.HTML_ENTITY_TO_CHARACTER,attr_named=html.entities.html5,c1={str(k):v.decode('utf-8')for k,v in UnicodeDammit.WINDOWS_1252_TO_UTF8.items()if 128<=k<=159},invalid_attrs=sorted(html._invalid_codepoints),attr_replacements={str(k):v for k,v in html._invalid_charrefs.items()},void=sorted(builder.empty_element_tags),list_attrs={k:sorted(v)for k,v in builder.cdata_list_attributes.items()})
    if rules.exists():assert json.loads(rules.read_text(encoding='utf-8'))==rule_data
    else:write_json(rules,rule_data)
    spaces=[chr(i)for i in range(0x110000)if chr(i).isspace()]+['\u200b','\ufeff','', ' \n\t ']
    anchors=['中','文','㐀','\u33ff','\ua000','A','0','$','，','。','（','）','<','🙂']
    texts=[]
    for a in anchors:
        for b in anchors:
            for space in spaces:
                for prefix in ['', '中:']:
                    value=prefix+a+space+b;texts.append(dict(input=value,expected=normalize_table_cell_text(value)))
    strings=[]
    bodies=['中 文','中 A 1 ， $ 文','中<b>文 字</b> A 文','<!--中 文-->','<![CDATA[中 文]]>','中 &amp; 文','&#x4e2d; &#25991;','<b>中 文</b>','<img src="x">中 文','中<br>文 字','\n\t','中 \x1c 文']
    templates=['<table><tr><td>{}</td></tr></table>','<HTML><BODY><TABLE><TR><TH Z="1" a=2>{}</TH></TR></TABLE></BODY></HTML>','<td><td>{}</td>','<table><tr><td>{}','<table><td>first</td></table><td>{}</td>','<!DOCTYPE html><table><tr><td>{}</td></tr></table>','<td title="中 文" class=" b  a ">{}</td>','<pre>  \n </pre><textarea> \n </textarea><td>{}</td>','<script>if (a < b) x="中 文";</script><td>{}</td>','<style>x{{color:red}}</style><td>{}</td>']
    for template in templates:
        for body in bodies:strings.append(template.format(body))
    for ref in ['&amp','&unknown;','&copycat;','&notit;','&nbsp;','&#0;','&#128;','&#xD800;','&#xFFFF;','&#13;','&#x4e2d','&#65B;','&amp.','&amp-']:
        strings.append('<td title="'+ref+'">'+ref+'</td><td>中 文</td>')
    for name in EntitySubstitution.HTML_ENTITY_TO_CHARACTER:
        strings.append('<td>&'+name+';</td><td>中 文</td>')
    for attrs in ['B=2 a=1','class="b\u3000 a" rel=" b  a "','data-x="a&amp;b" data-y="&quot;\'"','title="&notit;"','rowspan=2 colspan=3','a=first a=last','checked','a="中 文"','id=x/','headers="x\x1cy"','data-x="\n x \t"']:
        strings.append('<TABLE><TR><TD '+attrs+'>中 文</TD></TR></TABLE>')
    refs=json.loads((ROOT/'build/migration/legacy-table-match.json').read_text(encoding='utf-8'))
    real=[]
    for c in refs['real']:real.append(dict(source=c['source'],variant=c['variant'],raw_html=c['expected']['raw_html'],expected=normalize_table_html_cell_text(c['expected']['raw_html'])))
    cases=[]
    for s in strings:
        try:cases.append(dict(input=s,expected=normalize_table_html_cell_text(s)))
        except ValueError:cases.append(dict(input=s,raises=True))
    write_json(target,dict(texts=texts,html=cases,real=real,sources={'rapid_doc/model/table/utils.py':digest((ROOT/'rapid_doc/model/table/utils.py').read_bytes())},rules_sha256=digest(rules.read_bytes()),beautifulsoup=version('beautifulsoup4'),scope='Actual old default html.parser, direct td/th NavigableString children only, identity when no change; canonical serialization only when changed. Not arbitrary custom parser/builder/formatter configuration.'))
    print(f'Frozen {len(texts)} CJK strings/{len(cases)} HTML cases/{len(real)} actual model final HTML outputs')
if __name__=='__main__':main()
