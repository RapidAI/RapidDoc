"""固定合法 HTML5 实体字典供 Rust 解码，不依赖运行时 Python。"""
import html.entities,json
from common import ROOT

if __name__=='__main__':
    target=ROOT/'rust/rapiddoc/src/legacy_table_merge/entities.json'
    if target.exists():raise FileExistsError('Frozen entity table exists')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({k:v for k,v in html.entities.html5.items()if k.endswith(';')},ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
    print('Frozen semicolon-terminated HTML5 entities')
