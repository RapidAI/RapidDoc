"""补充冻结旧原生路由的缺省与显式null/非字符串引擎语义，不改已有策略参考。"""
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-engine-values.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc.model.docvortex_adapter import use_docvortex_table_rules
    configs=[{}]+[dict(rule_table_engine=x)for x in [None,False,0,True,{},[],'docvortex','legacy']]
    write_json(target,dict(cases=[dict(config=c,expected=bool(use_docvortex_table_rules(c)))for c in configs],source_sha256=digest((ROOT/'rapid_doc/model/docvortex_adapter.py').read_bytes())))
if __name__=='__main__':main()
