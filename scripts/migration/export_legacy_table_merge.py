"""下一阶段：冻结旧跨页表格 merge_table 的完整页面 JSON，拒绝覆盖。"""
import ast,copy,importlib,json
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/legacy-table-merge.json'
    if target.exists():raise FileExistsError('Frozen baseline exists')
    # 从生产 pipeline 入口加载，避免旧模块独立导入时已有的循环依赖。
    importlib.import_module('rapid_doc.backend.pipeline.model_json_to_middle_json')
    old=importlib.import_module('rapid_doc.utils.table_merge');cases=[]
    def literal(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.BinOp)and isinstance(node.op,(ast.Add,ast.Mult)):
            left,right=literal(node.left),literal(node.right)
            return left+right if isinstance(node.op,ast.Add)else left*right
        raise ValueError('Only existing literal HTML expressions are allowed')
    test=ROOT/'tests/test_table_merge_continuation.py';pairs=[]
    for function in ast.parse(test.read_text()).body:
        if not isinstance(function,ast.FunctionDef):continue
        values=[literal(node.value.args[0])for node in function.body if isinstance(node,ast.Assign)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Name)and node.value.func.id=='build_table_state_from_html']
        if len(values)==2:pairs.append((function.name,*values))
    assert len(pairs)==5
    def text(kind,value,bbox):return dict(type=kind,bbox=bbox,lines=[dict(bbox=bbox,spans=[dict(type='text',content=value,bbox=bbox)])])
    def table(html,caption=None,post=False,footnotes=0,width=100):
        bbox=[0,0,width,80];body=dict(type='table_body',bbox=bbox,lines=[dict(bbox=bbox,spans=[dict(type='table',html=html,bbox=bbox)])])
        blocks=[body]
        if caption is not None:blocks.append(text('table_caption',caption,[0,81 if post else -10,width,90 if post else -1]))
        for i in range(footnotes):blocks.append(text('table_footnote','note'+str(i),[0,91+i,width,92+i]))
        return dict(type='table',bbox=bbox,blocks=blocks,unknown='preserve')
    def page(*blocks):return dict(para_blocks=list(blocks),preproc_blocks=[],discarded_blocks=[],page_size=[200,200])
    def add(name,pages):
        for i,p in enumerate(pages):p['page_idx']=i
        expected=copy.deepcopy(pages);old.merge_table(expected)
        cases.append(dict(name=name,pages=pages,expected=expected))
    for name,previous,current in pairs:
        for caption,post in [(None,False),('续表',False),('Table (continued)',False),('New table',False),('下一节',True)]:
            for footnotes in range(3):add(f'{name}/{caption}/{post}/notes{footnotes}',[page(table(previous,footnotes=footnotes)),page(table(current,caption,post))])
    previous='<table><tr><th>ID</th><th>Name</th></tr><tr><td>1</td><td>A</td></tr></table>'
    current='<table><tr><th>ID</th><th>Name</th></tr><tr><td>2</td><td>B<img src="images/x.png"/></td></tr></table>'
    for width in [90,90.909090909,90.909090910,99.99,100,109.99999999,110,110.00000001,120]:
        add(f'width-boundary/{width}',[page(table(previous)),page(table(current,width=width))])
    for middle in [[],[text('text','break',[0,0,100,10])],[table(current)]]:
        add('three-pages/'+str(len(cases)),[page(table(previous)),page(*middle),page(table(current))])
    add('previous-last-block-is-text',[page(table(previous),text('text','break',[0,81,100,90])),page(table(current))])
    add('current-first-block-is-text',[page(table(previous)),page(text('text','break',[0,-10,100,-1]),table(current))])
    for html in ['', '<table></table>', '<div>no rows</div>', '<table><tr><td>A</td></tr></table>']:
        add('empty-or-different-grid/'+str(len(cases)),[page(table(previous)),page(table(html))])
    for flags in [[1,1],[0,1],[0,0],[]]:
        block=table(current);block['cell_merge']=flags
        add('explicit-cell-merge/'+str(flags),[page(table(previous)),page(block)])
    add('zero-pages',[]);add('one-page',[page(table(previous))])
    paths=['rapid_doc/utils/table_merge.py','rapid_doc/backend/pipeline/pipeline_middle_json_mkcontent.py','rapid_doc/utils/char_utils.py','tests/test_table_merge_continuation.py']
    write_json(target,dict(cases=cases,sources={p:digest((ROOT/p).read_bytes())for p in paths},profile='未修改旧 merge_table 完整页面 JSON；含既有 5 个回归 HTML、续表/后置标题、脚注、宽度边界、三页、行列跨度/稀疏续行、显式 cell_merge；仅冻结基线，Rust 尚未实现'))
    print(f'Frozen {len(cases)} complete old cross-page table merge cases; Rust implementation pending')

if __name__=='__main__':main()
