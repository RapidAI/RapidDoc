"""冻结栏内几何行序的作者合同；连接文本仍用旧 Python 参考，不导入 Rust。"""
import copy,json
from common import ROOT,write_json
from rapid_doc.utils.text_boundary import merge_text_line_contents


def main():
    def line(y,text,x=10,right=240,height=20):
        return dict(points=[[x,y],[right,y],[right,y+height],[x,y+height]],text=text,confidence=None)
    def page(lines,label='text'):
        return dict(width=600,height=500,coordinate_space='page_pixels',blocks=[dict(layout_index=7,
            kind='text',region=dict(label=label,coordinate=[0,0,600,500]),lines=lines,text=None)])
    cases=[]
    def add(name,lines,groups,diagnostics=(),label='text',split_columns=True,kinds=None):
        cases.append(dict(name=name,page=page(lines,label),groups=groups,split_columns=split_columns,
            kinds=kinds or ['title'if label=='doc_title'else 'algorithm'if label=='algorithm'else 'text']*len(groups),
            texts=[('\n'.join(lines[i]['text']for i in g)if label=='algorithm'else
                    merge_text_line_contents([lines[i]['text']for i in g]))for g in groups],diagnostics=sorted(diagnostics)))
    add('reversed-single',[line(70,'three'),line(10,'one'),line(40,'two')],[[1,2,0]],['lines_reordered'])
    add('already-ordered',[line(10,'one'),line(40,'two')],[[0,1]])
    columns=[line(40,'left two'),line(40,'right two',330,560),line(10,'left one'),line(10,'right one',330,560)]
    add('shuffled-columns',columns,[[2,0],[3,1]],['columns_split','lines_reordered'])
    add('no-column-split',columns,[[0],[1],[2],[3]],['line_order_preserved_overlap'],split_columns=False)
    add('same-row',[line(10,'right',100),line(10,'left',10,90)],[[0],[1]],['line_order_preserved_overlap'])
    add('overlapping-rows',[line(40,'later'),line(10,'earlier',height=35)],[[0],[1]],['line_order_preserved_overlap'])
    add('small-overlap',[line(27,'two'),line(10,'one')],[[1,0]],['lines_reordered'])
    add('blank-barrier',[line(70,'three'),line(40,''),line(10,'one')],[[0],[1],[2]],['line_order_preserved_blank'])
    tilted=line(40,'vertical');tilted['points']=[[10,60],[10,40],[30,40],[30,60]]
    add('non-horizontal',[tilted,line(10,'one')],[[0],[1]],['line_order_preserved_non_horizontal'])
    add('algorithm',[line(40,'    second'),line(10,'first:')],[[0,1]],['line_order_preserved_algorithm'],label='algorithm')
    add('vertical-label',columns,[[0],[1],[2],[3]],['line_order_preserved_vertical_text'],label='vertical_text')
    add('title',[line(40,'continued title'),line(10,'Title')],[[1,0]],['lines_reordered'],label='doc_title')
    add('partial-column',[line(40,'left two'),line(20,'right top',330,560),line(10,'left one'),line(30,'right bottom',330,560)],
        [[2,0],[1,3]],['columns_split','lines_reordered','line_order_preserved_overlap'])
    lists=json.loads((ROOT/'build/migration/list-paragraphs.json').read_text())
    values=lists['cases'][0]['page']['blocks'][0]['lines'];permutation=[3,5,1,0,4,2]
    add('shuffled-list',[copy.deepcopy(values[i])for i in permutation],[[3,2,5],[0,4],[1]],['lines_reordered'],kinds=['list_item','list_item','text'])
    write_json(ROOT/'build/migration/line-order.json',dict(cases=cases,runtime_fixture='list-paragraphs',permutation=permutation))
    print(f'Exported {len(cases)} independent geometric line-order contracts')


if __name__=='__main__':main()
