"""补充原始hybrid参考未覆盖的行轨拒绝、物理额外列与分段竖线，独立冻结。"""
import copy
import json
from dataclasses import asdict
from common import ROOT, digest, write_json


def main():
    path = ROOT / 'build/migration/native-table-sparse-hybrid-extra.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import sparse_hybrid as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput, NativeTableRule, NativeTableRectangle, NativeTableGlyph, NativeTableText, NativeTableTextRow, NativeTableToken
    source = json.loads((ROOT / 'build/migration/native-table-sparse-hybrid.json').read_text(encoding='utf-8'))
    base = next(c['request'] for c in source['cases'] if c['label'] == 'grid-4-0-horizontal')
    cases = []

    def add(r,label):
        v=r['table_input'];q=r['text']
        t=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],(),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']))
        text=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height'])
        d=[];cs=old.build_sparse_hybrid_candidates(t,text,d)
        cases.append(dict(label=label,request=r,expected=dict(candidates=[asdict(c)for c in cs],diagnostics=copy.deepcopy(d),html=[serialize_candidate_html(c)for c in cs])))
    for y in (-50.,0.,5.,15.,100.,101.,102.,103.,200.):
        r=copy.deepcopy(base);r['text']['rows'][1]['bbox']=[0.,y,240.,y+8.];add(r,f'row-track-{y}')
    for n in (3,4,5,6,7,10,11):
        for missing in range(1,n):
            r=copy.deepcopy(base);q=r['text'];template=copy.deepcopy(q);q['glyphs']=[];q['rows']=[]
            # 所有逻辑行均保留首列；最后一列按比例缺失，其他列维持连续dense run。
            for i in range(n):
                row=dict(row_index=i,bbox=[0.,6.+24*i,240.,14.+24*i],tokens=[],glyph_ids=[])
                for c in range(4-(i<missing)):
                    k=len(q['glyphs']);bbox=[6.+60*c,6.+24*i,22.+60*c,14.+24*i]
                    g=copy.deepcopy(template['glyphs'][0]);g.update(glyph_id=k,source_index=k,visual_row=i,bbox=bbox);q['glyphs'].append(g)
                    row['glyph_ids'].append(k);row['tokens'].append(dict(text=g['text'],bbox=bbox,glyph_ids=[k],source_char_indices=[k]))
                q['rows'].append(row)
            r['table_input']['table_bbox'][3]=n*24.+6.;r['table_input']['drawing_lines'][1]['bbox']=[0.,n*24.+5.9,240.,n*24.+6.]
            add(r,f'column-support-{n}-{missing}')
    for fraction in (.749999,.75,.979999,.98,.980001,1.):
        r=copy.deepcopy(base);h=r['table_input']['table_bbox'][3]
        for row in r['text']['rows']:
            a,b=row['tokens'][1:3];row['tokens']=[row['tokens'][0],dict(text=a['text']+b['text'],bbox=[a['bbox'][0],a['bbox'][1],b['bbox'][2],b['bbox'][3]],glyph_ids=a['glyph_ids']+b['glyph_ids'],source_char_indices=a['source_char_indices']+b['source_char_indices']),row['tokens'][3]]
        for x in (60.,120.,180.):
            r['table_input']['drawing_lines'].append(dict(bbox=[x-.1,0.,x+.1,h*fraction],width=.2,orientation='vertical'))
        add(r,f'physical-extra-col-{fraction}')
    for gap in (0.,1.,20.,25.5,25.50001):
        r=copy.deepcopy(base);h=r['table_input']['table_bbox'][3]
        for x in (60.,120.,180.):
            for a,b in ((0.,h/2.-gap/2.),(h/2.+gap/2.,h)):
                r['table_input']['drawing_lines'].append(dict(bbox=[x-.1,a,x+.1,b],width=.2,orientation='vertical'))
        add(r,f'segmented-vertical-{gap}')
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in ['sparse_hybrid.py','sparse_common.py','candidate.py','geometry.py']},cases=cases))
    print(f'Frozen {len(cases)} extra cases; {sum(bool(c["expected"]["candidates"])for c in cases)} nonempty')


if __name__=='__main__':main()
