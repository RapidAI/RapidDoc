"""补充三线表100%文字/零歧义发布门及旧median聚类事实，不覆写主参考。"""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-rule-band-extra.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import rule_band as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.sparse_common import cluster_members
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableText,NativeTableGlyph,NativeTableTextRow,NativeTableToken
    f=json.loads((ROOT/'build/migration/native-table-rule-band.json').read_text(encoding='utf-8'));base=next(c['request']for c in f['cases']if len(c['request']['text']['rows'])==8 and all(len(r['tokens'])==3 for r in c['request']['text']['rows'])and not c['request']['table_input']['rectangles'])
    cases=[]
    for half in [0.,.1]:
        r=copy.deepcopy(base);original=list(r['text']['glyphs']);rows=r['text']['rows'];r['text']['glyphs']=[];id_=0
        for row in rows:
            ids=[]
            for g in [g for g in original if g['visual_row']==row['row_index']]:
                for _ in range(5):
                    q=g|dict(glyph_id=id_,source_index=id_);r['text']['glyphs'].append(q);ids.append(id_);id_+=1
            row['glyph_ids']=ids
            # 同步原token来源到每个重复字形，保持候选输入自洽。
            for token in row['tokens']:
                owned=[g for g in r['text']['glyphs']if g['visual_row']==row['row_index']and g['bbox']==token['bbox']]
                token['glyph_ids']=[g['glyph_id']for g in owned];token['source_char_indices']=[g['source_index']for g in owned]
        b=rows[2]['bbox'];ghost=dict(glyph_id=id_,source_index=id_,text='Z',bbox=[42.5-half,b[1],42.5+half,b[3]],visual_row=2,explicit_space_before=False,explicit_break_before=False,spacing_bbox=None)
        r['text']['glyphs'].append(ghost);rows[2]['glyph_ids'].append(id_)
        v=r['table_input'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],(),tuple(NativeTableRule(**a)for a in v['drawing_lines']),tuple(NativeTableRectangle(**a)for a in v['rectangles']))
        t=r['text'];text=NativeTableText(tuple(NativeTableGlyph(**g)for g in t['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**tok)for tok in row['tokens']))))for row in t['rows']),t['median_glyph_width'],t['median_glyph_height']);d=[];out=old.build_rule_band_candidates(ti,text,d)
        cases.append(dict(request=r,expected=dict(candidates=[asdict(c)for c in out],diagnostics=d,html=[serialize_candidate_html(c)for c in out])))
    rng=random.Random(728);clusters=[]
    for values in [[],[0.,.8,1.6],[0.,1.,2.],[1.,1.,1.],[-1e9,0.,1e9]]+[[rng.uniform(-100,100)for _ in range(rng.randrange(1,100))]for _ in range(60)]:
        for tolerance in [0.,.5,1.]:clusters.append(dict(values=values,tolerance=tolerance,expected=cluster_members(values,tolerance)))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/'
    write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in ['rule_band.py','sparse_common.py']},cases=cases,clusters=clusters))
    print(f'Frozen {len(cases)} strict-publication cases +{len(clusters)} median clustering facts; diagnostics={[c["expected"]["diagnostics"]for c in cases]}')
if __name__=='__main__':main()
