"""冻结重复列数平局的首遇优先，不能擅自取最大列数。"""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-banded-numeric-ties.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import banded_numeric as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    f=json.loads((ROOT/'build/migration/native-table-banded-numeric.json').read_text(encoding='utf-8'));base=next(c['request']for c in f['cases']if c['label']=='grid-4-1-4-fill');cases=[]
    for first_coarse in (True,False):
        r=copy.deepcopy(base);q=r['text'];q['glyphs']=[];q['rows']=[];r['table_input']['chars']=[];r['table_input']['table_bbox'][3]=174.;r['table_input']['drawing_lines']=[dict(bbox=[0.,y,240.,y+.1],width=.1,orientation='horizontal')for y in (0.,173.9)]
        for i in range(7):
            row=copy.deepcopy(base['text']['rows'][0 if i==0 else 1]);row['row_index']=i;row['bbox']=[0.,i*24.+6.,240.,i*24.+14.];row['glyph_ids']=[]
            for c,t in enumerate(row['tokens']):
                id=len(q['glyphs']);g=copy.deepcopy(base['text']['glyphs'][c if i==0 else c+4]);g.update(glyph_id=id,source_index=id,visual_row=i);g['bbox'][1]=i*24.+6.;g['bbox'][3]=i*24.+14.;q['glyphs'].append(g);row['glyph_ids'].append(id);t.update(bbox=g['bbox'].copy(),glyph_ids=[id],source_char_indices=[id]);r['table_input']['chars'].append(dict(char_idx=id,char=g['text'],bbox=g['bbox'].copy()))
            if i>0 and (i<=3)==first_coarse:
                a,b=row['tokens'][2:];row['tokens']=row['tokens'][:2]+[dict(text=a['text']+b['text'],bbox=[a['bbox'][0],a['bbox'][1],b['bbox'][2],b['bbox'][3]],glyph_ids=a['glyph_ids']+b['glyph_ids'],source_char_indices=a['source_char_indices']+b['source_char_indices'])]
            q['rows'].append(row)
        v=r['table_input'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),0,tuple(v['chars']),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']));tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height']);d={};c=old.build_banded_numeric_candidate(ti,tx,d);assert c is not None;cases.append(dict(label=f'modal-column-first-coarse-{first_coarse}',request=r,expected=dict(candidate=asdict(c),diagnostics=d,html=serialize_candidate_html(c))))
    src='rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/banded_numeric.py';write_json(path,dict(sources={src:digest((ROOT/src).read_bytes())},cases=cases));print('Frozen mode ties:',[c['expected']['candidate']['cols']for c in cases])
if __name__=='__main__':main()
