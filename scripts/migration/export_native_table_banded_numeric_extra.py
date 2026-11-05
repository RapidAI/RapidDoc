"""补充旧分带数值的落格拒绝、严格发布、低于100%的顺序分及原始字符来源边界。"""
import copy,json
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    path=ROOT/'build/migration/native-table-banded-numeric-extra.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import banded_numeric as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    f=json.loads((ROOT/'build/migration/native-table-banded-numeric.json').read_text(encoding='utf-8'));base=next(c['request']for c in f['cases']if c['label']=='grid-4-1-4-fill');cases=[];errors=[]
    def add(r,label):
        v=r['table_input'];q=r['text'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v.get('chars',())),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']));tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height']);d={}
        try:c=old.build_banded_numeric_candidate(ti,tx,d)
        except (ValueError,TypeError)as e:errors.append(dict(label=label,request=r,expected_error=type(e).__name__));return
        cases.append(dict(label=label,request=r,expected=dict(candidate=None if c is None else asdict(c),diagnostics=copy.deepcopy(d),html=None if c is None else serialize_candidate_html(c))))
    r=copy.deepcopy(base);r['text']['glyphs'][5]['bbox']=[55.,30.,70.,38.];add(r,'body-glyph-split')
    r=copy.deepcopy(base);r['text']['rows'][1]['glyph_ids'].remove(5);add(r,'body-empty-column')
    r=copy.deepcopy(base);r['text']['glyphs'][5]['text']='A';add(r,'body-nonnumeric-glyph')
    r=copy.deepcopy(base);r['text']['glyphs'][1]['bbox']=[55.,6.,70.,14.];add(r,'header-glyph-split')
    def expanded():
        r=copy.deepcopy(base);q=r['text'];oldgs=copy.deepcopy(q['glyphs']);oldchars=copy.deepcopy(r['table_input']['chars']);q['glyphs']=[];r['table_input']['chars']=[]
        for row in q['rows']:
            row['glyph_ids']=[i*5+j for i in row['glyph_ids']for j in range(5)]
            for t in row['tokens']:t['glyph_ids']=[i*5+j for i in t['glyph_ids']for j in range(5)];t['source_char_indices']=t['glyph_ids'].copy()
        for g,c in zip(oldgs,oldchars):
            for j in range(5):
                g=copy.deepcopy(g);c=copy.deepcopy(c);id=len(q['glyphs']);g.update(glyph_id=id,source_index=id);c['char_idx']=id;q['glyphs'].append(g);r['table_input']['chars'].append(c)
        return r
    r=expanded();ghost=copy.deepcopy(r['text']['glyphs'][0]);ghost.update(glyph_id=100,source_index=100,text='X',bbox=[241.,6.,242.,14.]);r['text']['glyphs'].append(ghost);add(r,'builder-accepted-incomplete-capture')
    r=expanded();r['text']['rows'][0]['glyph_ids'].append(20);add(r,'order-less-than-one-accepted')
    for kind in ('negative','large','duplicate','numeric-string','missing','partial-missing','reversed-raw','header-raw-missing','header-raw-changed','body-raw-changed'):
        r=copy.deepcopy(base)
        if kind in ('negative','large','numeric-string'):
            ids={i:(-100+i if kind=='negative'else 2**32+i*8 if kind=='large'else i)for i in range(20)}
            for g in r['text']['glyphs']:g['source_index']=ids[g['source_index']]
            for row in r['text']['rows']:
                for t in row['tokens']:t['source_char_indices']=[ids[i]for i in t['source_char_indices']]
            for c in r['table_input']['chars']:c['char_idx']=str(ids[c['char_idx']])if kind=='numeric-string'else ids[c['char_idx']]
        elif kind=='duplicate':r['table_input']['chars'].append(copy.deepcopy(r['table_input']['chars'][0]))
        elif kind in ('missing','partial-missing'):
            for c in r['table_input']['chars'][::1 if kind=='missing'else 2]:c.pop('char_idx')
        elif kind=='reversed-raw':r['table_input']['chars'].reverse()
        elif kind=='header-raw-missing':r['table_input']['chars']=r['table_input']['chars'][4:]
        elif kind=='header-raw-changed':r['table_input']['chars'][0]['char']='Changed'
        elif kind=='body-raw-changed':r['table_input']['chars'][5]['char']='Changed'
        add(r,'raw-'+kind)
    for space in (' ','\t','\u00a0','\n'):
        r=copy.deepcopy(base)
        for g in r['text']['glyphs']:g['source_index']*=3
        for row in r['text']['rows']:
            for t in row['tokens']:t['source_char_indices']=[i*3 for i in t['source_char_indices']]
        for c in r['table_input']['chars']:c['char_idx']*=3
        a=r['text']['glyphs'][5];a.update(text='1',bbox=[66.,30.,73.,38.]);b=copy.deepcopy(a);b.update(glyph_id=20,source_index=17,text='2',bbox=[73.5,30.,82.,38.]);r['text']['glyphs'].append(b);r['text']['rows'][1]['glyph_ids'].append(20);t=r['text']['rows'][1]['tokens'][1];t['glyph_ids'].append(20);t['source_char_indices'].append(17)
        r['table_input']['chars'][5].update(char='1',bbox=[66.,30.,73.,38.]);r['table_input']['chars'].extend([dict(char_idx=16,char=space,bbox=[73.,30.,73.,30.]),dict(char_idx=17,char='2',bbox=[73.5,30.,82.,38.])]);add(r,f'explicit-space-{space!r}')
    for value in (None,'bad',1.5,'1.0'):
        r=copy.deepcopy(base);r['table_input']['chars'][0]['char_idx']=value;add(r,f'raw-index-{value!r}')
    src='rapid_doc/_vendor/docvortex_core/';paths=['analyzers/native/pdf/_table_recovery/'+p for p in ('banded_numeric.py','candidate.py')]+['analyzers/native/pdf/spatial_text.py'];write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in paths},cases=cases,errors=errors));print(f'Frozen {len(cases)} extra cases ({sum(c["expected"]["candidate"]is not None for c in cases)} nonempty) +{len(errors)} old raw-index errors')
    for c in cases[:6]:print(c['label'],c['expected']['diagnostics'].get('first_rejection_gate'),None if c['expected']['candidate']is None else c['expected']['candidate']['order_consistency'])
if __name__=='__main__':main()
