"""实际旧单元格视觉组、来源片段和跨行文字；固定Unicode16 Latin/NFKC分类。"""
import copy,json,re,unicodedata
from dataclasses import asdict
from common import ROOT,digest,write_json
def main():
    target=ROOT/'build/migration/native-table-cell-text.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import text
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableGlyph
    assert unicodedata.unidata_version=='16.0.0'
    ranges=[];start=None
    for n in range(0x110001):
        ok=n<0x110000 and text._is_latin_letter(chr(n))
        if ok and start is None:start=n
        if not ok and start is not None:ranges.append((start,n-1));start=None
    source='//! Generated CPython Unicode16 NFKC/Latin classifier by export_native_table_cell_text.py.\n'
    source+='pub const LATIN: &[(u32,u32)] = &[\n'+''.join(f'({a},{b}),\n'for a,b in ranges)+'];\n'
    path=ROOT/'rust/rapiddoc-native/src/table_latin.rs'
    if path.exists():assert re.sub(r'\s+','',path.read_text(encoding='utf-8'))==re.sub(r'\s+','',source)
    else:path.write_text(source,encoding='utf-8')
    cases=[];separators=[]
    def add(glyphs,height=10.):
        parts=text.build_cell_text_parts(glyphs,height)
        lookup={id(g):i for i,g in enumerate(glyphs)}
        groups=[[lookup[id(g)]for g in row]for row in text._python_cell_visual_groups(glyphs,height)]
        cases.append(dict(request=dict(glyphs=[asdict(g)for g in glyphs],median_height=height),expected=dict(parts=parts,text=''.join(t for t,_ in parts),groups=groups)))
    prior=json.loads((ROOT/'build/migration/native-table-text.json').read_text(encoding='utf-8'))
    for c in prior['cases']:
        if c['expected']:
            p=c['expected'];glyphs=[NativeTableGlyph(**g)for g in p['glyphs']]
            add(glyphs,p['median_glyph_height'])
            for residue in range(2):add([g for i,g in enumerate(glyphs)if i%2==residue],p['median_glyph_height'])
    lines=['Alpha','Beta','中','文','Power','Supply/%','kg/','m2','co-','operate','http://example.','com','https://next','httpſ://next','May,','2024','α','β','É','Ｆ','𝓐','K','ﬃ','A1','²a','a\u3000',' word','a_','µm']
    for before in lines:
        for after in lines:
            separators.append(dict(previous=before,next=after,expected=text._cell_row_separator(before,before,after)))
            gs=[NativeTableGlyph(0,0,before,(0.,0.,10.,10.),0),NativeTableGlyph(1,1,after,(0.,20.,10.,30.),1)]
            add(gs)
    # 同一全表视觉组被其他列桥接：足高双行才细分，水印/上下标保留原组。
    for rows in [[0,0,0,0],[2,2,1,1]]:
        for h in [8.49,8.5,10.,12.5,12.51]:
            for skew in [0.,3.,3.000000001]:
                gs=[NativeTableGlyph(i,i,chr(65+i),(float(i%2*20),float(i//2*25)+(skew if i%2 else 0),float(i%2*20+10),float(i//2*25)+h+(skew if i%2 else 0)),rows[i])for i in range(4)]
                add(gs)
    add([])
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/text.py','rapid_doc/_vendor/docvortex_core/foundation/_text.py']
    write_json(target,dict(cases=cases,separators=separators,sources={p:digest((ROOT/p).read_bytes())for p in refs},unicode_sha256=digest(source.encode('utf-8')),unicode_hash_format='canonical_generator_text_lf',scope='Actual old build_cell_text_parts and Python visual refinement, including all prepared glyph fixtures and Latin/unit/URL/hyphen boundaries; given glyph inputs, no candidate generation or HTML/script materialization.'))
    print(f'Frozen {len(cases)} actual old cell text/source-parts cases and {len(separators)} row separators')
if __name__=='__main__':main()
