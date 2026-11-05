"""冻结旧单元格HTML片段渲染；角色为该阶段的显式输入，不冒充整表自动分类。"""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode

def main():
    target=ROOT/'build/migration/table-styled-cell.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _render_styled_cell
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableCell,NativeTableGlyph
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_cell_text_parts
    cases=[]
    def add(label,cell,glyphs,height,roles,chars,tight,origins,preserve,possible=None):
        request=dict(cell=asdict(cell),glyphs=[asdict(g)for g in glyphs],median_height=height,roles=roles,chars_by_source=chars,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve,possible_space_indices=None if possible is None else sorted(possible))
        expected=_render_styled_cell(cell,glyphs,height,roles,chars_by_source=chars,tight_bboxes=tight,origins=origins,preserve_font_styles=preserve,_possible_space_indices=possible)
        cases.append(dict(label=label,request=encode(request),expected=expected))
    f=json.loads((ROOT/'build/migration/native-table-engine.json').read_text(encoding='utf-8'))
    for n,c in enumerate(f['cases']):
        if n%11 or not c['expected']['result']:continue
        r=c['expected']['result'];raw=c['request']['table_input']['chars'];chars={q['char_idx']:q for q in raw}
        for ci,cell in enumerate(r['cells'][:3]):
            cell=NativeTableCell(**cell);glyphs=sorted([NativeTableGlyph(**g)for g in r['text']['glyphs']if g['source_index']in cell.source_char_indices],key=lambda g:(g.visual_row,g.bbox[0],g.bbox[1],g.glyph_id))
            for preserve in (False,True):
                roles={g.source_index:('sup'if i%4==0 else 'sub')for i,g in enumerate(glyphs)if i%3==0}
                add(f'engine:{n}:{ci}:{preserve}',cell,glyphs,r['text']['median_glyph_height'],roles,chars,{}, {},preserve)
    rng=random.Random(183721)
    fonts=['Arial','Arial-Bold','ABCDEF+Arial','BOLDXX+Arial','Demibold','Helvetica-W7','Foo_bdmt','Foo-BI','Foo-gbi','Foo-BLACK','demİ','Arial-Italic','Courier','monospace','Arial-BI\n']
    for n in range(1000):
        glyphs=[];chars={};tight={};origins={};roles={};height=10.
        for i in range(rng.randrange(1,22)):
            text=rng.choice('abXYZ129中文β<> &"\n');row=i//8 if n%4==0 else 0;x=float((i%8)*8 if n%4==0 else i*8);y=float(row*14)
            source=i-2 if n%13==0 else i
            g=NativeTableGlyph(i,source,text,(x,y,x+6.,y+10.),row,explicit_space_before=n%9==0 and i%4==0)
            glyphs.append(g);c=dict(char=text,char_idx=source,bbox=list(g.bbox),tight_bbox=[x,y,x+4.,y+8.],origin=[x,y+8.],writing_angle=0.,font=dict(name=rng.choice(fonts),size=10.,flags=rng.choice([0,64,262144,262208]),weight=rng.choice([None,400,599,600,700])))
            chars[source]=c;roles[source]=rng.choice(['body','body','sup','sub'])
            if n%3==0:tight[source]=[x,y,x+3.,y+8.];origins[source]=[x,y+8.]
        content=''.join(t for t,_ in build_cell_text_parts(glyphs,height))
        if n%17==0:content+=' <mismatch>&"'
        cell=NativeTableCell(0,0,1,1,(0.,0.,200.,100.),content,tuple(g.source_index for g in glyphs))
        possible=None if n%5 else {g.source_index for g in glyphs[::2]}
        add(f'styled:{n}',cell,glyphs,height,roles,chars,tight,origins,n%3!=0,possible)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_text_styles.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py','rapid_doc/_vendor/docvortex_core/document/pdf/text/spacing.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'styled-cell stages;',sum('<sup>'in c['expected']or'<sub>'in c['expected']for c in cases),'script outputs;',sum('<b>'in c['expected']or'<i>'in c['expected']for c in cases),'font outputs')
if __name__=='__main__':main()
