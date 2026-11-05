"""Freeze old visual-line/typography preparation, without changing or mocking merging."""
import copy,json,random,unicodedata
from dataclasses import asdict
from pathlib import Path
from common import ROOT,digest,write_json
from export_pdf_character_primitives import encode
METRICS=('effective_height','em_height','median_glyph_width','font_signature','font_coverage','dominant_font_weight','leading_emphasis_width','leading_typography_width','paragraph_terminal')
LINE=('text','bbox','angle','source_index','visual_row_id','chars')+METRICS

def main():
    target=ROOT/'build/migration/table-script-preparation.json';fold=ROOT/'rust/rapiddoc/src/processing/table/table_text_styles/font_casefold.json'
    if target.exists()or fold.exists():raise FileExistsError('frozen evidence already exists')
    assert unicodedata.unidata_version=='16.0.0'
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.native_text import _fill_native_typography
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_text_styles import _table_char_map,_cell_glyphs,_cell_visual_lines,_non_grid_fraction_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import _fraction_member_indices,_prepare_fraction_rules
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableResult,NativeTableText,NativeTableCell,NativeTableGlyph
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    metrics=[];preparation=[]
    def metric(label,request):
        l=copy.deepcopy(request['line']);line=_LineItem(text=l['text'],bbox=tuple(l['bbox']),angle=l.get('angle',0),source_index=0,chars=l['chars'],em_height=l.get('em_height',0.))
        _fill_native_typography(line,tuple(request['page_size']))
        metrics.append(dict(label=label,request=encode(request),expected=encode({k:getattr(line,k)for k in METRICS})))
    def prep(label,v,r,tight):
        ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v['chars']),tuple(NativeTableRule(**q)for q in v['drawing_lines']))
        text=NativeTableText(tuple(NativeTableGlyph(**g)for g in r['text']['glyphs']),(),r['text']['median_glyph_width'],r['text']['median_glyph_height'])
        result=NativeTableResult(r['html'],r['rows'],r['cols'],tuple(NativeTableCell(**c)for c in r['cells']),text,r['source'],r['confidence'])
        chars=_table_char_map(ti.chars);rules=_non_grid_fraction_rules(ti,result);prepared=_prepare_fraction_rules(rules,ti.page_size,ti.angle);cells=[]
        for c in result.cells:
            gs=_cell_glyphs(result,c);cs=[chars[g.source_index]for g in gs if g.source_index in chars];lines=_cell_visual_lines(gs,chars,ti.page_size,ti.angle)
            members=_fraction_member_indices(ti.page_size,cs,tight,rules,ti.angle,prepared)
            cells.append(dict(row=c.row,col=c.col,glyphs=[asdict(g)for g in gs],chars=cs,visual_lines=[{k:getattr(l,k)for k in LINE}for l in lines],fraction_members=sorted(members)))
        preparation.append(dict(label=label,request=encode(dict(table_input=v,result=r,tight_bboxes=tight)),expected=encode(dict(fraction_rules=[asdict(q)for q in rules],cells=cells))))
    rng=random.Random(92176);names=['Arial','ABCDEF+Arial','Bold','Arial-Italic','Serif','Ｓerif','Straße','STRASSE','Σ','ς','_ -','İ','I\u0307','ﬃ','FFI','ABCDEF+Foo-Bar','Foo_bar','']
    for n in range(800):
        a=(0,90,180,270)[n%4];cs=[]
        text=rng.choice(['ABcd xyz','标题正文','body.12','body.١٢','body.12,–3','正文。】','body?','A B\nC','body!1–','word.1A','ＡBxy'])
        for i,t in enumerate(text):
            h=6. if i>=len(text)-2 and n%3==0 else rng.choice([8.,10.,12.]);x=float(i*8-5 if n%13==0 else i*8);b=_rotate_bbox_from_upright((x,50.-h,x+6.,50.),(300.,400.),a)
            name=names[n%len(names)]if i<len(text)//2 else names[(n+3)%len(names)]
            cs.append(dict(char=t,char_idx=i,bbox=list(b),font=dict(name=name,flags=rng.choice([0,64,262144,True,'64.9','٦٤',-1,2**63+1]),weight=rng.choice([400.,700.,900.,None,'700','invalid',1e13]))))
        if n%17==0:cs.append(dict(char='X',bbox=[-30.,-20.,-10.,-5.]))
        metric(f'metrics:{n}',dict(line=dict(text=text,bbox=[0.,0.,300.,400.],angle=a,chars=cs,em_height=17. if n%11==0 else 0.),page_size=[300.,400.]))
    for a in (0,90,180,270):
        for text in ['body.1','body.１２','body.١٢','body!1–','body.12,3','body.12A','body.12 ','body。】','body?）\"']:
            cs=[dict(char=t,char_idx=i,bbox=list(_rotate_bbox_from_upright((i*8.,50.-(6. if t.isdecimal()else 10.),i*8.+6.,50.),(300.,400.),a)))for i,t in enumerate(text)]
            metric(f'terminal:{a}:{text}',dict(line=dict(text=text,bbox=[0.,0.,300.,400.],angle=a,chars=cs),page_size=[300.,400.]))
    for n,c in enumerate(json.loads((ROOT/'build/migration/native-table-engine.json').read_text(encoding='utf-8'))['cases']):
        r=c['expected']['result']
        if r is None or (n%5 and not c['label'].startswith('real:')):continue
        v=c['request']['table_input'];tight={q['char_idx']:q.get('tight_bbox',q['bbox'])for q in v['chars']if type(q.get('char_idx'))is int}
        prep(f'engine:{n}',v,r,tight)
    # Multi-row cells, missing/duplicate source maps, and boundary-vs-fraction rules.
    for n in range(64):
        a=(0,90,180,270)[n%4];size=(300.,300.);cs=[];gs=[];tight={}
        for i,t in enumerate('ABxyZZ'):
            row=0 if i<2 else 1 if i<4 else 2;x=float((i%2)*10+20);y=60. if row==0 else 51. if row==1 else 69.;h=10. if row==0 else 6.
            b=_rotate_bbox_from_upright((x,y-h,x+6.,y),size,a);id=100+i*3
            cs.append(dict(char=t,char_idx=id,bbox=list(b),font=dict(name=names[n%len(names)],flags=0,weight=400.)))
            gs.append(asdict(NativeTableGlyph(i,id,t,(x,y-h,x+6.,y),row)));tight[id]=b
        if n%8==0:cs[0].pop('char_idx')
        if n%7==0:cs.append(dict(cs[1],char='ignored'))
        if n%9==0:gs.append(dict(gs[0],text='last glyph wins',glyph_id=100))
        rules=[dict(bbox=list(_rotate_bbox_from_upright(b,size,a)),width=.5,orientation='horizontal'if a in (0,180)else'vertical')for b in [(0.,.1,100.,.3),(19.,59.7,37.,60.3),(0.,99.7,100.,99.9),(200.,150.,220.,150.5)]]
        # orientation remains the old declared rule orientation; add angle-local filtering cases too.
        if n%3==0:
            for q in rules:q['orientation']='horizontal'
        v=dict(table_bbox=[0.,0.,300.,300.],page_size=list(size),angle=a,chars=cs,drawing_lines=rules,rectangles=[])
        r=dict(html='<table/>',rows=1,cols=1,cells=[asdict(NativeTableCell(0,0,1,1,(0.,0.,100.,100.),'ABxyZZ',tuple(g['source_index']for g in gs)))],text=dict(glyphs=gs,rows=[],median_glyph_width=6.,median_glyph_height=10.),source='vector_grid',confidence=1.)
        prep(f'multiline:{n}',v,r,tight)
    mapping={chr(i):chr(i).casefold()for i in range(0x110000)if not 0xd800<=i<=0xdfff and chr(i).casefold()!=chr(i)}
    fold.write_text(json.dumps(mapping,ensure_ascii=True,separators=(',',':')),encoding='utf-8')
    refs=['native_text.py','typography.py','table_text_styles.py','geometry.py','inline/scripts.py']
    sources={f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{p}':digest((ROOT/'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf'/p).read_bytes())for p in refs}
    write_json(target,dict(sources=sources,metrics=metrics,preparation=preparation,casefold=mapping,unicode_version=unicodedata.unidata_version))
    print('Frozen',len(metrics),'typography and',len(preparation),'cell preparation references;',len(mapping),'Unicode16 casefold mappings')
if __name__=='__main__':main()
