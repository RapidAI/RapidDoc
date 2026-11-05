"""冻结整行组标题的实际成功回退和独立空间投影/紧字框边界。"""
import copy,json,math,random
from common import ROOT,digest,write_json
from dataclasses import asdict
def main():
    path=ROOT/'build/migration/native-table-multiline-extra.json'
    if path.exists():raise FileExistsError(path)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery import sparse_multiline as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.candidate import serialize_candidate_html
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput,NativeTableRule,NativeTableRectangle,NativeTableGlyph,NativeTableText,NativeTableTextRow,NativeTableToken
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.spatial_text import project_pdf_table_text
    from rapid_doc._vendor.docvortex_core.document.pdf.text.spacing import needs_tight_space
    f=json.loads((ROOT/'build/migration/native-table-multiline.json').read_text(encoding='utf-8'));cases=[];projection=[];spacing=[]
    def add(r,label):
        v=r['table_input'];q=r['text'];ti=NativeTableInput(tuple(v['table_bbox']),tuple(v['page_size']),v['angle'],tuple(v.get('chars',())),tuple(NativeTableRule(**x)for x in v['drawing_lines']),tuple(NativeTableRectangle(**x)for x in v['rectangles']));tx=NativeTableText(tuple(NativeTableGlyph(**g)for g in q['glyphs']),tuple(NativeTableTextRow(**(row|dict(tokens=tuple(NativeTableToken(**t)for t in row['tokens']))))for row in q['rows']),q['median_glyph_width'],q['median_glyph_height']);d=[];cs=old.build_sparse_multiline_candidates(ti,tx,d);cases.append(dict(label=label,request=r,expected=dict(candidates=[asdict(c)for c in cs],diagnostics=copy.deepcopy(d),html=[serialize_candidate_html(c)for c in cs])))
    for c in f['cases']:
        if not c['label'].startswith('group-'):continue
        for width in (45.,95.,222.):
            r=copy.deepcopy(c['request'])
            for row in (1,5):
                for i in r['text']['rows'][row]['glyph_ids']:
                    g=r['text']['glyphs'][i];g['bbox'][0]=100.;g['bbox'][2]=100.+min(width,120.)
                    for raw in r['table_input']['chars']:
                        if raw.get('char_idx')==g['source_index']:raw['bbox']=g['bbox'].copy()
                t=r['text']['rows'][row]['tokens'][0];t['bbox'][0]=100.;t['bbox'][2]=100.+min(width,120.)
            add(r,'centered-'+c['label']+f'-{width}')
    # 单独正常化后的raw投影，空白/换行、连字/标点、逆框与四个角度均直接调用旧实现。
    def project(chars,bbox=None,angle=0):
        bbox=bbox or [0.,0.,240.,240.];projection.append(dict(request=dict(chars=chars,table_bbox=bbox,angle=angle),expected=project_pdf_table_text(chars,bbox,angle)))
    sample=[dict(char_idx=i,char=s,bbox=[5.+i*9.,5.,13.+i*9.,15.],tight_bbox=[5.+i*9.,5.,13.+i*9.,15.],font=dict(name='Arial',size=10.,flags=0),writing_angle=0.)for i,s in enumerate('ABCD')]
    for a in (0,90,180,270,45,-90):
        for text in ('ABCD','1⅔²四','ﬁ—“A','中 文A','A\tB\n','\r\n  ','⒈①ＡA'):
            chars=copy.deepcopy(sample)
            for c,s in zip(chars,text):c['char']=s
            project(chars,angle=a)
    for gap in (.6,2.,2.5,2.50001,3.5,3.500001,14.9999,15.,15.001,20.):
        for pending in (None,' ','\u00a0','\r','\n','\r\n'):
            chars=copy.deepcopy(sample[:2]);chars[1]['bbox']=[13.+gap,5.,21.+gap,15.];chars[1]['tight_bbox']=chars[1]['bbox'].copy();chars[1]['char_idx']=2 if pending else 1
            if pending:chars.insert(1,dict(char_idx=1,char=pending,bbox=[13.,5.,13.,5.]))
            project(chars)
    rng=random.Random(99401)
    for _ in range(140):
        chars=copy.deepcopy(sample);rng.shuffle(chars)
        for c in chars:
            dx=rng.uniform(-5,5);dy=rng.choice([0.,5.,10.,20.]);c['bbox']=[c['bbox'][0]+dx,5.+dy,c['bbox'][2]+dx,15.+dy]
        project(chars,angle=rng.choice([0,90,180,270]))
    for font in ('Arial','Mono','Courier','Consolas','Menlo','TypeWriter','Fixed','无衬线'):
        for flags in (0,1,2):
            for gap in (2.5,2.50001,3.):
                left,right=copy.deepcopy(sample[:2]);right['bbox']=[13.+gap,5.,21.+gap,15.];right['tight_bbox']=right['bbox'].copy();left['font'].update(name=font,flags=flags);right['font'].update(name=font,flags=flags);spacing.append(dict(left=left,right=right,expected=needs_tight_space(left,right)));project([left,right])
    for a in (0,math.pi/2,math.pi,math.pi*1.5,.001,.002):
        for text in ('AB','11','中A','ＡA','éA','ⅣA'):
            left,right=copy.deepcopy(sample[:2]);left['char']=text[0];right['char']=text[1];right['tight_bbox']=[17.,5.,25.,15.];left['writing_angle']=a;right['writing_angle']=a;spacing.append(dict(left=left,right=right,expected=needs_tight_space(left,right)));project([left,right])
    for change in ('index','size','origin','height','missing-font','missing-tight'):
        left,right=copy.deepcopy(sample[:2]);right['tight_bbox']=[17.,5.,25.,15.]
        if change=='index':right['char_idx']=3
        if change=='size':right['font']['size']=7.9
        if change=='origin':left['origin']=[0.,0.];right['origin']=[0.,1.500001]
        if change=='height':right['tight_bbox'][3]=21.
        if change=='missing-font':left.pop('font')
        if change=='missing-tight':left.pop('tight_bbox')
        spacing.append(dict(left=left,right=right,expected=needs_tight_space(left,right)));project([left,right])
    src='rapid_doc/_vendor/docvortex_core/';paths=['analyzers/native/pdf/_table_recovery/'+p for p in ('sparse_multiline.py','grouped_rule_band.py','sparse_common.py','candidate.py')]+['analyzers/native/pdf/spatial_text.py','document/pdf/text/spacing.py'];write_json(path,dict(sources={src+p:digest((ROOT/(src+p)).read_bytes())for p in paths},cases=cases,projection=projection,spacing=spacing));print(f'Frozen {len(cases)} extra multiline, {sum(bool(c["expected"]["candidates"])for c in cases)} nonempty, {sum(bool(c["expected"]["candidates"]) and len(c["expected"]["diagnostics"])==2 for c in cases)} grouped; {len(projection)} projections, {len(spacing)} tight-space facts')
if __name__=='__main__':main()
