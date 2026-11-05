"""Full unruled numeric and captioned text-pair producer references, with glyph evidence."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects
from export_native_tiled_tables import old_source
from export_supplemental_rule_tables import line
from export_closed_rule_pipeline import dumped
def base(kind):
    s=dict(page_size=[400.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[])
    if kind=='unruled_numeric':
        h=line('Header A B',[50.,100.,280.,110.],0,font_signature=['Arial-Bold',0],font_coverage=1.);h['chars']=[dict(char='A',bbox=[x,100.,x+5.,110.])for x in [50.,150.,250.]];s['lines']=[h]
        for i in range(3):s['lines']+=[line('10',[50.+100*j,115.+20*i,80.+100*j,125.+20*i],1+i*3+j)for j in range(3)]
    else:
        s['drawing_lines']=[dict(bbox=[40.,y,340.,y],width=0.,orientation='horizontal')for y in [100.,220.]];b=[40.,100.,340.,120.];s['path_infos']=[dict(bbox=b,rectangle_bboxes=[b],fill_visible=True,stroke_visible=False,fill_rgba=[100,100,100,255],segment_count=5,form_depth=0,source_index=0)]
        s['lines']=[line('Left Right',[50.,105.,270.,115.],0)]
        for i in range(4):
            for j,(left,right)in enumerate([(50.,180.),(200.,270.)]):
                l=line('Label',[left,130.+25*i,right,140.+25*i],1+i*2+j);l['chars']=[dict(char='a',bbox=l['bbox'])];s['lines'].append(l)
        s['lines'].append(line('Table 1: Data',[40.,225.,240.,235.],20))
    return dict(kind=kind,source=s,excluded_bboxes=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_unruled_numeric_column_tables,_detect_captioned_pair_text_tables
    source=old_source(q['source']);source.lines=objects(q['source']['lines']);source.image_bboxes=[]
    f=_detect_unruled_numeric_column_tables if q['kind']=='unruled_numeric'else _detect_captioned_pair_text_tables
    return clean(dict(candidates=[dumped(c)for c in f(source,q.get('excluded_bboxes',[]))]))
def main():
    target=ROOT/'build/migration/native-text-numeric-candidates.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    for kind in ['unruled_numeric','captioned_pair']:
        q=base(kind);assert reference(q)['candidates'],kind;add(kind,q)
        for count in range(11):q=base(kind);q['source']['lines']=q['source']['lines'][:count];add(kind+'-count-'+str(count),q)
        for semantic in [None,'header','footer','text','']:
            q=base(kind)
            for l in q['source']['lines']:l['semantic_type']=semantic
            add(kind+'-semantic-'+str(semantic),q)
        for ex in [[],[[0.,0.,400.,600.]],[[40.,100.,340.,220.]]]:q=base(kind);q['excluded_bboxes']=ex;add(kind+'-exclude-'+str(ex),q)
    for value in ['10','-10','−1','+1','1.2','1,000','10kg','10abcdefgh','10abcdefghi','10公斤','10 %','10‰','(10)','10-20','1e3','١٢','1.','*1',' 10 ','10\nkg']:
        q=base('unruled_numeric')
        for l in q['source']['lines'][1:]:l['text']=value
        add('numeric-'+repr(value),q)
    for mode in ['coverage-low','coverage-edge','not-bold','no-glyph','misaligned-glyph','null-char','compact-formula','caption-start','header-terminal','bad-color','missing-right-glyph','wide-left','header-two','glyph-empty']:
        kind='captioned_pair'if mode in ['bad-color','missing-right-glyph','wide-left','header-two','glyph-empty']else'unruled_numeric';q=base(kind);s=q['source']
        if mode=='coverage-low':s['lines'][0]['font_coverage']=.74999
        if mode=='coverage-edge':s['lines'][0]['font_coverage']=.75
        if mode=='not-bold':s['lines'][0]['font_signature']=['Arial',0]
        if mode=='no-glyph':s['lines'][0]['chars']=[]
        if mode=='misaligned-glyph':s['lines'][0]['chars'][1]['bbox'][0]+=3.
        if mode=='null-char':s['lines'][0]['chars'][1]['char']=None
        if mode=='compact-formula':s['lines'][1]['compact_formula_cluster']=True
        if mode=='caption-start':s['lines'][0]['caption_start']=True
        if mode=='header-terminal':s['lines'][0]['paragraph_terminal']=True
        if mode=='bad-color':s['path_infos'][0]['fill_rgba']=[121,121,121,255]
        if mode=='missing-right-glyph':s['lines'][2]['chars']=[]
        if mode=='wide-left':
            for l in s['lines'][1:-1:2]:l['chars'][0]['bbox'][2]=201.
        if mode=='header-two':s['lines'].append(line('Other',[280.,105.,320.,115.],21))
        if mode=='glyph-empty':
            for l in s['lines']:l['chars']=[]
        add(mode,q)
    rng=random.Random(271033)
    for i in range(70):
        q=base(rng.choice(['unruled_numeric','captioned_pair']));s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines']);l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/detection.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'facts;',[(k,sum(bool(c['expected']['candidates'])for c in cases if c['request']['kind']==k))for k in ['unruled_numeric','captioned_pair']])
if __name__=='__main__':main()
