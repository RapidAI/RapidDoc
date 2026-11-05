"""Freeze complete final aligned/mixed producers with real raw glyph evidence and special paths."""
import copy,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_bar_graphics import objects
from export_native_tiled_tables import old_source
from export_supplemental_rule_tables import line
from export_closed_rule_pipeline import dumped
def glyphs(s):
    chars=[]
    for l in s['lines']:
        b=l['bbox'];text=l['text'];width=(b[2]-b[0])/max(1,len(text));l['chars']=[]
        for i,t in enumerate(text):
            c=dict(char=t,bbox=[b[0]+i*width,b[1],b[0]+(i+1)*width,b[3]],char_idx=len(chars),rotation=0.)
            chars.append(c);l['chars'].append(c)
    s['chars']=chars
def base(kind):
    s=dict(page_size=[500.,600.],lines=[],drawing_lines=[],path_infos=[],chars=[],image_bboxes=[],form_bboxes=[],signature_bboxes=[])
    if kind=='aligned_numeric':
        s['lines']=[line('Metric',[50.,95.,120.,105.],0)]+[line('Year'+chr(65+j),[200.+60*j,95.,240.+60*j,105.],1+j)for j in range(3)]
        for i in range(3):s['lines']+=[line('Alpha',[50.,115.+20*i,120.,125.+20*i],4+4*i)]+[line(str(10+i),[210.+60*j,115.+20*i,240.+60*j,125.+20*i],5+4*i+j)for j in range(3)]
        s['drawing_lines']=[dict(bbox=[40.,y-.1,380.,y+.1],width=.2,orientation='horizontal')for y in [90.,170.]]
    else:
        s['lines']=[line('Table 1: Results',[50.,75.,240.,85.],0)]+[line(t,[x,100.,x+50.,110.],i+1)for i,(t,x)in enumerate([('Item',50.),('Amount',180.),('Rate',280.)])]
        for i in range(3):s['lines']+=[line('Alpha',[50.,125.+20*i,100.,135.+20*i],4+3*i),line('10',[180.,125.+20*i,210.,135.+20*i],5+3*i),line('20',[280.,125.+20*i,310.,135.+20*i],6+3*i)]
    glyphs(s);return dict(kind=kind,source=s,excluded_bboxes=[])
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_detection import _detect_aligned_numeric_tables,_detect_captioned_mixed_numeric_tables
    source=old_source(q['source']);source.lines=objects(q['source']['lines']);source.image_bboxes=q['source'].get('image_bboxes',[])
    call=_detect_aligned_numeric_tables if q['kind']=='aligned_numeric'else _detect_captioned_mixed_numeric_tables
    return clean(dict(candidates=[dumped(c)for c in call(source,q.get('excluded_bboxes',[]))]))
def main():
    target=ROOT/'build/migration/native-final-candidate-producers.json'
    if target.exists():raise FileExistsError(target)
    cases=[];errors=[]
    def add(label,q):
        q=clean(copy.deepcopy(q))
        try:expected=reference(q)
        except KeyError as exc:errors.append(dict(label=label,request=q,error='invalid_input',old_exception='KeyError',old_key=exc.args[0]));return
        cases.append(dict(label=label,request=q,expected=expected))
    for kind in ['aligned_numeric','captioned_mixed']:
        q=base(kind);assert reference(q)['candidates'],kind;add(kind,q)
        for count in [0,1,3,6,8,9,10,12,16]:q=base(kind);q['source']['lines']=q['source']['lines'][:count];add(kind+'-count-'+str(count),q)
        for ex in [[],[[0.,0.,500.,600.]],[[40.,90.,380.,170.]]]:q=base(kind);q['excluded_bboxes']=ex;add(kind+'-excluded-'+str(ex),q)
        for angle in [90,180,270]:
            q=base(kind)
            for l in q['source']['lines']:l['angle']=angle
            add(kind+'-angle-'+str(angle),q)
        for mode in ['empty-chars','empty-rules','complete-grid','long-prose','two-tables','integer-coordinates']:
            q=base(kind);s=q['source']
            if mode=='empty-chars':s['chars']=[]
            if mode=='empty-rules':s['drawing_lines']=[]
            if mode=='complete-grid':s['drawing_lines']+=[dict(bbox=[40.,y,380.,y],width=0.,orientation='horizontal')for y in [110.,130.,150.]]
            if mode=='long-prose':s['lines'][0]['text']='These ordinary words form a separate prose sentence above the table.'
            if mode=='two-tables':
                original=copy.deepcopy(s)
                for l in original['lines']:l['source_index']+=100;l['bbox'][1]+=200;l['bbox'][3]+=200
                for r in original['drawing_lines']:r['bbox'][1]+=200;r['bbox'][3]+=200
                s['lines']+=original['lines'];s['drawing_lines']+=original['drawing_lines'];glyphs(s)
            if mode=='integer-coordinates':
                for l in s['lines']:l['bbox']=[int(n)for n in l['bbox']]
            add(kind+'-'+mode,q)
    for t in ['10','(10)','(10','10)','-','–','—','*10','10%','10kg','1e3','1.','١٢','²','1,000','+10','−1','* 10']:
        for kind in ['aligned_numeric','captioned_mixed']:
            q=base(kind);s=q['source']
            for l in s['lines']:
                if l['text']in ['10','11','12','20']:l['text']=t
            glyphs(s);add(kind+'-numeric-'+repr(t),q)
    for mode in ['spanning-header','native-split','native-horizontal','note','note-continuation','note-font-break','note-multiple','caption-below','repeated-single-row','no-caption','formula','paragraph','caption-flag']:
        q=base('captioned_mixed');s=q['source']
        if mode in ['spanning-header','native-horizontal']:
            for l in s['lines'][1:]:l['bbox'][1]+=10.;l['bbox'][3]+=10.
            s['lines'].append(line('Group',[180.,95.,330.,105.],40))
        if mode=='native-horizontal':s['drawing_lines'].append(dict(bbox=[47.5,107.,332.5,107.],width=0.,orientation='horizontal'))
        if mode=='native-split':s['drawing_lines'].append(dict(bbox=[145.,100.,145.,180.],width=1.,orientation='vertical'))
        if mode.startswith('note'):
            s['lines'].append(line('*a Note about data',[50.,180.,260.,190.],40))
            if mode in ['note-continuation','note-font-break']:s['lines'].append(line('continued source explanation',[50.,192.,260.,202.],41,font_signature=['Other',0]if mode=='note-font-break'else None))
            if mode=='note-multiple':s['lines'].append(line('Note: one more data source',[50.,181.,260.,191.],41))
        if mode=='caption-below':s['lines'][0]['bbox'][1:4:2]=[195.,205.]
        if mode=='repeated-single-row':
            h=line('ItemAmountRate',[50.,250.,330.,260.],40);s['lines'].append(h)
            for j,x in enumerate([50.,180.,280.]):s['lines'].append(line('100',[x,265.,x+30.,275.],41+j))
        if mode=='no-caption':s['lines']=s['lines'][1:]
        if mode=='formula':s['lines'][1]['formula_candidate_only']=True
        if mode=='paragraph':s['lines'][1]['paragraph_terminal']=True
        if mode=='caption-flag':s['lines'][0]['caption_start']=True
        glyphs(s)
        if mode=='repeated-single-row':
            h=next(l for l in s['lines']if l['source_index']==40);h['chars']=[]
            for label,x in [('Item',50.),('Amount',180.),('Rate',280.)]:h['chars']+=[dict(char=t,bbox=[x+i*8.,250.,x+(i+1)*8.,260.])for i,t in enumerate(label)]
        add(mode,q)
    rng=random.Random(271033)
    for i in range(65):
        q=base(rng.choice(['aligned_numeric','captioned_mixed']));s=q['source'];rng.shuffle(s['lines']);rng.shuffle(s['drawing_lines']);l=rng.choice(s['lines']);l['bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-3.,3.]);add('permuted-'+str(i),q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/aligned_numeric.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rules.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,errors=errors));print('Frozen',len(cases),'facts;',[(k,sum(bool(c['expected']['candidates'])for c in cases if c['request']['kind']==k))for k in ['aligned_numeric','captioned_mixed']],'errors',len(errors))
if __name__=='__main__':main()
