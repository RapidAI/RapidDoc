"""Additional frozen region bounds and mixed year-word special paths; keep prior references immutable."""
import copy
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_final_candidate_producers import base,glyphs,reference
from export_supplemental_rule_tables import line
def regions(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.engine import coerce_native_table_rules,coerce_native_table_rectangles
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.aligned_numeric import regroup_numeric_text,numeric_row_groups,numeric_region_bounds
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.sparse_common import _local_rules
    from export_native_tiled_tables import old_source
    s=old_source(q['source']);w,h=s.page_size;inp=NativeTableInput((0.,0.,w,h),s.page_size,0,tuple(s.chars),coerce_native_table_rules(s.drawing_lines),coerce_native_table_rectangles(s.path_infos));text=build_native_table_text(inp)
    if text is None:return None
    t=regroup_numeric_text(text);rs=_local_rules(inp,w,h);out=[b for rows,anchors in numeric_row_groups(t)if len(rows)>=3 and len(anchors)>=3 if(b:=numeric_region_bounds(t,rows,anchors,rs))is not None]
    return clean(dict(table_input=asdict(inp),text=asdict(text),expected=out))
def main():
    target=ROOT/'build/migration/native-final-candidate-specials.json'
    if target.exists():raise FileExistsError(target)
    cases=[];region_cases=[]
    def add(label,q):q=clean(copy.deepcopy(q));cases.append(dict(label=label,request=q,expected=reference(q)))
    for mode in ['base','top-prose','tail-label','tail-number','wide-label','missing-header','thin-physical','extra-header','border-away','header-crossing','left-hanging']:
        q=base('aligned_numeric');s=q['source']
        if mode=='top-prose':s['lines'].insert(0,line('These ordinary words cross all columns of the table.',[40.,75.,380.,85.],50))
        if mode=='tail-label':s['lines'].append(line('continued',[50.,167.,120.,175.],50))
        if mode=='tail-number':s['lines'].append(line('12',[210.,167.,240.,175.],50))
        if mode=='wide-label':s['lines'][4]['bbox'][2]=250.
        if mode=='missing-header':s['lines']=s['lines'][4:]
        if mode=='thin-physical':s['drawing_lines']=s['drawing_lines'][:1]
        if mode=='extra-header':s['lines'].insert(0,line('Group',[200.,80.,360.,88.],50))
        if mode=='border-away':s['drawing_lines'][-1]['bbox'][1]+=25.;s['drawing_lines'][-1]['bbox'][3]+=25.
        if mode=='header-crossing':s['lines'][0]['bbox'][2]=250.
        if mode=='left-hanging':s['lines'].append(line('Section',[30.,108.,120.,114.],50))
        glyphs(s);add('aligned-'+mode,q);r=regions(q)
        if r is not None:r['label']=mode;region_cases.append(r)
    for label,years in [('plain','2020 2021 2022'),('range','2020-2021 2022–2023 2024A1'),('unicode','٢٠٢٠ ٢٠٢١ ٢٠٢٢'),('not-years','Name Other Labels')]:
        q=base('captioned_mixed');s=q['source'];s['lines']=[line('Table 1: Results',[50.,75.,240.,85.],0),line('Item',[50.,110.,90.,120.],1),line('Counts',[150.,95.,390.,105.],2),line(years,[150.,110.,390.,120.],3)]
        for i in range(3):s['lines']+=[line('Alpha',[50.,135.+20*i,90.,145.+20*i],4+4*i)]+[line('10',[150.+100*j,135.+20*i,180.+100*j,145.+20*i],5+4*i+j)for j in range(3)]
        glyphs(s);h=s['lines'][3];h['chars']=[]
        for j,word in enumerate(years.split()):
            if j:h['chars'].append(dict(char=' ',bbox=[140.+100*j,110.,145.+100*j,120.]))
            for i,t in enumerate(word):h['chars'].append(dict(char=t,bbox=[150.+100*j+4*i,110.,154.+100*j+4*i,120.]))
        s['chars']=[]
        for l in s['lines']:
            for c in l['chars']:c['char_idx']=len(s['chars']);s['chars'].append(c)
        add('mixed-word-parts-'+label,q)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_detection.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/_table_recovery/aligned_numeric.py'];write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases,regions=region_cases));print('Frozen',len(cases),'special paths;',sum(bool(c['expected']['candidates'])for c in cases),'positive;',len(region_cases),'region facts')
if __name__=='__main__':main()
