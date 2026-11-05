"""Freeze complete original table-note selection, marker syntax and Unicode16 categories."""
import copy,random,unicodedata
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
from export_native_rule_annotation_preparation import objects as row_objects,row,fragment
from export_native_bar_graphics import objects as line_objects
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_annotations import _collect_footnote_rows,_build_table_annotation
    rows=_collect_footnote_rows(row_objects(q['rows']),line_objects(q['lines']),tuple(q['rule_bbox']),q['median_height'],set(q['core_line_indices']),tuple(q['page_size']),q.get('angle',0));annotation=_build_table_annotation('footnote',rows,excluded_line_indices=set(q['core_line_indices']))
    if annotation is not None:annotation=asdict(annotation);annotation['line_bboxes']=[[k,b]for k,b in annotation['line_bboxes'].items()]
    return clean(dict(footnote_rows=[asdict(r)for r in rows],annotation=annotation))
def base(text='Note: measured values',marker='a'):
    lines=[dict(text='12 '+marker,bbox=[50.,110.,100.,120.],source_index=0,angle=0,effective_height=10.,chars=[]),dict(text=text,bbox=[40.,205.,200.,215.],source_index=1,angle=0,effective_height=10.,font_signature=['Font',0],font_coverage=1.,chars=[]),dict(text='continued explanatory words',bbox=[40.,220.,200.,230.],source_index=2,angle=0,effective_height=10.,font_signature=['Font',0],font_coverage=1.,chars=[])]
    rows=[row([fragment(l['source_index'],l['bbox'],text=l['text'])])for l in lines]
    return dict(rows=rows,lines=lines,rule_bbox=[40.,100.,240.,200.],median_height=10.,core_line_indices=[0],page_size=[400.,600.],angle=0)
def main():
    target=ROOT/'build/migration/native-table-footnotes.json';table=ROOT/'rust/rapiddoc/src/processing/table/table_footnotes/lns.json'
    if target.exists()or table.exists():raise FileExistsError('immutable note reference/category table exists')
    assert unicodedata.unidata_version=='16.0.0'
    ranges=[];start=None
    for n in range(0x110001):
        yes=n<0x110000 and unicodedata.category(chr(n))[0]in'LNS'
        if yes and start is None:start=n
        if not yes and start is not None:ranges.append([start,n-1]);start=None
    table.parent.mkdir(parents=True,exist_ok=True);table.write_text(__import__('json').dumps(ranges,separators=(',',':')),encoding='utf-8')
    cases=[]
    def add(label,q=None):q=clean(copy.deepcopy(q or base()));cases.append(dict(label=label,request=q,expected=reference(q)))
    add('explicit-chain')
    texts=['Note: values','Notes values','Source values','Sources: values','注：测量','说明测量','注释','* words','† words','for *','ordinary prose','notes_foo','ſource: text','a) values','(a) values','1. values','① values','★ values','a b','abcd values','[ab] values','Ａ) values','a\nvalue','a) line\nsecond']
    for text in texts:add('text-'+repr(text),base(text))
    for marker in ['a','1','ab','★','Ａ','①']:
        for referenced in [False,True]:add('aux-'+marker+str(referenced),base(marker+') values',marker if referenced else'other'))
    for compact in ['a','12 a','value a','long ordinary content a that is too long','a b c d e','(a)']:q=base('a) values');q['lines'][0]['text']=compact;add('compact-'+compact,q)
    for gap in [7.49999,7.5,7.50001,12.5,12.50001]:
        for text in ['Note: values','a) values']:
            q=base(text);q['rows'][1]=row([fragment(1,[40.,200.+gap,200.,210.+gap],text=text)]);add('first-gap-'+text+str(gap),q)
    for h in [8.49999,8.5,8.50001,10.5,10.5001,11.5,11.5001]:
        for text in ['Note: values','a) values']:
            q=base(text);q['lines'][1]['effective_height']=h;add('height-'+text+str(h),q)
    for font in [None,['Font',1],['Other',0],['Font',0]]:q=base();q['lines'][2]['font_signature']=font;add('font-'+str(font),q)
    for h in [7.4999,7.5,12.5,12.5001]:q=base();q['lines'][2]['effective_height']=h;add('chain-height-'+str(h),q)
    for x in [54.9999,55.,55.0001]:q=base();q['rows'][2]=row([fragment(2,[x,220.,220.,230.],text='continued')]);add('chain-indent-'+str(x),q)
    q=base();q['core_line_indices']=[0,1];add('selected-advances-bottom',q)
    q=base();q['rows']=list(reversed(q['rows']));add('input-order-kept',q)
    q=base();q['lines'].append(dict(q['lines'][1],effective_height=30.));add('duplicate-id-last-line-wins',q)
    for nh in [8.5,8.5001]:
        q=base('a) values');q['lines'][0]['text']='This long cell cannot use compact marker tokens';q['lines'][0]['chars']=[dict(char='x',bbox=[50.,110.,60.,120.]),dict(char='a',bbox=[60.,108.,65.,108.+nh])];add('superscript-'+str(nh),q)
    q=base('a) values')
    for i in range(4):q['lines'].append(dict(text='outside body',bbox=[30.,20.+i*15.,200.,30.+i*15.],source_index=10+i,angle=0,effective_height=10.))
    add('body-height-rejects-fullsize-aux',q)
    for l in q['lines'][-4:]:l['effective_height']=20.
    add('body-height-allows-smaller-aux',q)
    rng=random.Random(261118)
    for i in range(70):
        q=base(rng.choice(['Note: values','a) values','Source: values','ordinary prose']));r=q['rows'][rng.randrange(3)];f=rng.choice(r['fragments']);f['local_bbox'][rng.randrange(4)]+=rng.choice([-.001,.001,-10.,10.,15.]);add('geometry-'+str(i),q)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.table_annotations import _extract_auxiliary_table_note_marker
    markers=[]
    for m in ['a','ab','abc','abcd','1','①','Ａ','Ⅳ','★','*','†','a#','[a','é','零']:
        for form in [m+' text','('+m+') text','['+m+'] text',m+': text',m+'\ntext']:markers.append(dict(text=form,expected=_extract_auxiliary_table_note_marker(form)))
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_annotations.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_rows.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/table_constants.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/line_layout.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/geometry.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},unicode_version=unicodedata.unidata_version,category_sha256=digest(table.read_bytes()),cases=cases,markers=markers));print('Frozen',len(cases),'full note cases;',sum(bool(c['expected']['footnote_rows'])for c in cases),'positive;',len(markers),'marker facts')
if __name__=='__main__':main()
