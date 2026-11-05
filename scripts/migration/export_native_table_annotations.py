"""Freeze complete old annotation transaction and front-matter failure gate."""
import copy,json,random,unicodedata,sys
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean

def main():
    types_only='--types-only'in sys.argv
    target=ROOT/('build/migration/native-table-annotations-types.json'if types_only else'build/migration/native-table-annotations.json')
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_PageSource,_AxisLine,_TableCandidate,_TableAnnotation
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import table_materialization as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_roles import metadata_field,text_role
    from rapid_doc._vendor.docvortex_core._compute_backend import get_native
    assert get_native()is None and unicodedata.unidata_version=='16.0.0'
    cases=[];front=[];roles=[]
    def ann(v):return _TableAnnotation(v['kind'],tuple(v['bbox']),set(v['line_indices']),{int(k):tuple(b)for k,b in v['line_bboxes'].items()})
    def source(v):
        lines=[]
        for q in copy.deepcopy(v['lines']):
            q['bbox']=tuple(q['bbox'])
            if q.get('font_signature')is not None:q['font_signature']=tuple(q['font_signature'])
            lines.append(_LineItem(**q))
        return _PageSource(tuple(v['page_size']),lines,copy.deepcopy(v['chars']),[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in v['drawing_lines']],page_index=v.get('page_index'))
    def candidate(v):
        v=copy.deepcopy(v)
        for k in ['bbox','local_bbox','core_bbox']:
            if v[k]is not None:v[k]=tuple(v[k])
        v['line_indices']=set(v['line_indices']);v['annotations']=[ann(a)for a in v['annotations']];v['inferred_grid']=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in v['inferred_grid']]
        return _TableCandidate(**v)
    def add(label,q):
        q=copy.deepcopy(q);q.pop('failed_annotations',None);s=source(q['source']);c=candidate(q['candidate'])
        repairs=old._overlapping_table_character_repairs(s)|({int(k):tuple(b)for k,b in q.get('calibrated_char_bboxes',{}).items()}if q.get('calibrated_char_bboxes')else{})
        native=[dict(char,bbox=repairs[char['char_idx']])if char.get('char_idx')in repairs else char for char in s.chars]
        old._separate_embedded_table_notes(s,c);blocks,external,failed=old._materialize_table_annotations(s,c)
        selected=old._candidate_projection_line_indices(s,c)
        for a in c.annotations:selected.update(a.line_indices)
        selected.difference_update(external)
        for b in blocks:b['_font_signatures']=sorted(b['_font_signatures'])
        expected=clean(dict(candidate=asdict(c),native_chars=native,calibrated_char_bboxes=repairs,projection_line_indices=sorted(selected),body_bbox=old._table_body_materialization_bbox(c,failed),annotation_blocks=blocks,externalized_line_indices=sorted(external),failed_annotations=[asdict(a)for a in failed]))
        cases.append(dict(label=label,request=q,expected=expected))
    base=json.loads((ROOT/'build/migration/native-table-block.json').read_text(encoding='utf-8'))
    for c in base['cases']:add(c['label'],c['request'])
    typed=json.loads((ROOT/'build/migration/native-table-block-types.json').read_text(encoding='utf-8'))
    for c in typed['cases']:add(c['label'],c['request'])
    rng=random.Random(52109);texts=['Note: first inter-','national result','中文连续内容。','下一行','https://example.com/path/','continued?q=yes',' A\u00a0 B\u200b ','word\x02\n','with\u00ad ','end\u00admiddle','a\u0085b','\x00\x0c\x1f','   ','ﬁ ﬂ 𝔸','Ä','\u2028\u2029','foo\x02x','last.']
    names=[None,['ABCDEF+Arial',0],['Arial',64],['Straße',0],['STRASSE',0],['Σ',0],['ς',64],['',0],['_ -',0],['Serif',0]]
    for n in range(280):
        a=(0,90,180,270)[n%4];ls=[]
        for i in range(rng.randrange(1,9)):
            x=20.+(i%3)*2.;y=20.+i*12.;w=[100.,35.,85.][i%3];h=rng.choice([8.,10.,12.]);b=(x,y,x+w,y+h)
            sig=names[(n+i)%len(names)];ink=_rotate_bbox_from_upright((x+1,y+1,x+w-1,y+h-1),(500.,700.),a)if n%3 else None
            ls.append(dict(text=texts[(n+i)%len(texts)],bbox=list(_rotate_bbox_from_upright(b,(500.,700.),a)),source_index=i*3+2,angle=a,font_signature=sig,font_coverage=rng.choice([0.49,.5,.59999,.6,.74999,.75,1.]),em_height=rng.choice([0.,8.,12.5]),effective_height=h,style_scale_repaired=bool(n%5==0),ink_bbox=ink))
        if n%2:ls.reverse()
        outer=list(_rotate_bbox_from_upright((0.,0.,200.,150.),(500.,700.),a))
        aa=[dict(kind='footnote',bbox=outer,line_indices=[l['source_index']for l in ls]+([9999]if n%4==0 else[]),line_bboxes={})]
        if n%3==0:aa.insert(0,dict(kind='caption',bbox=outer,line_indices=[ls[0]['source_index']],line_bboxes={str(ls[0]['source_index']):outer}))
        if n%7==0:aa.append(dict(kind='footnote',bbox=outer,line_indices=[123456],line_bboxes={}))
        q=copy.deepcopy(base['cases'][0]['request']);q['source'].update(lines=ls,chars=[],drawing_lines=[],page_size=[500.,700.]);q['candidate'].update(bbox=outer,local_bbox=outer,core_bbox=outer,angle=a,line_indices=[],annotations=aa)
        add(f'annotation:{n}',q)
    for n in range(24 if types_only else 0):
        a=(0,90,180,270)[n%4];p=(500,700)
        ls=[dict(text=t,bbox=list(_rotate_bbox_from_upright(b,p,a)),source_index=i,angle=a,effective_height=10 if n%3 else 0,em_height=12 if n%5 else 0,style_scale_repaired=bool(n%2),font_signature=['Arial',0],font_coverage=1,ink_bbox=list(_rotate_bbox_from_upright((2,2,7,8),p,a))if i%2 else None)for i,t,b in[(2,'integer line',(1,1,21,11)),(5,'next line',(1,13,70,23))]]
        q=copy.deepcopy(base['cases'][0]['request']);outer=list(_rotate_bbox_from_upright((0,0,100,100),p,a))
        q['source'].update(page_size=list(p),lines=ls,chars=[],drawing_lines=[])
        q['candidate'].update(bbox=outer,core_bbox=outer,local_bbox=outer,angle=a,line_indices=[],annotations=[dict(kind='caption',bbox=outer,line_indices=[2,5],line_bboxes={'2':outer})])
        add(f'types:{n}',q)
    def front_case(label,v,bbox):
        s=source(v);r=old._restore_front_matter_text_panel(s,tuple(bbox));out=copy.deepcopy(v['lines'])
        for raw,l in zip(out,s.lines):
            if l.semantic_type!=raw.get('semantic_type')or l.explicit_section_title!=raw.get('explicit_section_title',False):raw.update(semantic_type=l.semantic_type,explicit_section_title=l.explicit_section_title)
        front.append(dict(label=label,request=dict(source=v,bbox=bbox),expected=dict(restore=r,lines=out)))
    headers=['Article info','Ａｒｔｉｃｌｅ： ｉｎｆｏ','文章信息','Abstract','ＡＢＳＴＲＡＣＴ','内容摘要','Author affiliations','summary','Abstract body','Ａrticle information','𝔄𝔟𝔰𝔱𝔯𝔞𝔠𝔱','ＳＵＭＭＡＲＹ','ReCeİved:','receıved','key\u001cwords','foo@example.com','a@b.c','a@..xx','ＣＯＮＴＡＣＴ','correſponding','corresponding_','_correspondence','Correspondence:']
    for t in headers+['Received yesterday','Revised','accepted','Available\u001conline','Key words','key wordscape','通讯作者','x@b.İſ','sourcetest']:
        roles.append(dict(text=t,field=metadata_field(t),role=text_role(t)))
    for n in range(240):
        head=headers[n%12];fields=['Received 2026','Accepted 2026']
        if n%11==0:fields=['Received one','Received two']
        if n%13==0:fields=['key\u001cwords: science','foo@bar.example']
        if n%17==0:fields=['correſponding author','available\u00a0online now']
        ls=[dict(text=t,bbox=[0.,30.+i*15.,80.,40.+i*15.],source_index=i,angle=0)for i,t in enumerate(fields)]
        ls+=[dict(text=headers[n%12],bbox=[100.,5.,260.,15.],source_index=9,angle=0),dict(text=head,bbox=[0.,5.,80.,15.],source_index=8,angle=0)]
        for i in range([1,2,3,4][(n//12)%4]):
            text='This is a complete prose line with enough words for the side panel.'if n%2 else'这是用于检测连续正文布局的一个完整中文段落内容。'
            x=100.+(30. if n%19==0 and i%2 else 0.)
            ls.append(dict(text=text,bbox=[x,30.+i*15.,260.+(40. if n%23==0 and i%2 else 0.),40.+i*15.],source_index=20+i,angle=90 if n%29==0 else 0))
        rules=[]
        if n%7==0:rules=[dict(bbox=b,width=.2,orientation=o)for b,o in[([90.,20.,90.,120.],'vertical'),([0.,20.,280.,20.],'horizontal'),([0.,80.,280.,80.],'horizontal')]]
        v=dict(page_size=[500.,700.],lines=ls,chars=[],drawing_lines=rules,page_index=1 if n%5==0 else None if n%31==0 else 0)
        front_case(f'front:{n}',v,[0.,0.,300.,200.])
    prefix='rapid_doc/_vendor/docvortex_core/'
    paths=['analyzers/native/pdf/'+p for p in ['table_materialization.py','line_layout.py','line_merging.py','native_text.py','typography.py','text_roles.py','geometry.py']]+['foundation/_text.py']
    if types_only:cases=[c for c in cases if c['label'].startswith('types:')];front=[];roles=[]
    write_json(target,dict(sources={prefix+p:digest((ROOT/(prefix+p)).read_bytes())for p in paths},cases=clean(cases),front=clean(front),roles=roles,real=[]if types_only else base['real']))
    print('Frozen',len(cases),'annotation transactions;',sum(bool(c['expected']['annotation_blocks'])for c in cases),'with output;',len(front),'front panels;',sum(c['expected']['restore']for c in front),'restored;',len(roles),'role/field facts')
if __name__=='__main__':main()
