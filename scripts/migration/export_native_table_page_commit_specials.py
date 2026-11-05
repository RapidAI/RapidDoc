"""Additional raw-glyph four-row numeric tables and page-stage boundaries; primary corpus immutable."""
import copy,json
from common import ROOT,write_json
from export_native_table_page_commit import reference
from export_final_candidate_producers import base,glyphs
from export_supplemental_rule_tables import line
def numeric():
    q=base('aligned_numeric');s=q['source'];s['lines'] += [line('Alpha',[50.,175.,120.,185.],40)]+[line('13',[210.+60*j,175.,240.+60*j,185.],41+j)for j in range(3)];s['drawing_lines'][-1]['bbox'][1]=189.9;s['drawing_lines'][-1]['bbox'][3]=190.1;s['page_index']=1;glyphs(s);return dict(source=s)
def main():
    target=ROOT/'build/migration/native-table-page-commit-specials.json'
    if target.exists():raise FileExistsError(target)
    original=json.loads((ROOT/'build/migration/native-table-page-commit.json').read_text(encoding='utf-8'));cases=[];errors=[]
    def add(label,q):
        q=copy.deepcopy(q)
        try:out=reference(q)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=out))
    add('four-row-numeric',numeric());assert any(b['_native_numeric_grid']for b in cases[0]['expected']['table_blocks'])
    for t in ['10','+10','-10','−10','١٢','1,000','1.2','10%','10 %','(10)','10kg','12.','x']:
        q=numeric();s=q['source']
        for l in s['lines']:
            if l['text']in ['10','11','12','13']:l['text']=t
        glyphs(s);add('numeric-'+t,q)
    for mode in ['protected-body','protected-all','duplicate-source','prior-claimed','prior-footer','prior-code','separator-float-int','maps','source-markers','metadata','bad-origins','empty']:
        q=numeric();s=q['source']
        if mode=='protected-body':s['lines'][5]['semantic_type']='footer'
        if mode=='protected-all':
            for l in s['lines']:l['semantic_type']='header'
        if mode=='duplicate-source':s['lines'][-1]['source_index']=s['lines'][0]['source_index']
        if mode=='prior-claimed':q['prior_claimed_line_indices']=[s['lines'][0]['source_index']]
        if mode in ['prior-footer','prior-code']:q['prior_fixed_blocks']=[dict(type='footer'if mode=='prior-footer'else'code',bbox=[0.,200.,100.,210.],angle=0,content='prior')]
        if mode=='separator-float-int':q['table_header_separator_bboxes']=[[40,90,380,90]];s['drawing_lines'][0]['bbox']=[40.,90.,380.,90.]
        if mode=='maps':q['tight_bboxes']={str(i):c['bbox']for i,c in enumerate(s['chars'])};q['origins']={str(i):[c['bbox'][0],c['bbox'][3]]for i,c in enumerate(s['chars'])}
        if mode=='source-markers':s['lines'].append(line('Body',[40.,205.,240.,215.],100));glyphs(s)
        if mode=='metadata':s['custom_metadata']={'keep':[1,2,'x']}
        if mode=='bad-origins':q['origins']={str(i):['bad']for i in range(len(s['chars']))}
        if mode=='empty':s.update(lines=[],chars=[],drawing_lines=[],path_infos=[])
        add(mode,q)
    write_json(target,dict(sources=original['sources'],scope='True four-row/three-column numeric flags, protected/raw member views, actual char recovery, prior block/claim/separator/maps boundaries.',cases=cases,errors=errors));print('Special table page',len(cases),'errors',len(errors),'tables',sum(bool(c['expected']['table_blocks'])for c in cases),'numeric flags',sum(b['_native_numeric_grid']for c in cases for b in c['expected']['table_blocks']))
if __name__=='__main__':main()
