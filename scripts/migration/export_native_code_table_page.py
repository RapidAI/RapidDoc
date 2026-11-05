"""Old rule/algorithm code -> full table commit -> filled code order, no injected candidates."""
import copy,json
from common import ROOT,write_json
from export_native_table_page_commit import reference
from export_native_code_containers import base,glyphs
def duplicate(q):
    ids=[l['source_index']for l in q['source']['lines']];return len(ids)!=len(set(ids))
def main():
    target=ROOT/'build/migration/native-code-table-page.json';pdf_target=ROOT/'build/migration/pdf-code-table-page.json'
    if target.exists()or pdf_target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[];errors=[]
    def add(label,q):
        q=copy.deepcopy(q)
        try:expected=reference(q,True)
        except(ValueError,KeyError)as e:errors.append(dict(label=label,request=q,error='invalid_input',old_exception=type(e).__name__));return
        cases.append(dict(label=label,request=q,expected=expected))
    for kind in ['filled','ruled','algorithm']:add(kind,base(kind))
    raw=json.loads((ROOT/'build/migration/native-code-containers.json').read_text(encoding='utf-8'))
    for c in raw['cases'][::7]:add('code/'+c['label'],c['request'])
    tables=json.loads((ROOT/'build/migration/native-table-page-commit-specials.json').read_text(encoding='utf-8'))
    for c in tables['cases']:
        if not duplicate(c['request']):add('table/'+c['label'],c['request'])
    for code_kind in ['filled','ruled','algorithm']:
        q=copy.deepcopy(tables['cases'][0]['request']);s=q['source']
        for l in s['lines']:l['bbox'][1]+=220.;l['bbox'][3]+=220.;l['source_index']+=1000
        for r in s['drawing_lines']:r['bbox'][1]+=220.;r['bbox'][3]+=220.
        code=base(code_kind)['source'];s['lines']+=code['lines'];s['drawing_lines']+=code['drawing_lines'];s['path_infos']+=code['path_infos'];glyphs(s);s['page_index']=1;add('code-and-table/'+code_kind,q)
    refs=json.loads((ROOT/'build/migration/native-table-page-commit.json').read_text(encoding='utf-8'))['sources'];refs.update(raw['sources']);write_json(target,dict(sources=refs,scope='Actual old continuous rule/algorithm code -> table detection/commit -> filled code sequence with real shared source row side effects.',cases=cases,errors=errors))
    pdf=[]
    for name in ['pdf-code-containers','pdf-table-page-commit']:
        previous=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'));seen=set()
        for c in previous['cases']:
            key=(c['pdf'],c['page_index'])
            if key in seen:continue
            seen.add(key);q=copy.deepcopy(c['request']);q.pop('kind',None);q.setdefault('tight_bboxes',{str(ch['char_idx']):ch.get('tight_bbox')for ch in q['source']['chars']});q.setdefault('origins',{str(ch['char_idx']):ch.get('origin')for ch in q['source']['chars']})
            pdf.append(dict(pdf=c['pdf'],pdf_sha256=c['pdf_sha256'],page_index=c['page_index'],label=c['label'],request=q,expected=reference(q,True)))
    write_json(pdf_target,dict(sources=refs,scope='67 distinct actual PDF pages through old code/table preparation order, keeping source evidence and outcomes.',cases=pdf));print('Frozen code/table stages',len(cases),'errors',len(errors),'code/table mixtures',[(c['label'],len(c['expected']['rule_code_blocks']),len(c['expected']['table_blocks']),len(c['expected']['code_blocks']))for c in cases if c['label'].startswith('code-and-table')]);print('PDF',len(pdf),'with code',sum(bool(c['expected']['rule_code_blocks']or c['expected']['code_blocks'])for c in pdf),'with tables',sum(bool(c['expected']['table_blocks'])for c in pdf))
if __name__=='__main__':main()
