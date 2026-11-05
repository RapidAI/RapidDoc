"""Old continuous document source chain through vector formulas; immutable references."""
import copy,json,random,unicodedata
from common import ROOT,write_json,digest
from export_native_document_images import reference

def main():
    target=ROOT/'build/migration/native-vector-document.json';text_target=ROOT/'build/migration/native-vector-text.json'
    if target.exists()or text_target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    for name in ['native-document-images','native-vector-formulas']:
        fixtures=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in fixtures['cases']:
            # Original full document scenes already cover vector checkpoint continuations.
            if name=='native-vector-formulas'and c['label'].startswith('continuous/'):continue
            q=c['request']if name=='native-document-images'else dict(sources=[c['request']['source']]);item=dict(label=name+'/'+c['label'],request=copy.deepcopy(q))
            try:item['expected']=reference(q['sources'],vector=True)
            except ValueError as e:item.update(error='invalid_input',old_exception=type(e).__name__)
            cases.append(item)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import formulas as f,text_roles as t
    samples=['','(1)',' （A.2-3） ','(x)','(words)','((1))','prefix (1)','( \n )','x = y','where x = y','alpha + beta','Here are three words','© original','original article','JOuRnAL','research article extra words five','foo_journal','İSSN','reſearch article','1/2','⅔','\U0001d6c1']
    rng=random.Random(819);words=['alpha','x','β','你好','these','sin(x)','foo_3','two','journal','DOI','A12','á','²','\u2000','\x1c','⅜','\U0001e5f1','(','）','[x]']
    for _ in range(600):samples.append(''.join(rng.choice(words)+rng.choice([' ','=','+','/','_',',','-',''])for _ in range(rng.randint(1,6))))
    # All math symbols and vulgar fraction names, plus deterministic neighboring codepoints.
    for i in range(0x110000):
        c=chr(i)
        if unicodedata.category(c)=='Sm'or'VULGAR FRACTION'in unicodedata.name(c,''):samples.append(c)
    text_cases=[dict(text=s,number=f._FORMULA_NUMBER_MARKER_RE.fullmatch(s.strip())is not None,standalone=f._standalone_formula_number_marker(s)is not None,math=f._formula_line_has_math_operator(s),prose=t.has_prose(s),publication=t.publication_text(s))for s in samples]
    write_json(text_target,dict(cases=text_cases));write_json(target,dict(cases=cases,text_sha256=digest(text_target.read_bytes())));print(len(cases),'whole document chains;',sum('error'in c for c in cases),'old exceptions;',len(text_cases),'text references')
if __name__=='__main__':main()
