"""Capture current Office OMML conversion, including original exceptions."""
from pathlib import Path
from xml.sax.saxutils import escape
import gzip
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
M='http://schemas.openxmlformats.org/officeDocument/2006/math'


def main():
    from lxml import etree
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.equation.omml import oMath2Latex
    from loguru import logger
    logger.remove()
    cases=[]
    def capture(xml,name):
        case=dict(name=name,xml=xml)
        try:
            converted=oMath2Latex(etree.fromstring(xml.encode()))
            case.update(raw=converted.latex,latex=str(converted))
        except Exception as error:case['error']=type(error).__name__
        cases.append(case)
    def r(text,prop=''):return f'<m:r><m:rPr>{prop}</m:rPr><m:t>{escape(text)}</m:t></m:r>'
    def e(text):return '<m:e>'+r(text)+'</m:e>'
    def wrap(content):return f'<m:oMath xmlns:m="{M}">{content}</m:oMath>'
    generated=[r(text) for text in ['x+y','{x}_1^2#&$%','\\{x}\\\\foo','  a   b    c','αβ→x','中文🌟','·⋅×÷−','ÅÅΩΩ','éé','℃½⅓ℝ𝔸','∯∰∱∲∳', 'a b']]
    for scr in ['script','fraktur','double-struck','sans-serif','monospace','roman','']:
        generated += [r(' ABC ',f'<m:scr m:val="{scr}"/>'),r('   ',f'<m:scr m:val="{scr}"/>')]
    for typ in [None,'bar','skw','noBar','lin','unknown','{bad}','{']:
        prop='' if typ is None else f'<m:fPr><m:type m:val="{typ}"/><m:brk/></m:fPr>'
        generated += [f'<m:f>{prop}<m:num>{r("a")}</m:num><m:den>{r("b")}</m:den></m:f>',f'<m:f>{prop}</m:f>']
    for tag,prop,values in [('acc','chr',[None,'̂','⃗','unknown','{bad}']),('bar','pos',[None,'top','bot','unknown']),('groupChr','chr',[None,'⏟','⏞','unknown'])]:
        for val in values:
            p='' if val is None else f'<m:{prop} m:val="{val}"/>'
            generated += [f'<m:{tag}><m:{tag}Pr>{p}<m:brk/></m:{tag}Pr>{e("x")}</m:{tag}>',f'<m:{tag}>{e("x")}</m:{tag}>']
    for left,right in [('(',')'),('',''),('∥','‖'),('{','}'),('[',']'),('→','←')]:
        generated.append(f'<m:d><m:dPr><m:begChr m:val="{left}"/><m:endChr m:val="{right}"/></m:dPr>{e("x")}{e("last")}</m:d>')
    for name in ['sin','cos','lim','foo','sinh','max']:
        generated.append(f'<m:func><m:fName>{r(name)}</m:fName>{e("x")}</m:func>')
    for degree in ['', '2','n']:
        generated.append(f'<m:rad><m:deg>{r(degree)}</m:deg>{e("x")}</m:rad>')
    for rows in [[],['x'],['x#(1)'],['x# (α)  '],['x#(1) y'],['x','y']]:
        generated.append('<m:eqArr>'+''.join(e(t) for t in rows)+'</m:eqArr>')
    for upper in [False,True]:
        tag='limUpp' if upper else 'limLow'
        for base in ['lim','max','min','f']:
            for limit in ['n→∞','⏟','⏞','0']:
                generated.append(f'<m:{tag}>{e(base)}<m:lim>{r(limit)}</m:lim></m:{tag}>')
    for val in [None,'∑','∫','⨀','raw']:
        prop='' if val is None else f'<m:chr m:val="{val}"/>'
        generated.append(f'<m:nary><m:naryPr>{prop}</m:naryPr><m:sub>{r("i")}</m:sub><m:sup>{r("n")}</m:sup>{e("x")}</m:nary>')
    generated += ['<m:m><m:mPr/><m:mr>'+e('a')+e('b')+'</m:mr><m:mr>'+e('c')+e('d')+'</m:mr></m:m>',
        '<m:sSubSup>'+e('x')+'<m:sub>'+r('i')+'</m:sub><m:sup>'+r('2')+'</m:sup></m:sSubSup>',
        '<m:unknown>'+r('ignored')+'</m:unknown><m:box>'+r('kept')+'</m:box>',
        '<m:limLow>'+e('lim')+'<m:acc/></m:limLow>',
        '<m:func/>','<m:func><m:fName>'+r('foo')+'</m:fName></m:func>',
        '<m:rad/>','<m:d/>','<m:nary><m:otherPr/></m:nary>',
        '<m:rPr><m:brk/><m:chr m:val="x"/></m:rPr>',
        '<m:groupChr><m:groupChrPr><m:chr m:val="⏟"/></m:groupChrPr>'+e('x')+'</m:groupChr>',
    ]
    # Exercise restoration of command braces after the original escape pass.
    generated += [r('prefix '+char+'_{suffix}') for char in ['é','ℝ','Å','α','℃','𝔸','\\','{','}']]
    for index,body in enumerate(generated):capture(wrap(body),f'generated {index}')
    manifest=json.loads((ROOT/'tests/fixtures/office_rust_reference_068/manifest.json').read_text(encoding='utf-8'))
    actual=0
    for case in manifest['cases']:
        if case['format'] not in {'docx','xlsx','pptx'}:continue
        with zipfile.ZipFile(ROOT/case['input']) as package:
            for part in package.namelist():
                if not part.endswith('.xml'):continue
                payload=package.read(part)
                if b'oMath' not in payload:continue
                tree=etree.fromstring(payload)
                for index,node in enumerate(tree.iter(f'{{{M}}}oMath')):
                    capture(etree.tostring(node,encoding='unicode'),case['id']+':'+part+f':{index}')
                    actual+=1
    mapping=json.loads((ROOT/'rust/rapiddoc-office/src/equation/omml_data.json').read_text(encoding='utf-8'))
    converter=oMath2Latex.__new__(oMath2Latex)
    unicode=[dict(char=char,latex=converter.process_unicode(char)) for char in mapping['unicode']]
    fixture=dict(cases=cases,unicode=unicode,real_equations=actual)
    target=ROOT/'tests/fixtures/office_rust_reference_068/omml.json.gz'
    target.write_bytes(gzip.compress(json.dumps(fixture,ensure_ascii=False).encode(),mtime=0))
    print('OMML references:',len(cases),'equations;',actual,'real;',sum('error' in c for c in cases),'expected errors;',len(unicode),'scalar mappings')


if __name__=='__main__':main()
