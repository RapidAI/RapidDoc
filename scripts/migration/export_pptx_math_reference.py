"""Freeze the installed pypptx-with-oxml secondary formula engine and its data."""
import ast
import contextlib
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    import pptx.oxml
    import pptx.oxml.text as source
    from pptx.oxml import parse_xml
    text_path=Path(source.__file__)
    registry=ast.parse(Path(pptx.oxml.__file__).read_text(encoding='utf-8'))
    kinds={}
    for n in ast.walk(registry):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='register_element_cls' and len(n.args)==2 and isinstance(n.args[0],ast.Constant) and isinstance(n.args[1],ast.Name):
            cls=getattr(source,n.args[1].id,None)
            if cls is not None and hasattr(cls,'to_latex'):kinds[n.args[0].value]=n.args[1].id
    alphabetic=[]
    for code in range(0x110000):
        if chr(code).isalpha():
            if alphabetic and alphabetic[-1][1]+1==code:alphabetic[-1][1]=code
            else:alphabetic.append([code,code])
    data=dict(characters=source.MATH_CHAR_TO_LATEX_CONVERSION,kinds=kinds,alphabetic=alphabetic,source_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest())
    (ROOT/'rust/rapiddoc-office/src/pptx/math_fallback_data.json').write_text(json.dumps(data,ensure_ascii=True,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    M='http://schemas.openxmlformats.org/officeDocument/2006/math'
    A14='http://schemas.microsoft.com/office/drawing/2010/main'
    cases=[]
    def add(name,body):
        xml=f'<a14:m xmlns:a14="{A14}" xmlns:m="{M}">{body}</a14:m>'
        c=dict(name=name,xml=xml)
        try:
            with contextlib.redirect_stdout(io.StringIO()):c['latex']=parse_xml(xml.encode()).to_latex()
        except Exception as e:c['error']=type(e).__name__
        cases.append(c)
    def run(s):
        from html import escape
        return '<m:r><m:t>'+escape(s)+'</m:t></m:r>'
    def arg(tag,s):return '<m:'+tag+'>'+run(s)+'</m:'+tag+'>'
    for char in data['characters']:add('character/'+str(ord(char)), '<m:oMath>'+run(char)+'</m:oMath>')
    for s in ['','x','xy','1','1+2','θ','sin','custom','\\alpha ','汉字','a_b','{x}']:
        r=run(s);e='<m:e>'+r+'</m:e>'
        for tag in ['sSub','sSup','sSubSup','sPre']:
            add(tag+'/'+s,'<m:oMath><m:'+tag+'>'+e+arg('sub',s)+arg('sup',s)+'</m:'+tag+'></m:oMath>')
        add('fraction/'+s,'<m:oMath><m:f>'+arg('num',s)+arg('den','2')+'</m:f></m:oMath>')
        for hide in [None,'0','1','false','true','off','on','']:
            pr=''if hide is None else '<m:radPr><m:degHide m:val="'+hide+'"/></m:radPr>'
            add('rad/'+s+'/'+str(hide),'<m:oMath><m:rad>'+pr+arg('deg',s)+e+'</m:rad></m:oMath>')
        for fn in ['sin','Pr','custom','x','lim','α','∑','\u0345\u0345','\u05b0\u05b1']:
            add('func/'+fn+'/'+s,'<m:oMath><m:func>'+arg('fName',fn)+e+'</m:func></m:oMath>')
        for pos in [None,'top','bot','invalid']:
            pr=''if pos is None else '<m:barPr><m:pos m:val="'+pos+'"/></m:barPr>'
            add('bar/'+s+'/'+str(pos),'<m:oMath><m:bar>'+pr+e+'</m:bar></m:oMath>')
    for op in ['', '∫','∬','∭','∮','∯','∰','∑','∏','∐','⋃','⋂','⋀','⋁','⨆','⨄','⨁','⨂','⨀','⨌','xyz','★']:
        for s in ['x','f(x)dx','dx','{x}','x_i','']:
            for hide in ['', '<m:subHide m:val="true"/>','<m:supHide m:val="1"/>']:
                add('nary/'+op+'/'+s+'/'+hide,'<m:oMath><m:nary><m:naryPr><m:chr m:val="'+op+'"/>'+hide+'</m:naryPr>'+arg('sub','i=1')+arg('sup','n')+arg('e',s)+'</m:nary></m:oMath>')
    for begin,end in [('(',')'),('[',']'),('{','}'),('|','|'),('‖','‖'),('⟨','⟩'),('⌈','⌉'),('⌊','⌋'),('',''),('x','y'),('↑','↓')]:
        for sep in ['', '+','|','±']:
            for matrix in [False,True]:
                inside='<m:m><m:mr>'+arg('e','a')+arg('e','b')+'</m:mr><m:mr>'+arg('e','c')+arg('e','d')+'</m:mr></m:m>' if matrix else run('x')
                add('delimiter/'+begin+'/'+end+'/'+sep+'/'+str(matrix),'<m:oMath><m:d><m:dPr><m:begChr m:val="'+begin+'"/><m:endChr m:val="'+end+'"/><m:sepChr m:val="'+sep+'"/></m:dPr><m:e>'+inside+'</m:e></m:d></m:oMath>')
    for char in ['', '→','←','̅','_','̂','^','̃','~','̇','̈','́','̀','̌','̆','∑','x']:
        for tag in ['groupChr','acc']:
            for pos in [None,'top','bot']:
                pr=f'<m:{tag}Pr><m:chr m:val="{char}"/>'+(''if pos is None else f'<m:pos m:val="{pos}"/>')+f'</m:{tag}Pr>'
                add('accent/'+tag+'/'+char+'/'+str(pos),'<m:oMath><m:'+tag+'>'+pr+arg('e','xy')+'</m:'+tag+'></m:oMath>')
    for body in ['', '<m:oMath/>','<m:oMath><m:f/></m:oMath>','<m:oMath><m:d/></m:oMath>','<m:oMath><m:r/></m:oMath>','<m:oMath><m:eqArr>'+arg('e','a')+arg('e','b')+'</m:eqArr></m:oMath>','<m:oMathPara><m:oMath>'+run('a')+'</m:oMath><m:oMath>'+run('b')+'</m:oMath></m:oMathPara>']:
        add('boundary/'+str(len(cases)),body)
    for hide in [None,'0','1','true','false','on','off','TRUE','bad','']:
        flag=''if hide is None else '<m:radPr><m:degHide m:val="'+hide+'"/></m:radPr>'
        add('nested/degree/'+str(hide),'<m:oMath><m:rad>'+flag+'<m:deg>'+arg('e','3')+'</m:deg>'+arg('e','x')+'</m:rad></m:oMath>')
    for body in ['<m:f><m:num>'+arg('e','x')+'<m:r/></m:num>'+arg('den','2')+'</m:f>',
                 '<m:nary><m:naryPr><m:subHide m:val="true"/></m:naryPr><m:sub><m:r/></m:sub>'+arg('e','x')+'</m:nary>',
                 '<m:rPr><m:f/></m:rPr>'+run('x')]:
        add('lazy/'+str(len(cases)),'<m:oMath>'+body+'</m:oMath>')
    target=ROOT/'tests/fixtures/office_rust_reference_068/pptx_math.json.gz'
    target.write_bytes(gzip.compress(json.dumps(dict(cases=cases),ensure_ascii=True).encode(),mtime=0))
    print(len(data['characters']),'fork characters;',len(kinds),'element mappings;',len(cases),'secondary formula queries')

if __name__=='__main__':main()
