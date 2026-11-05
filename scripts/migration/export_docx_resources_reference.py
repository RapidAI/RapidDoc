"""Freeze Mammoth style-map grammar and full DocVortex DOCX resource outputs."""
import gzip,json,sys
from pathlib import Path
from io import BytesIO
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def main():
    import mammoth
    from mammoth.options import _read_style_map
    from mammoth import html,writers
    from mammoth.conversion import _document_matcher_matches
    from types import SimpleNamespace
    from docx import Document
    from docx.oxml import parse_xml
    from docx.opc.part import Part
    from docx.opc.packuri import PackURI
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from rapid_doc._vendor.docvortex_core.analyzers.native.office.docx.docx_converter import DocxConverter
    from loguru import logger
    logger.remove()
    maps=["p => h2:fresh","p => p:separator(' | ')","p => block > p:fresh","p => h1|h2.box.a[data-key='A & B']:fresh","p => !","p =>", "p.foo => h3", "p[style-name='custom'] => h4", "p[style-name^='Cus'] => h4", "p:ordered-list(1) => ol > li:fresh", "p:unordered-list(0) => div", "r => span.kind", "r.Strong => i", "r[style-name='Strong'] => i", "b => em", "i => strong", "u => u", "strike => del", "all-caps => strong", "small-caps => small", "highlight => mark", "highlight[color='yellow'] => mark.yellow", "br[type='line'] => hr", "br[type='page'] => br", "br[type='column'] => hr", "table => section > table:fresh", "table.TableGrid => table.kind", "comment-reference => sup", "p => p:separator('\\n')", "p => p[class='x'].y[class='z'].a", "p => p.one\\.two", "p => p:separator('\\\\\\\'')", "p => a[href='javascript:alert(1)']", "p => p:fresh:separator('x')", "# ignored\np => div", "  p => h2  ", "p=>h1", "p =>h1", "p => h1>h2", "p => h1 >h2", "p => h1 > h2", "p => p:separator('x'):fresh", "p[style-name=\"Custom\"] => p", "p => p[missing]", "p => p[role='a'][role='b']", "br => br", "unknown => p", "p:ordered-list(9999999999999999999999999999999999) => div", "p[style-name='Straße'] => p", "p.foo\\ bar => p", "p => p.a\\ b", "p => p[data-k='a\\tb']", "p => p[emoji='😀']", "p => ! trailing", "p => p[]", "p => p:separator('')"]
    queries=[dict(kind=kind,id=ident,name=name,numbering=num,qualifier=q) for kind,ident,name,num,q in [('p',None,None,None,None),('p','foo','Custom',None,None),('p',None,'CUSTOM',None,None),('p',None,'StraSSe',None,None),('p',None,'Other',['0',True],None),('p',None,'Other',['-1',False],None),('r','Strong','Strong',None,None),('table','TableGrid','Table Grid',None,None),('b',None,None,None,None),('i',None,None,None,None),('u',None,None,None,None),('strike',None,None,None,None),('all-caps',None,None,None,None),('small-caps',None,None,None,None),('highlight',None,None,None,'yellow'),('highlight',None,None,None,'red'),('br',None,None,None,'line'),('br',None,None,None,'page'),('br',None,None,None,'column'),('comment-reference',None,None,None,None)]]
    grammar=[]
    kinds={'p':'paragraph','r':'run','b':'bold','i':'italic','u':'underline','strike':'strikethrough','all-caps':'all_caps','small-caps':'small_caps','br':'break','comment-reference':'comment_reference'}
    from mammoth.documents import numbering_level
    for text in maps:
        result=_read_style_map(text);outputs=[]
        for q in queries:
            e=SimpleNamespace(style_id=q['id'],style_name=q['name'],numbering=numbering_level(int(q['numbering'][0]),q['numbering'][1]) if q['numbering'] else None,color=q['qualifier'],break_type=q['qualifier'])
            match=next((rule for rule in result.value if _document_matcher_matches(rule.document_matcher,e,kinds.get(q['kind'],q['kind']))),None)
            if match is None:outputs.append(None);continue
            nodes=match.html_path.wrap(lambda:[html.text('A & <B>')])+match.html_path.wrap(lambda:[html.text('C')]);writer=writers.writer('html');html.write(writer,html.collapse(html.strip_empty(nodes)));outputs.append(writer.as_string())
        grammar.append(dict(text=text,warnings=[m.message for m in result.messages],outputs=outputs))
    cases=[]
    def capture(name,doc,style_map=''):
        out=BytesIO();doc.save(out)
        if style_map:mammoth.embed_style_map(out,style_map)
        raw=out.getvalue();case=dict(name=name,package_hex=raw.hex());engine=DocxConverter()
        try:engine.convert(BytesIO(raw));case['pages']=engine.pages
        except Exception as e:case['error']=type(e).__name__
        cases.append(case)
    def xml(s):return parse_xml(('<root xmlns:w="'+W+'">'+s+'</root>').encode())[0]
    for i,text in enumerate(maps):
        d=Document();d.add_paragraph('outside table remains')
        for k in range(2):
            t=d.add_table(1,1);t.style='Table Grid';p=t.cell(0,0).paragraphs[0];p.style='Heading 2'
            for word in ['first','second']:
                r=p.add_run(word);r.bold=True;r.italic=True;r.underline=True;r.font.strike=True;r.font.all_caps=True;r.font.small_caps=True
                r._r.get_or_add_rPr().append(xml('<w:highlight w:val="yellow"/>'));r.add_break()
            t.cell(0,0).add_paragraph('other paragraph')
        capture('embedded map '+str(i),d,text)
    for mode in ['mixed','missing','no parts','separators','repeated','nested note','note ref in note','unused bad note','comments','missing comments','hidden note','note table boundary']:
        d=Document();d.add_paragraph('body before')
        for index in range(2):
            t=d.add_table(1,1);p=t.cell(0,0).paragraphs[0];p.add_run('table '+str(index))
            for kind in ['footnote','endnote','footnote']:
                r=p.add_run();r._r.append(xml('<w:'+kind+'Reference w:id="'+('2' if mode=='missing' and index else '1')+'"/>'))
            if mode in ['comments','missing comments']:p.add_run()._r.append(xml('<w:commentReference w:id="3"/>'))
        if mode!='no parts':
            for kind,rel in [('footnote',RT.FOOTNOTES),('endnote',RT.ENDNOTES)]:
                typ=' w:type="separator"' if mode=='separators' else ''
                body='<w:p><w:r><w:t>note body</w:t></w:r></w:p>'
                if mode=='note table boundary':body='<w:tbl><w:tblPr><w:tblStyle w:val="NoteTable"/></w:tblPr><w:tr><w:tc><w:p><w:r><w:t>note table</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                if mode=='nested note':body+='<w:tbl><w:tr><w:tc><w:p><w:r><w:t>note table</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
                if mode=='note ref in note':body+='<w:p><w:r><w:footnoteReference w:id="999"/></w:r></w:p>'
                extra='<w:'+kind+' w:id="8"><w:p><w:r><w:sym w:char="not hex"/></w:r></w:p></w:'+kind+'>' if mode=='unused bad note' else ''
                raw=('<w:'+kind+'s xmlns:w="'+W+'"><w:'+kind+' w:id="1"'+typ+'>'+body+'</w:'+kind+'>'+extra+'</w:'+kind+'s>').encode()
                part=Part(PackURI('/word/'+kind+'s.xml'),'application/vnd.openxmlformats-officedocument.wordprocessingml.'+kind+'s+xml',raw,d.part.package);d.part.relate_to(part,rel)
        if mode=='comments':
            raw=('<w:comments xmlns:w="'+W+'"><w:comment w:id="3" w:initials=" AB "><w:p><w:r><w:t>comment body</w:t></w:r></w:p></w:comment></w:comments>').encode();part=Part(PackURI('/word/comments.xml'),'application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml',raw,d.part.package);d.part.relate_to(part,RT.COMMENTS)
        capture('notes '+mode,d,"comment-reference => sup" if 'comments' in mode else 'r => !' if mode=='hidden note' else 'table.NoteTable => table' if mode=='note table boundary' else '')
    for mapping in ["p.Heading1 => section > h3:fresh\np => div", "r[style-name='Strong'] => span.strong\nb => !", "p:ordered-list(1) => dl > dt:fresh\np:unordered-list(2) => ul > li:fresh", "u => u\nhighlight => mark", "p => p:separator(' / ')\nr => span:separator(' + ')","table => table:separator('x')", "table => !", "p => p[data-k='x']:fresh\np => !"]:
        d=Document();n=d.part.numbering_part.element;n.append(xml('<w:abstractNum w:abstractNumId="99"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>'));n.append(xml('<w:num w:numId="99"><w:abstractNumId w:val="99"/></w:num>'))
        for index in range(2):
            t=d.add_table(1,1);cell=t.cell(0,0)
            for level in [0,1,0]:
                p=cell.add_paragraph();p._p.get_or_add_pPr().append(xml('<w:numPr><w:ilvl w:val="'+str(level)+'"/><w:numId w:val="99"/></w:numPr>'))
                for text in ['A','B']:p.add_run(text,style='Strong')
            p=cell.add_paragraph('heading',style='Heading 1');p.add_run('bold').bold=True
            p=cell.add_paragraph();r=p.add_run('underline without val');r._r.get_or_add_rPr().append(xml('<w:u/>'));r=p.add_run('blank highlight');r._r.get_or_add_rPr().append(xml('<w:highlight w:val=""/>'))
        capture('mapped lists and precedence '+mapping,d,mapping)
    for mapping in ['r => !','p => !','table => !']:
        d=Document();t=d.add_table(1,1);p=t.cell(0,0).paragraphs[0];p.add_run('hidden external image');rid=d.part.relate_to('file:///definitely-not-present.png',RT.IMAGE,is_external=True)
        drawing='<w:r xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:drawing><wp:inline><wp:docPr id="1" name="image"/><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="'+rid+'"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        p._p.append(xml(drawing));d.add_paragraph('after hidden resource');capture('hidden image '+mapping,d,mapping)
    # A note and the main document reuse an rId for different image parts.
    # Table images retain original bytes, so HTML can be compared exactly.
    from PIL import Image
    def png(color):
        b=BytesIO();Image.new('RGB',(2,2),color).save(b,format='PNG');return b.getvalue()
    def image_xml(rid):
        return '<w:r xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:drawing><wp:inline><wp:docPr id="1" name="image"/><a:graphic><a:graphicData><pic:pic><pic:blipFill><a:blip r:embed="'+rid+'"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
    for broken in [False,True]:
        d=Document();t=d.add_table(1,1);p=t.cell(0,0).paragraphs[0];p.add_run('reference');p.add_run()._r.append(xml('<w:footnoteReference w:id="1"/>'))
        body_image=Part(PackURI('/word/media/main.png'),'image/png',png('red'),d.part.package);rid=d.part.relate_to(body_image,RT.IMAGE)
        raw=('<w:footnotes xmlns:w="'+W+'"><w:footnote w:id="1"><w:p>'+image_xml(rid)+'</w:p></w:footnote></w:footnotes>').encode();notes=Part(PackURI('/word/footnotes.xml'),'application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml',raw,d.part.package);d.part.relate_to(notes,RT.FOOTNOTES)
        note_image=Part(PackURI('/word/media/note.png'),'image/png',b'bad image' if broken else png('blue'),d.part.package);notes.rels.add_relationship(RT.IMAGE,note_image,rid)
        t=d.add_table(1,1);p=t.cell(0,0).paragraphs[0];p.add_run('main image');p._p.append(xml(image_xml(rid)));d.add_paragraph('body after note resources')
        capture('note image relationships '+str(broken),d)
    target=ROOT/'tests/fixtures/office_rust_reference_068/docx_resources.json.gz';target.write_bytes(gzip.compress(json.dumps(dict(grammar=grammar,queries=queries,cases=cases),ensure_ascii=True).encode(),mtime=0));print('DOCX resources:',len(grammar),'maps x',len(queries),'queries;',len(cases),'full documents')
if __name__=='__main__':main()
