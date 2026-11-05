"""Original final PdfModel visible-text normalization, including HTML and span formula boundaries."""
import copy,json
from common import ROOT,write_json
def main():
    target=ROOT/'build/migration/native-model-normalization.json';config=ROOT/'rust/rapiddoc/src/processing/layout/native_model_normalization.json'
    if target.exists()or config.exists():raise FileExistsError('immutable normalization references exist')
    from rapid_doc._vendor.docvortex_core.content import normalization as old
    natural=sorted(k.value if hasattr(k,'value')else k for k in old._NATURAL_LANGUAGE_TYPES);cases=[]
    def add(label,blocks):
        q=copy.deepcopy([blocks]);out=copy.deepcopy(q);old.normalize_pdf_model_text(out);again=copy.deepcopy(out);old.normalize_pdf_model_text(again);assert again==out;cases.append(dict(label=label,request=q,expected=out))
    texts=['Ａｚ０９：．／＼－＿％＋＝＠＃＆＊，。（）','before Ａ \\(Ｂ１２\\) Ｃ','before Ａ \\[Ｂ１２\\] Ｃ','Ａ \\(Ｂ unclosed Ｃ','Ａ \\[Ｂ unclosed Ｃ','Ａ \\\\(Ｂ\\) Ｃ','no fullwidth text','😀ＡⅣﬁＡ\nＢ','Ａ\x00Ｂ']
    for kind in natural+['table','table_body','equation','code','image','unknown']:
        for text in texts:add(kind+'/'+repr(text),[dict(type=kind,content=text,bbox=[0.,0.,1.,1.],retained=3)])
    for text in texts:
        for split in [1,2,5,len(text)//2]:
            spans=[dict(type='text',content=text[:split],styles=['bold']),dict(type='hyperlink',target='https://example.test/Ａ１２：',content=[dict(type='text',content=text[split:])])];add('spans/'+repr((text,split)),[dict(type='text',content=spans)])
            spans.insert(1,dict(type='equation',content='opaque Ｆ'));add('span-barrier/'+repr((text,split)),[dict(type='text',content=spans)])
    for tag in ['b','span','a','eq','math','pre','code','script','style','svg','template','textarea','object','embed','canvas','iframe','ns:math']:
        for attrs in ['',' class="docvortex-math"',' data-docvortex-latex=""',' data-formula-display=""',' data-block-type="algorithm_body"']:
            markup=f'<TABLE Z="x" class="  keep  style "><tr><td>Ａ<{tag}{attrs} href="https://a.test/Ｂ">Ｃ</{tag}>Ｄ</td></tr></TABLE>';add('table/'+tag+'/'+attrs,[dict(type='table',content=markup)])
    for markup in ['<table><td>&#65313; &amp; &#xFF11;</td></table>','<table><td>\\<b>(Ａ</b><i>Ｂ\\</i>)Ｃ</td></table>','<table><td>\\<!-- comment -->(Ａ\\)Ｂ</td></table>','<table><td>\\<img src="Ａ"/>(Ａ\\)Ｂ</td></table>','<table><td>Ａ<table><tr><td>Ｂ</td></tr></table>Ｃ</td></table>','<table><td>Ａ<br>Ｂ<hr>Ｃ</td></table>','<table><td data-docvortex-latex="">Ａ</td><th>Ｂ</th></table>','<table><td>unchanged <b href="Ａ">text</b></td></table>','<table><td>Ａ</td><td>Ｂ</td></table>']:
        add('html/'+markup,[dict(type='table',content=markup)])
    write_json(target,dict(cases=cases));write_json(config,dict(natural=natural,opaque=sorted(old._OPAQUE_HTML_TAGS)));print(len(cases),'complete original model-text cases; idempotence verified')
if __name__=='__main__':main()
