"""Freeze actual old links/styles -> complete InlineSpan blocks, no mapping substitutions."""
import copy,json,html,html.entities,random
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-inline-output.json';config=ROOT/'rust/rapiddoc/src/processing/document/inline_output/config.json'
    if target.exists()or config.exists():raise FileExistsError('immutable inline output reference exists')
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline import materialize as old,types as t
    from rapid_doc._vendor.docvortex_core.schema import INLINE_STYLE_ORDER
    from loguru import logger
    logger.remove();cases=[]
    def add(label,blocks,styles=(),links=(),size=(400.,600.),chain=None):
        q=clean(dict(blocks=blocks,style_lines=styles,link_lines=links,page_size=size));out=copy.deepcopy(blocks)
        ss=[t.PDFTextStyleLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextStyleRange(r['start'],r['end'],tuple(r['styles']))for r in l['style_ranges']),l['source_index'])for l in styles]
        ll=[t.PDFTextLinkLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextLinkRange(**r)for r in l['link_ranges']),l['source_index'])for l in links]
        old.apply_pdf_text_links(out,ll,size);old.apply_pdf_text_styles(out,ss,size);old.materialize_pdf_inline_spans(out)
        c=dict(label=label,request=q,expected=clean(out))
        if chain is not None:c['inline_case']=chain
        cases.append(c)
    def block(content,kind='text',bbox=(0.,0.,1.,1.)):return dict(type=kind,content=content,bbox=bbox,source='preserved')
    def style(text,spans=None,source=0,bbox=(10.,20.,200.,30.)):
        return dict(text=text,source_index=source,bbox=bbox,style_ranges=spans if spans is not None else[dict(start=0,end=len(text),styles=['bold'])])
    def link(text,spans=None,source=0,bbox=(10.,20.,200.,30.),url='https://example.test/a'):
        return dict(text=text,source_index=source,bbox=bbox,link_ranges=spans if spans is not None else[dict(start=0,end=len(text),target=url)])
    contents=['hello world','a ﬁ b','😀中文 text','a\u00a0b\u200bc','left \\(x+y\\) right','x <b>raw</b> text',' <sup>2</sup> x','a\\( \\)b','a\\(unterminated','first\nsecond','']
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.common import _normalize_match_fragment
    for content in contents:
        text=_normalize_match_fragment(content)
        for kind in ['text','index','paragraph_title','doc_title','caption','table','image','ref_text']:
            add('content-'+repr(content)+'-'+kind,[block(content,kind)],[style(text)],[link(text)])
    for styles in [['bold'],['italic'],['underline'],['strikethrough'],['superscript'],['subscript'],['bold','underline'],['bold','strikethrough']]:
        add('styles-'+str(styles),[block('  ab cd  ')],[style('abcd',[dict(start=0,end=4,styles=styles)])])
    for content,line_text in [('left \\(formula\\) right','leftFORMULAright'),('one \\(x\\) two \\(y\\) end','oneXtwoYend'),('abc xyz abc xyz','missingabc'),('abc abc','abc'),('xxabcxx','badabc'),('left target right','badtargetright'),('ab','badab'),('xy','badxy')]:
        add('fallback-'+content,[block(content)],[style(line_text)],[link(line_text)])
        for spanstyles in [['bold'],['underline']]:
            add('fallback-part-'+content+str(spanstyles),[block(content)],[style(line_text,[dict(start=max(0,len(line_text)-3),end=len(line_text),styles=spanstyles)])],[link(line_text,[dict(start=max(0,len(line_text)-3),end=len(line_text),target='https://a.test')])])
    for gap in ['',' ','\n',' - ',' &amp; ',' &#38; ',' &#x61; ',' &copy; ','BODY',' \\(x\\) ']:
        content='first'+gap+'second';ss=[style('first',source=7),style('second',source=8)];ll=[link('first',source=7),link('second',source=8)]
        add('across-'+repr(gap),[block(content)],ss,ll)
    for content in ['international','inter-national','inter national','international international']:
        for same in [True,False]:
            add('dehyphen-'+content+str(same),[block(content)],[style('inter-',source=2),style('national',source=3)],[link('inter-',source=2),link('national',source=3,url='https://example.test/a'if same else'https://other.test')])
    for bbox in [(0.,0.,1.,1.),(0.,0.,400.,600.),(10.,20.,200.,30.),(11.,20.,50.,30.),(250.,250.,300.,300.)]:
        add('assignment-'+str(bbox),[block('same',bbox=bbox),block('same',bbox=(0.,0.,400.,600.))],[style('same')],[link('same')])
    for content in ['x<sup>2</sup>y','x<sub>2</sub>y','<sup><sub>x</sub></sup>','<sup>x</sub>','unclosed<sup>x','a<sup></sup>b','a<sup> x </sup>b']:
        b=block(content);b['_pdf_native_script_markup']=True;add('owned-'+content,[b])
    rng=random.Random(27002)
    for i in range(90):
        content=rng.choice(['abcdef abcdef','ab cd ef','a ﬁ b 😀','left \\(z\\) right']);text=_normalize_match_fragment(content)
        ranges=[]
        for _ in range(rng.randint(1,4)):
            start=rng.randrange(max(1,len(text)));end=rng.randint(start,len(text));ranges.append(dict(start=start,end=end,styles=rng.choice([['bold'],['underline'],['bold','strikethrough']])))
        add('random-'+str(i),[block(content)],[style(text,ranges)],[link(text,[dict(start=0,end=min(4,len(text)),target='https://a.test'),dict(start=2,end=len(text),target='https://b.test')])])
    inline=json.loads((ROOT/'build/migration/native-inline-evidence.json').read_text(encoding='utf-8'));source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    for i,c in enumerate(inline['cases']):
        if'source_case'not in c:continue
        sc=source['cases'][c['source_case']];blocks=[block(l['text'],bbox=l['bbox'])for l in sc['expected']['lines']]
        # Blocks are explicit visual-line-derived inputs, not a complete page block detector.
        add('PDF-inline-'+str(i),blocks,c['expected']['style_lines'],c['expected']['link_lines'],sc['request']['page_size'],chain=i)
    data=dict(natural=sorted(str(x)for x in t.PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES),inline=sorted(str(x)for x in t._PDF_INLINE_SPAN_BLOCK_TYPES),targets={k:sorted(str(x)for x in v)for k,v in t._PDF_TEXT_STYLE_TARGET_BLOCK_TYPES.items()},styles=list(t.PDF_TEXT_STYLE_ORDER),span_styles=list(INLINE_STYLE_ORDER),entities=html.entities.html5,invalid_charrefs={str(k):v for k,v in html._invalid_charrefs.items()},invalid_codepoints=sorted(html._invalid_codepoints))
    write_json(config,data)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/matching.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/materialize.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/common.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py','rapid_doc/_vendor/docvortex_core/content/spans.py','rapid_doc/_vendor/docvortex_core/schema.py','rapid_doc/_vendor/docvortex_core/foundation/_text.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'complete inline-output cases;',sum('inline_case'in c for c in cases),'PDF chains')
if __name__=='__main__':main()
