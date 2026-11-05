"""Freeze actual old script-line assignments, output spans and materialized diagnostics."""
import copy,json,random
from dataclasses import asdict
from common import ROOT,digest,write_json
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-script-output.json'
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline import materialize as old,types as t
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.scripts import detect_pdf_text_script_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _LineItem,_AxisLine
    from loguru import logger
    logger.remove();cases=[]
    def add(label,blocks,scripts,styles=(),links=(),size=(400.,600.),chain=None):
        blocks=copy.deepcopy(blocks);q=clean(dict(blocks=blocks,script_lines=scripts,style_lines=styles,link_lines=links,page_size=size))
        ss=[t.PDFTextStyleLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextStyleRange(r['start'],r['end'],tuple(r['styles']))for r in l['style_ranges']),l['source_index'])for l in styles]
        ll=[t.PDFTextLinkLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextLinkRange(**r)for r in l['link_ranges']),l['source_index'])for l in links]
        pp=[t.PDFTextScriptLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextScriptRange(**r)for r in l['script_ranges']),l['source_index'],l['angle'])for l in scripts]
        diagnostics=[];old.apply_pdf_inline_evidence(blocks,ll,ss,pp,size,materialized_diagnostics=diagnostics)
        c=dict(label=label,request=q,expected=clean(dict(blocks=blocks,materialized_ranges=diagnostics)))
        if chain is not None:c['output_case']=chain
        cases.append(c)
    def b(text,box=(0.,0.,1.,1.),kind='text',regions=None):
        v=dict(type=kind,content=text,bbox=box,retained='metadata')
        if regions is not None:v['_inline_math_regions']=regions
        return v
    def s(text='base2',start=4,end=5,role='superscript',bbox=(10.,20.,200.,30.),tight=(50.,18.,56.,24.),source=0,formula=False):
        return dict(text=text,bbox=bbox,source_index=source,angle=0,script_ranges=[dict(start=start,end=end,style=role,bbox=tight,stable_body_count=4,formula_region=formula)])
    for text,script in [('base2',s()),('base2 repeated base2',s()),('abc2xyz',s('abc2xyz',3,4)),('left \\(x\\) right2',s('leftXright2',10,11)),('x2',s('x2',1,2))]:
        for kind in ['text','index','paragraph_title','header','caption','table','image']:
            for role in ['superscript','subscript']:
                p=copy.deepcopy(script);p['script_ranges'][0]['style']=role;add('plain-'+text+'-'+kind+'-'+role,[b(text,kind=kind)],[p])
    for whole in ['no match','base2','other']:
        for fallback in ['2','base2','2 2','xx2yy']:
            add('fallback-'+whole+'-'+fallback,[b(whole,box=(0.,0.,300.,100.)),b(fallback,box=(40.,10.,80.,40.))],[s(tight=(50.,18.,56.,24.))])
    for formula in [False,True]:
        for region in [[],[(45.,15.,60.,30.)],[(.1,.02,.2,.06)],[(300.,300.,310.,310.)]]:
            add('late-region-'+str(formula)+str(region),[b('base2',regions=region)],[s(formula=formula)])
    for contents in [['a','x2'],['x2','a'],['base2','2'],['2','base2']]:
        add('cursor-'+str(contents),[b(' '.join(contents))],[s(contents[0],len(contents[0])-1,len(contents[0]),source=0),s(contents[1],len(contents[1])-1,len(contents[1]),source=1)])
    for indices in [(0,0),(7,2),(2,7)]:
        add('identity-'+str(indices),[b('base2 other3')],[s(source=indices[0]),s('other3',5,6,source=indices[1])])
    for content in ['inter2national','inter2-national','first \\(x\\) base2','base2 base2','  base2  ','baseﬁ2']:
        add('content-'+content,[b(content)],[s()])
    add('empty-cleans-every-block',[b('x',regions=[]),b('table',kind='table',regions=[(0.,0.,1.,1.)])],[])
    rng=random.Random(27003)
    for i in range(90):
        text=rng.choice(['abc2 def3','base2 base2','x2 + y3','a2b3c4']);scripts=[]
        for n in range(rng.randint(1,3)):
            pos=rng.randrange(len(text));scripts.append(s(text,pos,pos+1,rng.choice(['superscript','subscript']),source=n))
        # Avoid deliberately contradictory roles: the strict old TextSpan schema rejects sup+sub overlap.
        try:add('random-'+str(i),[b(text)],scripts)
        except ValueError:pass
    output=json.loads((ROOT/'build/migration/native-inline-output.json').read_text(encoding='utf-8'));inline=json.loads((ROOT/'build/migration/native-inline-evidence.json').read_text(encoding='utf-8'));source=json.loads((ROOT/'build/migration/native-source-lines.json').read_text(encoding='utf-8'))
    for index,c in enumerate(output['cases']):
        if'inline_case'not in c:continue
        ic=inline['cases'][c['inline_case']];sc=source['cases'][ic['source_case']];lines=[_LineItem(**copy.deepcopy(l))for l in sc['expected']['lines']];all_chars=[x for l in lines for x in l.chars]
        tight={x['char_idx']:x['tight_bbox']for x in all_chars if x.get('tight_bbox')is not None};origins={x['char_idx']:x['origin']for x in all_chars if x.get('origin')is not None}
        drawings=[_AxisLine(tuple(r['bbox']),r['width'],r['orientation'])for r in sc['expected']['drawing_lines']]
        payload=detect_pdf_text_script_lines(lines,tuple(sc['request']['page_size']),tight,origins,all_chars=all_chars,drawing_lines=drawings)
        q=c['request'];add('PDF-output-'+str(index),q['blocks'],clean([asdict(p)for p in payload]),q['style_lines'],q['link_lines'],q['page_size'],chain=index)
    refs=['rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/matching.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/materialize.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/scripts.py','rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/inline/types.py']
    write_json(target,dict(sources={p:digest((ROOT/p).read_bytes())for p in refs},cases=cases));print('Frozen',len(cases),'script outputs,',sum(bool(c['expected']['materialized_ranges'])for c in cases),'diagnostic positives,',sum('output_case'in c for c in cases),'PDF chains,',sum(bool(l['script_ranges'])for c in cases if'output_case'in c for l in c['request']['script_lines']),'actual PDF script-positive lines')
if __name__=='__main__':main()
