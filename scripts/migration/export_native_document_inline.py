"""Run original full document-inline orchestration, then raw-source through inline output."""
import copy,json
from types import SimpleNamespace as NS
from common import ROOT,write_json
from export_native_inline_clusters import clean
from export_native_spacing import reference as spacing
def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _materialize_document_inline
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline import types as t
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.spacing import PDFTextSpacingLine
    def evidence(l,kind):
        if kind=='style':return t.PDFTextStyleLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextStyleRange(**dict(r,styles=tuple(r['styles'])))for r in l['style_ranges']),l['source_index'])
        if kind=='link':return t.PDFTextLinkLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextLinkRange(**r)for r in l['link_ranges']),l['source_index'])
        return t.PDFTextScriptLine(tuple(l['bbox']),l['text'],tuple(t.PDFTextScriptRange(**dict(r,bbox=tuple(r['bbox'])))for r in l['script_ranges']),l['source_index'],l['angle'])
    pages=copy.deepcopy(q['finalized_pages']);prepared=[NS(script_lines=[evidence(l,'script')for l in p.get('script_lines',[])])for p in q['prepared_pages']]
    src=NS(page_sizes=[tuple(p)for p in q['page_sizes']],page_style_lines=[[evidence(l,'style')for l in p]for p in q['page_style_lines']],page_link_lines=[[evidence(l,'link')for l in p]for p in q['page_link_lines']],page_spacing_lines=[[PDFTextSpacingLine(**dict(l,bbox=tuple(l['bbox']),boundaries=tuple(l['boundaries']),suppressed_boundaries=tuple(l.get('suppressed_boundaries',[]))))for l in p]for p in q.get('page_spacing_lines',[])])
    diagnostics=copy.deepcopy(q.get('script_diagnostics'));_materialize_document_inline(pages,prepared,src,diagnostics)
    out=dict(pages=pages)
    if diagnostics is not None:out['script_diagnostics']=diagnostics
    return clean(out)
def request(pages,prepared,raw):
    return dict(finalized_pages=pages,prepared_pages=prepared,page_sizes=[p.get('page_size',p.get('source',{}).get('page_size'))for p in prepared],page_style_lines=[p.get('style_lines',[])for p in prepared],page_link_lines=[p.get('link_lines',[])for p in prepared],page_spacing_lines=[p.get('spacing_lines',spacing(dict(lines=p['source']['lines'],space_before=p.get('space_before')),'prepare'))for p in raw])
def main():
    targets=[ROOT/f'build/migration/{n}.json'for n in ['native-document-inline','native-final-document-inline-chain']]
    if any(t.exists()for t in targets):raise FileExistsError('immutable document inline references exist')
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=clean(q);c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except(ValueError,TypeError,KeyError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for name in ['native-inline-output','native-script-output']:
        for i,c in enumerate(json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))['cases']):
            q=c['request'];base=dict(finalized_pages=[q['blocks']],prepared_pages=[dict(script_lines=q.get('script_lines',[]))],page_sizes=[q['page_size']],page_style_lines=[q['style_lines']],page_link_lines=[q['link_lines']])
            for mode in ['base','spacing','diagnostics']:
                v=copy.deepcopy(base)
                if mode!='base':v['page_spacing_lines']=[[dict(bbox=l['bbox'],text=l['text'],source_index=l['source_index'],boundaries=list(range(1,len(l['text']),3)),suppressed_boundaries=[])for l in q['style_lines']or q['link_lines']]]
                if mode=='diagnostics':v['script_diagnostics']=[dict(page_index=0,page_size=q['page_size'],retained='original snapshot',script_lines=q.get('script_lines',[]),lines=[])]
                add(f'{name}/{i}/{mode}',v)
    base=copy.deepcopy(cases[0]['request'])
    for field in ['finalized_pages','prepared_pages','page_sizes','page_style_lines','page_link_lines']:
        q=copy.deepcopy(base);q[field]=[];add('strict-zip/'+field,q)
    for field in ['page_spacing_lines','script_diagnostics']:
        q=copy.deepcopy(base);q['finalized_pages']*=2;q['prepared_pages']*=2;q['page_sizes']*=2;q['page_style_lines']*=2;q['page_link_lines']*=2;q[field]=[{}]if field=='script_diagnostics'else[[]];add('strict-short/'+field,q)
    write_json(targets[0],dict(cases=cases));print(len(cases),'full inline orchestrations;',sum('error'in c for c in cases),'exceptions')
    f=json.loads((ROOT/'build/migration/native-final-page-output-chain.json').read_text(encoding='utf-8'))
    for c in f['cases']:
        c['request']['materialize_inline']=True
        if 'expected'not in c:continue
        e=c['expected'];q=request([p['model_list']for p in e['block_pages']],e['pages'],c['request']['pages']);e['finalized_pages']=reference(q)['pages']
    write_json(targets[1],f);print(len(f['cases']),'raw-source through complete inline document chains')
if __name__=='__main__':main()
