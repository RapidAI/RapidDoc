"""Whole original raw-marginal and repeated-separator stages, with text references."""
import copy,json,random
from common import ROOT,write_json,digest
from export_form_runtime import typed
from export_native_late_image_text import project
from export_native_table_page_policies import empty
from export_native_image_chain import line
def reference(raw):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.auxiliary_text import _classify_raw_page_marginals
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.pipeline import _detect_repeated_header_separator_bboxes
    sources=[typed(s)for s in copy.deepcopy(raw)];_classify_raw_page_marginals(sources);separators=_detect_repeated_header_separator_bboxes(sources);out=copy.deepcopy(raw)
    for s,o in zip(sources,out):o['lines']=[project(l)for l in s.lines]
    return dict(sources=out,header_separator_bboxes=[[list(b)for b in sorted(bs)]for bs in separators])
def base(pages=3):
    result=[]
    for i in range(pages):
        s=empty();s['page_size']=[400.,600.];s['lines']=[line('Original journal heading',[30.,15.,280.,25.],0),line('Research article',[30.,30.,160.,40.],1),line('An original body paragraph with many complete words',[30.,160.,370.,170.],2),line(str(i+1),[350.,570.,365.,580.],3)];s['drawing_lines']=[dict(bbox=[20.,50.,380.,50.],orientation='horizontal',width=.5)];result.append(s)
    return result
def main():
    target=ROOT/'build/migration/native-marginals.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,raw):
        raw=copy.deepcopy(raw)
        for s in raw:s['lines']=[project(l)for l in typed(s).lines]
        cases.append(dict(label=label,request=dict(sources=raw),expected=reference(raw)))
    for mode in ['repeat','single','two','empty','mirrored','non-adjacent','roman','chinese','wrapped-pages','invalid-roman','side-page','landscape-page','font-mismatch','low-font-coverage','too-large','caption','footnote-low','footnote-high','semantic','leader','formula','hierarchy','citation','arxiv','long-similar','short-similar','different','digit-mask','middle','near-header','two-rules','connected-table','no-header-evidence','only-one-above','short-rule','low-rule','rotated']:
        ss=base();
        if mode=='single':ss=ss[:1]
        if mode=='two':ss=ss[:2]
        if mode=='empty':ss=[]
        if mode=='non-adjacent':ss=base(5);ss[1]['lines']=[];ss[2]['lines']=[];ss[3]['lines']=[]
        for i,s in enumerate(ss):
            if mode=='mirrored'and i%2:s['lines'][0]['bbox']=[120.,15.,370.,25.];s['lines'][3]['bbox']=[35.,570.,50.,580.]
            if mode=='roman':s['lines'][3]['text']=['I','II','III'][i]
            if mode=='chinese':s['lines'][3]['text']=['第十一页','第十二页','第十三页'][i]
            if mode=='wrapped-pages':s['lines'][3]['text']=f'Page {i+1} of 12'
            if mode=='invalid-roman':s['lines'][3]['text']='MMMMMM'
            if mode=='side-page':s['lines'][3]['bbox']=[8.,495.,23.,505.]
            if mode=='landscape-page'and i==1:s['page_size']=[600.,400.];s['lines'][3]['bbox']=[150.,375.,165.,385.]
            if mode=='font-mismatch':s['lines'][0].update(font_signature=['Different'if i%2 else'Body',0],font_coverage=1.)
            if mode=='low-font-coverage':s['lines'][0].update(font_signature=['Different'if i%2 else'Body',0],font_coverage=.5)
            if mode=='too-large':s['lines'][0]['effective_height']=40.
            if mode=='caption':s['lines'][0]['caption_start']=True
            if mode in ['footnote-low','footnote-high']:s['lines'][3].update(semantic_type='page_footnote',bbox=[350.,565.,365.,575.]if mode=='footnote-low'else[350.,545.,365.,555.])
            if mode=='semantic':s['lines'][0]['semantic_type']='paragraph_title'
            if mode=='leader':s['lines'][0]['text']=f'Chapter {i} ........ {i+10}'
            if mode=='formula':s['lines'][0]['text']=f'(A{i+1})'
            if mode=='hierarchy':s['lines'][0]['text']=f'1.2.{i+1}'
            if mode=='citation':s['lines'][0]['text']=f'202{i}; Original source citation'
            if mode=='arxiv':s['lines'][0]['text']=f'arXiv: 202{i}.012345'
            if mode=='long-similar':s['lines'][0]['text']='Original journal stable repeated header '+('a'if i==1 else'b')
            if mode=='short-similar':s['lines'][0]['text']='small '+str(chr(65+i))
            if mode=='different':s['lines'][0]['text']=['alpha original research','unrelated title words','third separate topic'][i];s['lines'][1]['text']=str(chr(65+i))
            if mode=='digit-mask':s['lines'][0]['text']=f'Volume {i+1} year 202{i}'
            if mode=='middle':s['lines'][0]['bbox']=[30.,250.,280.,260.]
            if mode=='near-header':s['lines'][0]['bbox']=[30.,60.,280.,70.]
            if mode=='two-rules':s['drawing_lines'].append(copy.deepcopy(s['drawing_lines'][0]))
            if mode=='connected-table':s['drawing_lines'] += [dict(bbox=b,orientation=o,width=.5)for b,o in [([20.,100.,380.,100.],'horizontal'),([20.,50.,20.,100.],'vertical'),([380.,50.,380.,100.],'vertical')]]
            if mode=='no-header-evidence':s['lines'][0]['text']=f'unique {chr(65+i)*30}';s['lines'][1]['text']=str(chr(65+i))
            if mode=='only-one-above':s['lines'][1]['bbox']=[30.,70.,160.,80.]
            if mode=='short-rule':s['drawing_lines'][0]['bbox']=[100.,50.,300.,50.]
            if mode=='low-rule':s['drawing_lines'][0]['bbox']=[20.,100.,380.,100.]
            if mode=='rotated':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                for l in s['lines']:l['bbox']=list(_rotate_bbox_from_upright(tuple(l['bbox']),tuple(s['page_size']),90));l['angle']=90
        add(mode,ss)
    for mode in ['local-continuation','local-outer','local-panel','local-no-proof','local-footer','leading-title','leading-rule','leading-offset','leading-small-body']:
        ss=base();
        for s in ss:
            s['lines']=[line('Short label',[30.,40.,80.,50.],0),line('Long original body phrase',[30.,55.,300.,65.],1),line('Continuation body phrase',[30.,70.,300.,80.],2)]
            s['drawing_lines']=[]
            if mode=='local-no-proof':s['lines']=s['lines'][:2]
            if mode=='local-outer':s['lines']=s['lines'][:2]+[line('outer text',[20.,5.,150.,15.],3)]
            if mode=='local-panel':s['lines']=s['lines'][:2];s['path_infos']=[dict(bbox=[20.,30.,350.,100.],segment_count=5,form_depth=0,source_index=0,fill_visible=True,stroke_visible=False,fill_rgba=[50,100,150,255],rectangle_bboxes=[])]
            if mode=='local-footer':
                for l in s['lines']:l['bbox'][1]+=470.;l['bbox'][3]+=470.
            if mode.startswith('leading-'):
                s['lines']=[line('Document identifier',[60.,30.,340.,40.],0),dict(line('Larger title',[140.,50.,260.,64.],1),effective_height=14.),line('First complete long body paragraph',[30.,85.,370.,95.],2),line('Second complete long body paragraph',[30.,105.,370.,115.],3)]
                if mode=='leading-rule':s['drawing_lines']=[dict(bbox=[20.,45.,380.,45.],orientation='horizontal',width=.5)]
                if mode=='leading-offset':s['lines'][0]['bbox']=[30.,30.,310.,40.]
                if mode=='leading-small-body':
                    for l in s['lines'][2:]:l['effective_height']=14.
        add(mode,ss)
    for name in ['native-vector-document']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if c['label'].startswith('native-document-images/'):add('continuous/'+c['label'],c['request']['sources'])
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf import auxiliary_text as old
    numbers=['','1','- １ -','page II of IV','第两百页','零零一','MMMMM','IIII','IC','一百二十','一二三','1.2','(1)','Page １２３４５','第〇页','第十十页','第百百页','page ı','\x1c3\x1f','三'*4301]
    pairs=[('Repeated journal header volume 2','Repeated journal header volume 3'),('2023; source','2024; source'),('(A2)','(A3)'),('1.2.3','1.2.4'),('short aa','short ab'),('aaab'*80+'c','aaab'*80+'d'),('Original journal research a','Original journal research b')]
    rng=random.Random(31);words=['journal','original','research','中','１２','β',' ','_',':','A','\x1c','İ','2019','arxiv','...','ﬁ']
    for _ in range(350):
        a=''.join(rng.choice(words)for _ in range(rng.randint(1,12)));b=a if rng.randrange(3)==0 else a[:max(0,len(a)-2)]+rng.choice(words);pairs.append((a,b))
    text=[dict(first=a,second=b,matches=old._marginal_text_matches(a,b))for a,b in pairs];numbers=[dict(text=s,value=None if(v:=old._parse_page_number_value(s))is None else str(v))for s in numbers]
    write_json(target,dict(cases=cases,text=text,numbers=numbers,sources={k:digest((ROOT/f'rapid_doc/_vendor/docvortex_core/analyzers/native/pdf/{k}').read_bytes())for k in ['auxiliary_text.py','pipeline.py','table_rules.py']}));print(len(cases),'raw marginal documents;',len(text),'text pairs;',len(numbers),'page numbers')
if __name__=='__main__':main()
