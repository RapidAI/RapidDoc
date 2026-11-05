"""All original text_assembly.rows rules, grouping and tight geometry."""
import copy,json
from common import ROOT,write_json
from export_native_geometry import lines
from export_native_late_image_text import project
from export_native_title_layout import line

def reference(q):
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.text_assembly import rows as old
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_to_upright,_transform_axis_lines
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.models import _TextLane,_AxisLine
    ls=lines(q['lines']);p=tuple(q['page_size']);a=q['angle'];g=[(l,_rotate_bbox_to_upright(l.bbox,p,a))for l in ls if l.angle==a];lane=_TextLane(left=q['left'],right=q['right'],lines=g,is_span=q['is_span']);rules=_transform_axis_lines([_AxisLine(bbox=tuple(r['bbox']),orientation=r['orientation'],width=r['width'])for r in q['drawing_lines']],p,a);h=p[0]if a in [90,270]else p[1];reg,mad=q['regular_gap'],q['gap_mad'];visuals=[_rotate_bbox_to_upright(tuple(b),p,a)for b in q['visual_bboxes']]
    out={}
    for name,fn,args in [('top','_top_marginal_text_break_sources',(lane,h)),('bullets','_repeated_bullet_break_sources',(g,)),('italic','_italic_quote_to_body_break_sources',(lane,)),('caption','_caption_to_body_break_sources',(lane,)),('structured','_structured_text_break_sources',(lane,reg,mad)),('isolated','_isolated_indented_paragraph_break_sources',(lane,reg,mad)),('visual','_centered_visual_reset_break_sources',(lane,visuals,h)),('typography','_leading_typography_reset_break_sources',(lane,reg,mad)),('formula','_formula_style_text_row_break_sources',(lane,)),('front','_front_matter_keyword_break_sources',(lane,h,q['page_index'])),('explicit','_explicit_text_break_sources',(lane,)),('prose','_prose_paragraph_break_sources',(lane,reg,mad)),('cjk_prose','_cjk_prose_break_sources',(lane,reg,mad)),('cjk_entries','_cjk_entry_break_sources',(lane,))]:out[name]=sorted(getattr(old,fn)(*args))
    out['emphasized']=old._component_starts_with_emphasized_row([l for l,_ in g]);out['hanging']={str(k):v for k,v in old._build_hanging_indent_group_map(lane,[tuple(b)for b in q['table_bboxes']],rules).items()};out['local_lanes']={str(k):dict(left=l.left,right=l.right,is_span=l.is_span,members=[next(i for i,(x,_)in enumerate(g)if x is r)for r,_ in l.lines])for k,l in old._infer_local_text_lane_map(lane).items()};out['starts_reference']=[old._starts_structural_reference_entry(x,y)for x,y in zip(g,g[1:])];out['tight']=old._local_tight_output_line_bboxes([l for l,_ in g],p,a);return out

def main():
    target=ROOT/'build/migration/native-text-rows.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();cases=[]
    def add(label,q):
        q=copy.deepcopy(q)
        for k,v in dict(page_size=[400.,600.],angle=0,left=30.,right=370.,is_span=False,regular_gap=4.,gap_mad=1.,table_bboxes=[],drawing_lines=[],visual_bboxes=[],page_index=0).items():q.setdefault(k,v)
        q['lines']=[project(l)for l in lines(q['lines'])];c=dict(label=label,request=q)
        try:c['expected']=reference(q)
        except (ValueError,TypeError,IndexError,AttributeError,ZeroDivisionError)as e:c.update(error='invalid_input',old_exception=type(e).__name__)
        cases.append(c)
    for c in json.loads((ROOT/'build/migration/native-text-connections.json').read_text(encoding='utf-8'))['cases']:add(c['label'],c['request'])
    def make(texts,lefts=None,widths=None,ys=None,heights=None):return [line(t,[float((lefts or[30.]*len(texts))[i]),float((ys or[180.+14.*j for j in range(len(texts))])[i]),float((lefts or[30.]*len(texts))[i]+(widths or[340.]*len(texts))[i]),float((ys or[180.+14.*j for j in range(len(texts))])[i]+(heights or[10.]*len(texts))[i])],i)for i,t in enumerate(texts)]
    directs=[]
    directs.append(('hanging',dict(lines=make(['Original entry first line','Original continuation','Next entry first line','Next continuation','Final entry first line','Final continuation'],lefts=[30,50,30,50,30,50],widths=[290,200,290,200,290,200]))))
    directs.append(('local-lanes',dict(lines=make(['Original narrow paragraph words']*4,widths=[160]*4))))
    directs.append(('isolated',dict(lines=make(['Tail.','Original paragraph begins here','Following ordinary words'],lefts=[30,45,30],widths=[80,320,330]))))
    directs.append(('prose',dict(lines=make(['Original full line words','Ordinary short sentence.','Following original paragraph','Following ordinary continuation'],widths=[340,230,340,340]))))
    directs.append(('prose-indent',dict(lines=make(['Original ordinary sentence.','Following indented paragraph','Original continuation words','Another indented paragraph'],lefts=[30,45,30,45],widths=[340,320,340,320]))))
    directs.append(('cjk-prose',dict(lines=make(['这是前一段落完整结束。','这是下一段落原始的完整正文内容','这是另外的正文内容'],ys=[180,198,212]))))
    directs.append(('cjk-label',dict(lines=make(['这是前一段落完整结束。','风险提示：这是原来的正文内容','这是后续原来的正文内容'],ys=[180,198,212]))))
    directs.append(('cjk-entries',dict(lines=make(['1、原始条目正文','2、原始条目正文','3、原始条目正文']))))
    directs.append(('cjk-books',dict(lines=make(['《原始报告》','原始内容——2026-10-28','《原始报告》','原始内容——2026-10-28','《原始报告》']))))
    directs.append(('bullets',dict(lines=make(['• Original item','• Next item','• Final item']))))
    directs.append(('separate-bullets',dict(lines=make(['•','Original item words','•','Next item words'],lefts=[30,45,30,45],widths=[5,300,5,300],ys=[180,180,200,200]))))
    directs.append(('cross-column-bullets',dict(lines=make(['Original item introduction:','• Original left item','• Original right item','• Next right item'],lefts=[30,30,210,210],widths=[140,140,140,140],ys=[160,180,180,200]))))
    directs.append(('explicit',dict(lines=make(['Abstract: Original abstract','1. Original entry','2. Next entry','E-mail: original@example.org','主要研究方向：原始内容','当前研究方向：原始内容'],widths=[200]*6))))
    directs.append(('intro-list',dict(lines=make(['Original introduction:','(i) Original list entry','Original following words'],lefts=[30,45,30],widths=[200,300,340]))))
    directs.append(('front',dict(lines=make(['Keywords: original','中图分类号：原始']))))
    directs.append(('formula',dict(lines=make(['Original formula context words']*3,widths=[240]*3,lefts=[80]*3))))
    directs[-1][1]['lines'][0]['paragraph_formula_context']=True;directs[-1][1]['lines'][1]['paragraph_formula_context']=True
    directs.append(('top',dict(lines=make(['12','Original ordinary body text words','Original ordinary body text words','Original ordinary body text words'],ys=[10,40,54,68],widths=[15,340,340,340]))))
    directs.append(('italic',dict(lines=make(['Original indented quote','Original quote ends here.','Original body text has six distinct ordinary words','Original body continues here'],lefts=[45,45,30,30],widths=[300,300,340,340]))))
    for l in directs[-1][1]['lines'][:2]:l['font_signature']=['Body-Italic',64]
    directs.append(('caption',dict(lines=make(['Fig. 1 Original caption','Original larger body words'],ys=[180,198],heights=[10,12]))));directs[-1][1]['lines'][0]['caption_start']=True
    directs.append(('visual',dict(lines=make(['Original short annotation','Original wide annotation'],lefts=[160,80],widths=[80,240]),visual_bboxes=[[60.,80.,340.,178.]])))
    directs.append(('structured',dict(lines=make(['Original short sentence.','Original labelled paragraph','Original short sentence.','Original labelled paragraph','Original short sentence.','Original labelled paragraph'],widths=[180,340]*3))));
    for l in directs[-1][1]['lines'][1::2]:l['leading_emphasis_width']=25.
    directs.append(('typography',copy.deepcopy(directs[-1][1])))
    for l in directs[-1][1]['lines'][1::2]:l['leading_typography_width']=25.
    directs.append(('centered-sentence',dict(lines=make(['Original short sentence.','Original centered sentence.'],lefts=[30,100],widths=[160,200],ys=[180,200]))))
    directs.append(('tight',dict(lines=make(['Original ink geometry','[1] Original reference'],lefts=[50,30]))))
    for l in directs[-1][1]['lines']:l['ink_bbox']=[l['bbox'][0]+1,l['bbox'][1]+1,l['bbox'][2]-1,l['bbox'][3]-1]
    for label,q in directs:
        add('direct/'+label,q)
        for mode in ['reverse','table','rule','span','duplicate-id','zero-height','no-font','semantic','rotation']:
            v=copy.deepcopy(q)
            if mode=='reverse':v['lines'].reverse()
            if mode=='table':v['table_bboxes']=[[0.,192.,400.,194.]]
            if mode=='rule':v['drawing_lines']=[dict(bbox=[0.,192.,400.,192.],orientation='horizontal',width=.5)]
            if mode=='span':v['is_span']=True
            if mode=='duplicate-id':
                for l in v['lines']:l['source_index']=7
            if mode=='zero-height':
                for l in v['lines']:l['effective_height']=0.
            if mode=='no-font':
                for l in v['lines']:l['font_signature']=None
            if mode=='semantic':v['lines'][0]['semantic_type']='paragraph_title'
            if mode=='rotation':
                from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.geometry import _rotate_bbox_from_upright
                v.update(page_size=[600.,400.],angle=90)
                for l in v['lines']:
                    l.update(angle=90,bbox=_rotate_bbox_from_upright(tuple(l['bbox']),tuple(v['page_size']),90))
                    if l.get('ink_bbox'):l['ink_bbox']=_rotate_bbox_from_upright(tuple(l['ink_bbox']),tuple(v['page_size']),90)
                v['visual_bboxes']=[_rotate_bbox_from_upright(tuple(b),tuple(v['page_size']),90)for b in v.get('visual_bboxes',[])]
            add('direct/'+label+'/'+mode,v)
    write_json(target,dict(cases=cases));print(len(cases),'complete original row-rule cases;',sum('error'in c for c in cases),'exceptions');print({k:sum(bool(c.get('expected',{}).get(k))for c in cases)for k in cases[0]['expected']})
if __name__=='__main__':main()
