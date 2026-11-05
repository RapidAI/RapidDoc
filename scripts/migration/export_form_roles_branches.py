"""Few full old branch scenes, not per-helper build or a broad regression run."""
import copy,json
from common import ROOT,write_json
from export_form_runtime import normalized_reference,document_reference
from export_native_table_page_policies import empty

def info(i=0,parent=None,bbox=None,frame=None):
    bbox=bbox or[0.,0.,400.,600.];frame=frame or[0.,0.,400.,600.]
    return dict(instance_id=i,parent_id=parent,occurrence=[0]if parent is None else[0,i-1],bbox=bbox,declared_bbox=frame,media_bbox=[0.,0.,400.,600.],matrix=[1.,0.,0.,1.,0.,0.],clip=None,paint_order=i,text_indices=[],path_indices=[],image_bboxes=[],structure_valid=True)

def figures(shared=False):
    s=empty();s['page_size']=[400.,600.];f=info();s['form_infos']=[f,info(1,0,[30.,200.,170.,330.],[30.,200.,170.,330.]),info(2,0,[220.,200.,360.,330.],[220.,200.,360.,330.])];s['form_bboxes']=[[0.,0.,400.,600.]]
    captions=[('Figure 1: Both panels',[30.,340.,360.,350.])]if shared else[('Figure 1: First panel',[30.,340.,170.,350.]),('图2：第二图',[220.,340.,360.,350.])]
    for i,(text,b)in enumerate(captions):
        ch=dict(char='X',char_idx=i,bbox=[b[0],b[1],b[0]+5,b[3]],tight_bbox=[b[0],b[1],b[0]+5,b[3]])
        s['lines'].append(dict(text=text,bbox=b,source_index=i,angle=0,effective_height=10.,chars=[ch],font_signature=['Body',0]));s['chars'].append(ch);f['text_indices'].append(i)
    return s

def main():
    target=ROOT/'build/migration/native-form-roles-branches.json'
    if target.exists():raise FileExistsError(target)
    from loguru import logger
    logger.remove();original=json.loads((ROOT/'build/migration/native-form-runtime-canonical.json').read_text());sources=original['request']['sources'];cases=[]
    def add(label,s):
        cases.append(dict(label=label,request=dict(sources=copy.deepcopy(s)),normalized=normalized_reference(s),expected=document_reference(s)))
    add('independent-caption-ownership',[figures()]);add('one-caption-two-panels',[figures(True)])
    for mode in ['structure-unknown','non-root-decoration','missing-body','duplicate-known','mixed-known-unknown','root-blocks-descendant']:
        s=copy.deepcopy(sources[6]if mode=='non-root-decoration'else sources[0]if mode in['duplicate-known','mixed-known-unknown']else sources[4]);f=s['form_infos'][0]
        if mode=='structure-unknown':f['structure_valid']=False;f['declared_bbox']=None
        if mode=='non-root-decoration':f['paint_order']=1
        if mode=='missing-body':s['lines']=[];s['chars']=[];f['text_indices']=[]
        if mode in ['duplicate-known','mixed-known-unknown']:
            other=copy.deepcopy(f);other['instance_id']=len(s['form_infos']);other['occurrence']=[99];other['structure_valid']=mode=='duplicate-known';other['text_indices']=f['text_indices'][1:];f['text_indices']=f['text_indices'][:1];s['form_infos'].append(other)
        if mode=='root-blocks-descendant':
            f['structure_valid']=False;f['declared_bbox']=None
            child=copy.deepcopy(f);child.update(instance_id=10,parent_id=0,occurrence=[0,9],structure_valid=True,declared_bbox=[0.,0.,400.,600.]);s['form_infos'].append(child)
        add(mode,[s])
    table=copy.deepcopy(json.loads((ROOT/'build/migration/native-code-table-grid-mixtures.json').read_text())['cases'][0]['request']['source']);table['lines']=[l for l in table['lines']if l['source_index']>=1000];table['page_size']=[400.,600.];table['path_infos']=[];table['form_bboxes']=[[0.,0.,400.,600.]];f=info();table['form_infos']=[f];table['chars']=[]
    for i,l in enumerate(table['lines']):
        ch=dict(char='X',char_idx=i,bbox=l['bbox'],tight_bbox=l['bbox']);l['chars']=[ch];table['chars'].append(ch);f['text_indices'].append(i)
    for i,r in enumerate(table['drawing_lines']):
        if r['orientation']=='horizontal':
            b=list(r['bbox']);b[3]=b[1]+.1;table['path_infos'].append(dict(bbox=b,source_index=i,form_depth=1,segment_count=2,stroke_visible=True,fill_visible=False,rectangle_bboxes=[]));f['path_indices'].append(i)
    add('table-family-inheritance',[copy.deepcopy(sources[4]),table]);add('table-without-family',[table]);add('sparse-before-family',[copy.deepcopy(sources[5]),copy.deepcopy(sources[4])]);add('empty-document',[])
    write_json(target,dict(sources=original['sources'],cases=cases));print(len(cases),'complete role/document branch scenes')

if __name__=='__main__':main()
