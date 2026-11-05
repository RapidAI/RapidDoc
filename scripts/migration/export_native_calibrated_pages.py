"""Compose original post-geometry split/reclaim/fraction stages and block identities."""
import argparse,copy,json
from common import ROOT,write_json
from export_native_late_image_text import reference as images
from export_native_leading_fractions import reference as fractions
from export_form_runtime import typed
from export_native_late_image_text import project

def reference(q):
    out=copy.deepcopy(q);image=images(q);offset=0
    for key in ['caption_graphics','form_image_blocks','graphic_blocks','raster_image_blocks']:
        n=len(q.get(key,[]));out[key]=image['image_containers'][offset:offset+n];offset+=n
    out['reclaimed_line_indices']=image['reclaimed_line_indices'];out['remaining_lines']=fractions(dict(lines=image['remaining_lines'],drawing_lines=q['source']['drawing_lines'],page_size=q['source']['page_size']))
    out['fixed_blocks']=list(q.get('prior_fixed_blocks',[]))
    for key in ['rule_code_blocks','annotation_blocks','table_blocks','code_blocks','form_image_blocks','graphic_blocks','raster_image_blocks','vector_formula_blocks','caption_graphics','index_blocks']:out['fixed_blocks']+=out.get(key,[])
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--canonical',action='store_true');args=parser.parse_args()
    target=ROOT/('build/migration/native-calibrated-pages-canonical.json'if args.canonical else'build/migration/native-calibrated-pages.json')
    if target.exists():raise FileExistsError(target)
    cases=[]
    for name in ['native-late-image-text','native-leading-fractions','native-vector-document']:
        f=json.loads((ROOT/f'build/migration/{name}.json').read_text(encoding='utf-8'))
        for c in f['cases']:
            if name=='native-vector-document':
                if 'expected'not in c:continue
                requests=copy.deepcopy(c['expected']['pages'])
            elif name=='native-leading-fractions':
                if c['label'].startswith('continuous/'):continue
                r=c['request'];requests=[dict(source=dict(lines=r['lines'],drawing_lines=r['drawing_lines'],page_size=r['page_size']),remaining_lines=r['lines'],strong_graphic_bboxes=[],table_bboxes=[])]
            else:requests=[copy.deepcopy(c['request'])]
            for i,q in enumerate(requests):
                if args.canonical:q['remaining_lines']=[project(l)for l in typed(dict(q['source'],lines=q['remaining_lines'])).lines]
                cases.append(dict(label=f'{name}/{c["label"]}/{i}',request=q,expected=reference(q)))
    write_json(target,dict(cases=cases));print(len(cases),'original calibrated-page continuations')
if __name__=='__main__':main()
