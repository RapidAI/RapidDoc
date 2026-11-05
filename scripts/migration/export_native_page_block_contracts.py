"""Python whitespace and character metadata contracts for local list inference."""
from common import ROOT,write_json
from export_native_caption_block_rules import reference
from export_native_text_block_edges import block
from export_native_title_layout import line
from export_native_final_text_lines import typed_lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-page-block-contracts.json'
    if target.exists():raise FileExistsError(target)
    cases=[]
    for suffix in ['\x1c','\x1d','\x1e','\x1f','\u0085','\u00a0','\u2007','\u3000','Original words','\x1cOriginal words']:
        ls=[project(l)for l in typed_lines([line(f'{letter}. {suffix}',[30.,180.+14*i,370.,190.+14*i],i)for i,letter in enumerate('abc')])];b=block('Original local-list text',[30.,180.,370.,218.],rows=[l['bbox']for l in ls],_text_lines=ls);q=dict(blocks=[b],page_size=[400.,600.],page_index=0,image_bboxes=[],numeric_grid_bboxes=[],bullet_paths=[],kind='lists');cases.append(dict(label='whitespace/'+repr(suffix),request=q,expected=reference(q)))
    write_json(target,clean(dict(cases=cases)));print(len(cases),'local-list whitespace contracts')
if __name__=='__main__':main()
