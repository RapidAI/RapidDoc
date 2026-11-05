"""Full original pixel-derived glyph weight, including first-tie grouping and conditional rendering."""
import copy,json
from dataclasses import asdict
from types import SimpleNamespace as NS
from PIL import Image,ImageDraw
from common import ROOT,write_json,digest
from export_native_final_text_lines import typed_lines
from export_native_late_image_text import project
from export_native_inline_clusters import clean
def main():
    target=ROOT/'build/migration/native-glyph-weight.json';folder=ROOT/'build/migration/glyph-weight';folder.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    from rapid_doc._vendor.docvortex_core.analyzers.native.pdf.inline.glyph_weight import _glyph_font_groups_python,glyph_weight_style_lines
    from loguru import logger
    logger.remove();cases=[]
    for width in [3,4,5,6,7,8,9,10,11]:
        image=Image.new('RGB',(660,120),'white');draw=ImageDraw.Draw(image);raw=[]
        for f in range(2):
            cs=[]
            for n in range(24):
                col=n%8;row=n//8;x=10+col*35+f*330;y=10+row*32;box=[float(x),float(y),float(x+24),float(y+24)]
                draw.rectangle((x+6,y+3,x+6+(3 if f==0 else width)-1,y+20),fill='black')
                cs.append(dict(char='ABCDEFGH'[col],char_idx=f*100+n,bbox=box,tight_bbox=box,font=dict(name='UnknownReference'if f==0 else'UnknownTarget',flags=0,size=24.,weight=400. if f==0 else None),writing_angle=0.))
            raw.append(dict(text=''.join(c['char']for c in cs),bbox=[10.+f*330,10.,300.+f*330,100.],angle=0,source_index=f,chars=cs,effective_height=24.,em_height=24.,font_signature=['UnknownReference'if f==0 else'UnknownTarget',0]))
        path=folder/f'width-{width}.png'
        if path.exists():raise FileExistsError(path)
        image.save(path);image.close()
        for mode in ['base','reverse','known-weight','italic','already-bold','insufficient-count','insufficient-letters','reference-tie','geometry-map','wrong-direction','no-render','bad-origin']:
            ls=copy.deepcopy(raw);tight={}
            if mode=='reverse':ls.reverse()
            if mode=='known-weight':
                for c in ls[1]['chars']:c['font']['weight']=500.
            if mode=='italic':
                for c in ls[1]['chars']:c['font']['flags']=64
            if mode=='already-bold':
                for c in ls[1]['chars']:c['font']['weight']=700.
            if mode=='insufficient-count':ls[1]['chars']=ls[1]['chars'][:19]
            if mode=='insufficient-letters':ls[1]['chars']=[c for c in ls[1]['chars']if c['char']!='H']
            if mode=='reference-tie':
                for l in ls:
                    for c in l['chars']:c['font']['weight']=None
            if mode=='geometry-map':
                tight={str(c['char_idx']):c['bbox']for l in ls for c in l['chars']}
                for l in ls:
                    for c in l['chars']:c.pop('tight_bbox')
            if mode=='wrong-direction':ls[1]['angle']=90
            if mode=='bad-origin':
                for c in ls[1]['chars']:c['tight_bbox']=[-100.,-100.,-90.,-80.]
            ls=typed_lines(ls);q=clean(dict(lines=[project(l)for l in ls],tight_bboxes=tight,scale=1.));geometry=NS(tight_bboxes={int(i):tuple(b)for i,b in tight.items()});groups,counts=_glyph_font_groups_python(ls,geometry);fonts=list(groups);eligible=[f for f in fonts if counts[f]>=20 and len(groups[f])>=8];reference=max(eligible,key=lambda f:counts[f])if len(eligible)>=2 else None;candidate=[f for f in eligible if f!=reference and f[2]is None and len(groups[f].keys()&groups[reference].keys())>=8]if reference is not None else[]
            plan=clean(dict(groups=[dict(font=f,count=counts[f],letters=list(groups[f].items()))for f in fonts],reference=fonts.index(reference)if candidate else None,candidates=[fonts.index(f)for f in candidate]));calls=[]
            def render():calls.append(True);return NS(pil_image=Image.open(path),scale=1.)
            out=glyph_weight_style_lines(ls,geometry,None if mode=='no-render'else render)
            cases.append(dict(label=f'{width}/{mode}',request=q,png=None if mode=='no-render'else path.relative_to(ROOT).as_posix(),png_sha256=digest(path.read_bytes()),plan=plan,expected=clean([asdict(l)for l in out]),render_calls=len(calls)))
    write_json(target,dict(cases=cases));print(len(cases),'complete original glyph-weight cases;',sum(bool(c['expected'])for c in cases),'bold positives')
if __name__=='__main__':main()
