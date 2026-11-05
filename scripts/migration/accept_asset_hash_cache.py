"""Freeze/check asset output and invalid-PNG errors across lazy page hashing."""

# Historical checkpoints are development tools, not part of the installed SDK.
__import__('runpy').run_path(str(__import__('pathlib').Path(__file__).with_name('preview_compat.py')))['activate']()
import argparse
import hashlib
import io
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args()
    from PIL import Image
    from rapid_doc_rust import assemble_model_page_with_assets
    from accept_doc_performance import installed_binary

    def png(mode,w=48,h=40):
        channels={'L':1,'LA':2,'RGB':3,'RGBA':4}[mode]
        pixels=bytes((x*17+y*37+c*61)%256 for y in range(h) for x in range(w) for c in range(channels))
        with Image.frombytes(mode,(w,h),pixels) as image:
            output=io.BytesIO();image.save(output,format='PNG');return output.getvalue()

    class Writer:
        def __init__(self):self.rows=[]
        def write(self,name,data):
            with Image.open(io.BytesIO(data)) as image:
                self.rows.append(dict(name=name,mode=image.mode,size=list(image.size),
                    pixels=hashlib.sha256(image.tobytes()).hexdigest(),png=hashlib.sha256(data).hexdigest()))

    def det(kind,box=(2,3,20,22),order=0,html=''):
        x,y,r,b=box
        return dict(category_id=kind,poly=[x,y,r,y,r,b,x,b],score=.95,original_order=order,latex='x+y',html=html)

    rows=[]
    for mode in ['L','LA','RGB','RGBA']:
        image=png(mode)
        fill=dict(uuid='FIXED-UUID',bbox=[1,2,5,6],source=png('RGB',9,7))
        cases=[('empty',[],{},image),
            ('image',[det(3)],{},image),('table',[det(5,html='<table><tr><td>x</td></tr></table>')],{},image),
            ('seal',[det(14)],{},image),
            ('fill',[det(5,html='<table><tr><td>FIXED-UUID FIXED-UUID</td></tr></table>')],dict(table_fill_images=[fill]),image),
            ('mixed',[det(3),det(14,order=1)],{},image),
            ('original',[det(3)],dict(extract_original_image=True,original_images=[dict(bbox=[2,3,20,22],source=png('RGB',9,7))]),image),
            ('invalid-empty',[],{},b'not a PNG'),('invalid-image',[det(3)],{},image[:33])]
        for name,dets,options,data in cases:
            writer=Writer()
            try:
                page=assemble_model_page_with_assets(data,dict(layout_dets=dets),image_writer=writer,
                    page_width=48,page_height=40,page_index=7,**options)
                row=dict(mode=mode,name=name,page=page,assets=writer.rows)
            except Exception as error:
                assert name.startswith('invalid-'),(mode,name,error)
                row=dict(mode=mode,name=name,error_type=type(error).__name__,error=str(error))
            assert name.startswith('invalid-')==('error' in row),(mode,name)
            rows.append(row)
    if args.reference:
        old=json.loads(args.reference.read_text(encoding='utf8'))
        assert json.dumps(old['rows'],sort_keys=True)==json.dumps(rows,sort_keys=True),'Asset or PNG error contract changed'
    args.output.write_text(json.dumps(dict(installed_rust=installed_binary(),rows=rows,
        matches_reference=bool(args.reference)),ensure_ascii=False,indent=2),encoding='utf8')
    print(f'{len(rows)} asset/error checks; exact={bool(args.reference)}')


if __name__=='__main__':main()
