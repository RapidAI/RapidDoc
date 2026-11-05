"""Original public API reference for a nonzero page selection; never overwrite."""
import os
from io import BytesIO
from common import ROOT,digest,write_json

def main():
    target=ROOT/'build/migration/native-api-range.json'
    if target.exists():raise FileExistsError(target)
    os.environ['MINERU_DEVICE_MODE']='cpu';os.environ['MINERU_DONOT_CLEAN_MEM']='1'
    from rapid_doc import RapidDoc
    from PIL import Image
    options=dict(layout_config=dict(model_type='docvortex_rule'),formula_enable=False)
    source=ROOT/'demo/pdfs/示例1-论文模板.pdf'
    result=RapidDoc(ocr_config={'seal_enable':False},**options)(source,start_page_id=1,end_page_id=1)
    assets=[]
    for key,blob in result.images.items():
        with Image.open(BytesIO(blob))as im:assets.append(dict(name=key,mode=im.mode,width=im.width,height=im.height,pixels_sha256=digest(im.tobytes())))
    write_json(target,dict(cases=[dict(source=source.relative_to(ROOT).as_posix(),sha256=digest(source.read_bytes()),options=options,start_page_id=1,end_page_id=1,
        expected=dict(markdown=result.markdown,middle_json=result.middle_json,content_list_json=result.content_list_json),assets=assets)],scope='Original PDF API nonzero native-layout page selection. CPU; formula/stamp disabled.'))
if __name__=='__main__':main()
