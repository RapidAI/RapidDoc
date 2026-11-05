"""Original cv2.fillPoly crop masking; keep this compatibility kernel exact."""
from io import BytesIO

def mask_crops(plan,formula_bbox_expand_px=2):
    import cv2
    import numpy as np
    from PIL import Image
    dets=plan['model']['layout_dets']
    # Original tables use rectangular get_crop_np_img; expanded formulas
    # explicitly pop polygon_points in _expand_formula_crop_res.
    for kind in ['ocr']+(['formulas']if formula_bbox_expand_px<=0 else[]):
        for task in plan[kind]:
            polygon=dets[task['layout_index']].get('polygon_points')
            if not polygon:continue
            u=task['useful_list'];px,py,x0,y0,x1,y1=map(int,u[:6])
            with Image.open(BytesIO(plan['images'][task['source']]))as image:rgb=np.asarray(image.convert('RGB')).copy()
            # crop_img applies the polygon to the crop before padding is pasted.
            points=np.asarray(polygon,dtype=np.int32).reshape((-1,1,2))-np.array([x0,y0])
            mask=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
            cv2.fillPoly(mask,[points],1)
            crop=rgb[py:py+(y1-y0),px:px+(x1-x0)]
            if crop.shape[:2]!=mask.shape:raise ValueError('Polygon crop dimensions mismatch')
            crop[~mask.astype(bool)]=255
            buffer=BytesIO();Image.fromarray(rgb).save(buffer,format='PNG')
            plan['images'][task['source']]=buffer.getvalue()
    return plan
