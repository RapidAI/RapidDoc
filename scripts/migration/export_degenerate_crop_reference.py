"""Freeze actual Python/OpenCV crop output for collapsed and nonconvex boxes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
from rapid_doc.utils.ocr_utils import get_rotate_crop_image

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = np.array([(i * 37 + i // 13) % 256 for i in range(64*48*3)],np.uint8).reshape(48,64,3)
    cases = []
    for name, points in [
        ('collapsed_width',[[20,12],[20,12],[20,42],[20,42]]),
        ('collapsed_height',[[10,12],[40,12],[40,12],[10,12]]),
        ('collapsed_all',[[10,12]]*4),
        ('collinear',[[10,10],[20,20],[30,30],[40,40]]),
        ('crossed',[[4,4],[44,32],[44,4],[4,32]]),
        ('concave',[[4,4],[44,4],[15,15],[4,32]]),
        ('pole',[[2,2],[42,2],[2,32],[42,32]]),
        ('convex',[[4,4],[44,4],[44,32],[4,32]]),
    ]:
        crop = get_rotate_crop_image(source,np.array(points,np.float32))
        cases.append(dict(name=name,points=points,width=crop.shape[1],height=crop.shape[0],
                          sha256=hashlib.sha256(crop.tobytes()).hexdigest()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(opencv=cv2.__version__,cases=cases),indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    main()
