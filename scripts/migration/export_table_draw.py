"""固定 UNET thickness=2 / LINE_AA 的完整灰度像素，不仅比较二值框。"""
import numpy as np
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_line_rec import draw_lines
from common import ROOT,write_json,digest

def main():
    cases=[];rng=np.random.default_rng(443)
    groups=[("horizontal",[[10,20,80,20]]),("vertical",[[30,10,30,80]]),("diagonal",[[10,10,80,60]]),("reverse",[[80,60,10,10]]),("point",[[40,30,40,30]]),("edge",[[-20,-8,140,70],[80,-10,20,110]]),("outside",[[-10,-10,-5,-5]]),("short",[[10,10,11,11],[20,20,21,20],[30,30,30,31]]),("cross",[[0,40,127,40],[60,0,60,95],[0,0,127,95]])]
    for i in range(15):groups.append((f"random-{i}",rng.uniform(-30,160,(10,4)).astype(np.float32).tolist()))
    directory=ROOT/"build/migration/table-draw";directory.mkdir(parents=True,exist_ok=True)
    for name,lines in groups:
        lines=np.array(lines,np.float32);image=draw_lines(np.zeros((96,128),np.uint8),lines,255,2)
        path=directory/f"{name}.gray";path.write_bytes(image.tobytes())
        cases.append(dict(id=name,width=128,height=96,lines=lines.tolist(),expected=str(path),sha256=digest(image.tobytes())))
    write_json(ROOT/"build/migration/table-draw.json",dict(cases=cases))
    print(f"Exported {len(cases)} exact antialiased line raster contracts")

if __name__=="__main__":main()
