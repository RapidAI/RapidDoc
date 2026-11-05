"""8 连通域的 BBDT 顺序、行端点凸包、线中心及格框过滤合同。"""
import cv2,numpy as np
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_line_rec import get_table_line,_iter_connected_component_coords,min_area_rect_box_from_components
from common import ROOT,write_json,digest

def main():
    masks=[]
    for angle in [0,10,30,45,60,-20]:
        value=np.zeros((192,256),np.uint8)
        cv2.fillPoly(value,[cv2.boxPoints(((128,96),(130,25),angle)).astype(np.int32)],1)
        masks.append((f"rotated-{angle}",value))
    value=np.zeros((192,256),np.uint8);value[12,12:160]=1;value[30:170,180]=1
    for i in range(40):value[160-i,20+i]=1
    value[1,220]=1;masks.append(("thin",value))
    value=np.zeros((192,256),np.uint8)
    for x,y in [(130,2),(10,3),(70,3),(10,50),(70,51)]:value[y:y+22,x:x+22]=1
    masks.append(("block-order",value))
    value=np.ones((192,256),np.uint8)
    for y in [10,70,130,180]:cv2.line(value,(10,y),(245,y),0,2,cv2.LINE_AA)
    for x in [10,90,170,245]:cv2.line(value,(x,10),(x,180),0,2,cv2.LINE_AA)
    masks.append(("regions",value))
    rng=np.random.default_rng(315)
    for i in range(6):masks.append((f"noise-{i}",(rng.uniform(size=(72,96))>.97).astype(np.uint8)))
    masks.append(("blank",np.zeros((48,64),np.uint8)))
    cases=[]
    for name,value in masks:
        for axis in [None,0,1]:
            lines=[];polygons=[]
            if axis is None:polygons=min_area_rect_box_from_components(_iter_connected_component_coords(value),False,value.shape[1],value.shape[0],True,False)
            else:lines=get_table_line(value,axis=axis,lineW=10)
            l=np.asarray(lines,np.float32).reshape(-1,4);p=np.asarray(polygons,np.float32).reshape(-1,4,2)
            cases.append(dict(id=f"{name}-{axis}",width=value.shape[1],height=value.shape[0],mask=value.ravel().tolist(),axis=axis,line_extent=10,lines=l.tolist(),polygons=p.tolist(),sha256=digest(l.astype("<f4").tobytes()+p.astype("<f4").tobytes())))
    write_json(ROOT/"build/migration/table-components.json",dict(cases=cases))
    print(f"Exported {len(cases)} connected-component geometry contracts")

if __name__=="__main__":main()
