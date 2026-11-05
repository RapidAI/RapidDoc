"""固定 UNET 有序补线、交点延长和 FP32 写回的几何合同。"""
import numpy as np
from rapid_doc.model.table.rapid_table_self.table_structure.unet.utils.utils_table_line_rec import adjust_lines,final_adjust_lines
from common import ROOT,write_json,digest

def main():
    rng=np.random.default_rng(116);cases=[]
    source=[("empty",[],[]),
        ("cross-extension",[[10,50,90,50]],[[5,10,5,90],[95,10,95,90]]),
        ("parallel",[[0,0,50,0],[0,10,50,10]],[[70,0,70,100]]),
        ("gaps",[[0,10,30,10],[40,10,80,10],[90,10,120,10]],[[0,0,0,100],[60,0,60,100]]),
        ("reversed",[[90,10,0,10],[10,0,10,80]],[[0,20,100,20]]),
        ("strict-20",[[20,0,100,0]],[[0,-20,0,20]]),
        ("degenerate",[[1,1,1,1]],[[0,0,0,10]]),
        ("sloped",[[10,50,90,55]],[[5,10,2,90],[95,10,98,90]])]
    for index in range(20):
        rows=[];columns=[]
        for _ in range(6):
            x,y=rng.uniform(0,200,2);rows.append([x,y,x+rng.uniform(20,90),y+rng.uniform(-4,4)])
            x,y=rng.uniform(0,200,2);columns.append([x,y,x+rng.uniform(-4,4),y+rng.uniform(20,90)])
        source.append((f"random-{index}",rows,columns))
    for name,rows,columns in source:
        for more,extend in [(True,True),(False,True),(True,False)]:
            r=list(np.asarray(rows,np.float32).reshape(-1,4));c=list(np.asarray(columns,np.float32).reshape(-1,4))
            if more:r+=adjust_lines(r,alph=100,angle=50);c+=adjust_lines(c,alph=15,angle=50)
            if extend:r,c=final_adjust_lines(r,c)
            outputs={"rows":np.asarray(r,np.float32).reshape(-1,4),"columns":np.asarray(c,np.float32).reshape(-1,4)}
            cases.append(dict(id=f"{name}-{more}-{extend}",rows=np.asarray(rows,np.float32).reshape(-1,4).tolist(),columns=np.asarray(columns,np.float32).reshape(-1,4).tolist(),config=dict(more_rows=more,more_columns=more,extend=extend),expected={name:dict(values=value.tolist(),sha256=digest(value.astype("<f4").tobytes()))for name,value in outputs.items()}))
    write_json(ROOT/"build/migration/table-lines.json",dict(cases=cases))
    print(f"Exported {len(cases)} table line connection contracts")

if __name__=="__main__":main()
