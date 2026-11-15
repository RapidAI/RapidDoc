"""Freeze model objects and OCR for native TATR postprocessor parity."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from loguru import logger
from rapid_doc.model.table.rapid_table import RapidTableModel
from rapid_doc.model.table.rapid_table_self.table_structure.table_transformer import main as tatr

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("build/migration/table-evidence-sync.json"))
    parser.add_argument("--output",type=Path,default=Path("build/migration/table-transformer-sync.json"))
    args=parser.parse_args()
    logger.remove()
    model=RapidTableModel(None,{"model_type":"table_transformer","engine_cfg":{"intra_op_num_threads":4}})
    result=[]
    original=tatr.outputs_to_objects
    current=[]
    def capture(*a):
        objects=original(*a)
        current.extend(objects)
        return objects
    with patch.object(tatr,"outputs_to_objects",capture):
        for item in json.loads(args.input.read_text(encoding="utf8")):
            current.clear()
            image=cv2.imread(item["image"])
            tokens=tatr.ocr_to_tokens(item["rows"])
            model.table_model.table_structure([image],[item["rows"]],token_cells=[])
            raw_objects=[dict(label=tatr.LABELS.index(o["label"]),score=o["score"],bbox=o["bbox"]) for o in current]
            captured=[]
            build=tatr.cells_to_result
            def capture_cells(cells,*a):
                captured.extend(cells)
                return build(cells,*a)
            with patch.object(tatr,"cells_to_result",capture_cells):
                # Reuse the frozen raw objects, avoid another model call.
                with patch.object(tatr,"outputs_to_objects",return_value=list(current)):
                    tatr.decode(None,None,(image.shape[1],image.shape[0]),tokens)
            cells=[dict(bbox=c["bbox"],row_nums=c["row_nums"],column_nums=c["column_nums"],
                        column_header=c["column header"],text=c["cell text"],
                        token_indices=[s["line_num"] for s in c["spans"]]) for c in captured]
            result.append(dict(id=item["id"],objects=raw_objects,tokens=tokens,cells=cells))
            print(item["id"],len(cells),flush=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()),encoding="utf8")

if __name__=="__main__":
    main()
