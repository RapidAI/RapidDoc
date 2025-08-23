import time

from kitty_doc.model.table.rapid_table_self.table_cls import TableCls
from layout.rapid_layout_self import RapidLayoutInput, RapidLayout, ModelType as LayoutModelType
from table.rapid_table_self import ModelType, RapidTable, RapidTableInput
from rapidocr import RapidOCR

ocr_engine = RapidOCR()

table_cls = TableCls()
img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
# img_path = "table_recognition.png"
cls, elasp = table_cls(img_path)
print(cls, elasp)
if cls == "wired":
    cfg = RapidLayoutInput(model_type= LayoutModelType.RT_DETR_L_WIRED_TABLE_CELL_DET, conf_thresh=0.4,
                       model_dir_or_path= r"C:\ocr\models\ppmodel\table\RT-DETR-L_wired_table_cell_det\rt_detr_l_wired_table_cell_det.onnx")

    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRED,
                                 model_dir_or_path=r"C:\ocr\models\ppmodel\table\SLANeXt_wired\slanext_wired.onnx")
else:
    cfg = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRELESS_TABLE_CELL_DET, conf_thresh=0.4,
                           model_dir_or_path=r"C:\ocr\models\ppmodel\table\RT-DETR-L_wireless_table_cell_det\rt_detr_l_wireless_table_cell_det.onnx")
    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRELESS,
                                 model_dir_or_path=r"C:\ocr\models\ppmodel\table\SLANeXt_wireless\slanext_wireless.onnx")
layout_engine = RapidLayout(cfg=cfg)

input_args = RapidTableInput(model_type=ModelType.SLANETPLUS)


table_engine = RapidTable(input_args)


ori_ocr_res = ocr_engine(img_path)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]

start_time = time.time()
cell_res = layout_engine([img_path])

results = table_engine(img_path, ocr_results=ocr_results, cell_results=cell_res[0].boxes)
# results = table_engine(img_path, ocr_results=ocr_results)
print(f"总运行时间: {time.time() - start_time}秒")
# print(results)
results.vis(save_dir="outputs_wireless_tan113", save_name="vis")
