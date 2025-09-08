import time

from kitty_doc.model.table.rapid_table_self.table_cls import TableCls
from kitty_doc.model.layout.rapid_layout_self import RapidLayoutInput, RapidLayout, ModelType as LayoutModelType
from kitty_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput
from rapidocr import RapidOCR
from rapidocr import EngineType as OcrEngineType, OCRVersion, ModelType as OCRModelType

ocr_config = {
    # "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.xml",
    # "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.xml",
    # "Rec.rec_batch_num": 1,

    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    # "Det.model_type": OCRModelType.SERVER,
    # "Rec.model_type": OCRModelType.SERVER,
    # 新增自定义
    # "engine_type": OcrEngineType.ONNXRUNTIME,
    # "Det.rec_batch_num": 1,
}
ocr_engine = RapidOCR(params=ocr_config)
# ocr_engine = RapidOCR()

table_cls = TableCls()
# img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
# img_path = r'D:\CodeProjects\doc\KittyDoc\github\KittyDoc\output_plus_l\88b41949e122304fe0b98f45c08aaf14\auto\images\122bbdac0e543c5afc6ab7b04396897ceb4cf42113c0ce05b209d25d7a85f316.jpg'
# img_path = r'C:\ocr\img\309d3419-570c-4398-b975-7d5c4231a601.png'

# img_path = r'D:\CodeProjects\doc\paddleocr-v3\doc\output9\imgs\img_in_table_box_215_99_1192_940.jpg'
# img_path = r"D:\CodeProjects\doc\paddleocr-v3\doc\output93\imgs\img_in_table_box_423_194_2394_1881.jpg"

# img_path = r'D:\CodeProjects\doc\KittyDoc\github\KittyDoc\output_plus_96\img_in_table_box_423_194_2394_1881\auto\images\97187f17b72b1b04600b09d45a412ab07ad971726a07ad441765e817f7c10ef2.jpg'

# img_path = "img_in_table_box_0_0_2530_2179.jpg"
img_path = 'afaba6c4d454b8f1ba193c66efa1761f.png'
# img_path = f"39195dfa12189c38e9a1f53049dce2384dc4b40581274154f8fce4da8825fb89.jpg"

cls, elasp = table_cls(img_path)
start_time = time.time()

print(cls, elasp)
if cls == "wired":
    cfg = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRED_TABLE_CELL_DET, conf_thresh=0.4)
    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRED)
else:
    cfg = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRELESS_TABLE_CELL_DET, conf_thresh=0.4)
    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRELESS)
layout_engine = RapidLayout(cfg=cfg)
#
# input_args = RapidTableInput(model_type=ModelType.UNET)


table_engine = RapidTable(input_args)


ori_ocr_res = ocr_engine(img_path)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]

# start_time = time.time()
cell_res = layout_engine([img_path])
cell_results = (cell_res[0].boxes, cell_res[0].scores)
results = table_engine(img_path, ocr_results=ocr_results, cell_results=cell_results)
# results = table_engine(img_path, ocr_results=ocr_results)
print(f"总运行时间: {time.time() - start_time}秒")
# print(results)
results.vis(save_dir="outputs_wireless_cells542_vb", save_name="vis")
# cell_res[0].vis("outputs_wireless_cells53_vb/cell_vis.png")
