import time

# from rapid_table import RapidTable
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType as TableEngineType

from rapidocr import RapidOCR, EngineType, OCRVersion

from rapid_doc.model.table.rapid_table_self.table_cls import TableCls

# 默认配置
default_params = {
    "Global.use_cls": False,
    "Det.engine_type": EngineType.OPENVINO,
    "Rec.engine_type": EngineType.OPENVINO,
    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    # "Det.model_type": ModelType.SERVER,
    # "Rec.model_type": ModelType.SERVER,
    "Det.limit_side_len": 960,
    "Det.limit_type": 'max',
    "Det.std": [0.229, 0.224, 0.225],
    "Det.mean": [0.485, 0.456, 0.406],
    "Det.box_thresh": 0.3,
    "Det.use_dilation": True,
    "Det.unclip_ratio": 1.8,
}

img_path = r"https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"

ocr_engine = RapidOCR(params=default_params)

input_args = RapidTableInput(
    model_type=ModelType.UNET,
    # engine_type=TableEngineType.OPENVINO
)
table_engine = RapidTable(input_args)



ori_ocr_res = ocr_engine(img_path)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
start_time = time.time()
results = table_engine(img_path, ocr_results=ocr_results)
results.vis(save_dir="outputs-SLANETPLUS", save_name="vis")
print(f"总运行时间: {time.time() - start_time}秒")


# table_cls = TableCls()
#
# start_time = time.time()
# cls, elasp = table_cls(img_path)
# print(f"总运行时间: {time.time() - start_time}秒")
# print(cls)