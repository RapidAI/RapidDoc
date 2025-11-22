from rapidocr import RapidOCR

# from rapid_table import ModelType, RapidTable, RapidTableInput
from rapidocr import RapidOCR, EngineType, OCRVersion

from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType as TableEngineType

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
ocr_engine = RapidOCR()

input_args = RapidTableInput(
    model_type=ModelType.UNET,
    engine_type=TableEngineType.OPENVINO
)

table_engine = RapidTable(input_args)

img_paths = [
    "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg",
    # 'table_05.png'
    ]
ocr_results_list = []
for img in img_paths:
    ori_ocr_res = ocr_engine(img)
    ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
    ocr_results_list.append(ocr_results)

results = table_engine(img_paths, ocr_results=ocr_results_list, batch_size=4)

indexes = list(range(len(img_paths)))
results.vis(save_dir="outputs2", save_name="vis", indexes=indexes)