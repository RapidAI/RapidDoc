from rapidocr import RapidOCR

from rapid_table import ModelType, RapidTable, RapidTableInput
from rapidocr import RapidOCR, EngineType, OCRVersion

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
ocr_engine = RapidOCR(params=default_params)

input_args = RapidTableInput(
    model_type=ModelType.SLANETPLUS,
)

table_engine = RapidTable(input_args)

img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"

ori_ocr_res = ocr_engine(img_path)
ocr_results = [[ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]]

results = table_engine([img_path], ocr_results=ocr_results)
results.vis(save_dir="outputs2", save_name="vis")