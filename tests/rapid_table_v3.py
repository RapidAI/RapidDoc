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
    "Det.box_thresh": 0.5,
    "Det.use_dilation": True,
    "Det.unclip_ratio": 1.6,
}
ocr_engine = RapidOCR(params=default_params)

input_args = RapidTableInput(
    model_type=ModelType.UNET,
    engine_type=TableEngineType.OPENVINO
)

table_engine = RapidTable(input_args)

img_paths = [
    "2a18af309c8ea0e9419ab8ea69d24868ef86288da9d7d57de6e19a769c2d2630.jpg",
    ]
ocr_results_list = []

for img in img_paths:
    ori_ocr_res = ocr_engine(img, return_word_box=True)
    ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores, ori_ocr_res.word_results]
    ocr_result = []
    for word_results in ori_ocr_res.word_results:
        ocr_result.extend([[word_result[2], word_result[0], word_result[1]] for word_result in word_results])
    ocr_result = [list(x) for x in zip(*ocr_result)]
    ocr_results_list.append(ocr_result)


results = table_engine(img_paths, ocr_results=ocr_results_list, batch_size=4)

results.vis(save_dir=r"outputs3", save_name="vis")