import time
from rapidocr import RapidOCR

from rapid_table import ModelType, RapidTable, RapidTableInput

ocr_engine = RapidOCR()
input_args = RapidTableInput(
    model_type=ModelType.UNITABLE
)

# 昇腾NPU
ocr_engine = RapidOCR()

table_engine = RapidTable(input_args)

img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"


ori_ocr_res = ocr_engine(img_path)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
start_time = time.time()
results = table_engine(img_path, ocr_results=ocr_results)
results.vis(save_dir="outputs-UNITABLE", save_name="vis")
print(f"总运行时间: {time.time() - start_time}秒")