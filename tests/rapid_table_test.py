import time
from rapidocr import RapidOCR

from kitty_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput

ocr_engine = RapidOCR()
input_args = RapidTableInput(
    model_type=ModelType.SLANETPLUS
)

ocr_engine = RapidOCR()

table_engine = RapidTable(input_args)

# img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
img_path = r'D:\CodeProjects\doc\KittyDoc\github\KittyDoc\output_plus_l\88b41949e122304fe0b98f45c08aaf14\auto\images\122bbdac0e543c5afc6ab7b04396897ceb4cf42113c0ce05b209d25d7a85f316.jpg'

ori_ocr_res = ocr_engine(img_path)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
start_time = time.time()
results = table_engine(img_path, ocr_results=ocr_results)
results.vis(save_dir="outputs-SLANETPLUS", save_name="vis")
print(f"总运行时间: {time.time() - start_time}秒")