import time

# from rapid_table import RapidTable
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType as TableEngineType

from rapidocr import RapidOCR, EngineType, OCRVersion

from rapid_doc.model.table.rapid_table_self.table_cls import TableCls

input_args = RapidTableInput(
    engine_type=TableEngineType.OPENVINO
)

img_paths = [
    "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg",
    ]


table_cls = TableCls(input_args)

start_time = time.time()
cls, elasp = table_cls(img_paths, batch_size=4)
print(f"总运行时间: {time.time() - start_time}秒")
print(cls)