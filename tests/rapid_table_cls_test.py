import time

# from rapid_table import RapidTable
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType as TableEngineType

from rapidocr import RapidOCR, EngineType, OCRVersion

from rapid_doc.model.table.rapid_table_self.table_cls import TableCls

input_args = RapidTableInput(
    model_type=ModelType.Q_CLS,
    engine_type=TableEngineType.ONNXRUNTIME
)

img_paths = [
    # r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_12.png',
    # r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png',
    # "787fff5a0b10f78474fd00802d346c7ad08e7e728d8254d73a456ace44b91e71.jpg",
    r"D:\CodeProjects\doc\RapidAI\RapidDoc\output3\比亚迪财报\auto\images\2a18af309c8ea0e9419ab8ea69d24868ef86288da9d7d57de6e19a769c2d2630.jpg"
]


table_cls = TableCls(input_args)

start_time = time.time()
cls, elasp = table_cls(img_paths, batch_size=4)
print(f"总运行时间: {time.time() - start_time}秒")
print(cls)