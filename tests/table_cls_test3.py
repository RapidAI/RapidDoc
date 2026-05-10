from table_cls import TableCls
from table_cls.main import ModelType

table_cls = TableCls(model_type=ModelType.Q_CLS.value)

cls0, elasp0 = table_cls(r'table_18.png')
cls1, elasp1 = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png')
cls2, elasp2 = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_13.png')
cls3, elasp3 = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_17.png')
print(cls0, cls1, cls2, cls3)
# YOLO_CLS
# wireless wired wireless wired

# PADDLE_CLS
# wired wired wireless wired

# Q_CLS
# wired wired wired wireless
