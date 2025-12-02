from table_cls import TableCls
from table_cls.main import ModelType

table_cls = TableCls(model_type=ModelType.Q_CLS.value)

cls1, elasp1 = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png')
cls, elasp = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png')
print(cls, elasp)
