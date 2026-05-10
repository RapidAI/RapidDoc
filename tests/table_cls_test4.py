# from table_cls import TableCls
# from table_cls.main import ModelType
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTableInput
from rapid_doc.model.table.rapid_table_self.table_cls import TableCls

table_cls = TableCls(RapidTableInput(model_type=ModelType.PADDLE_Q_CLS,
                                     # model_dir_or_path={
                                     #     'paddle_cls': r'D:\CodeProjects\doc\RapidAI\models\paddle_cls.onnx',
                                     #     'q_cls': r'D:\CodeProjects\doc\RapidAI\models\q_cls.onnx'
                                     # }
                                     ))

cls, elasp = table_cls([r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png',
                        r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_12.png',
                          r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_13.png',
                          r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_17.png',
                        ])
# cls, elasp = table_cls([r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_12.png',])
# cls, elasp = table_cls(r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\images\table_11.png')
print(cls, elasp)
