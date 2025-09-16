# 参数使用

```python
from kitty_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze

def doc_analyze(
        pdf_bytes_list,
        lang_list,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,
)
```
在mineru参数基础上新增了layout_config、ocr_config、formula_config、table_config、checkbox_config参数
#### 1、使用gpu推理
```bash
# 在安装完kitty_doc之后，卸载cpu版的onnxruntime
pip uninstall onnxruntime
# 这里一定要确定onnxruntime-gpu与GPU对应
# 可参见https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
pip install onnxruntime-gpu
```
```python
# 在 Python 中指定 GPU（必须在导入 kitty_doc 之前设置）
import os
# 使用默认 GPU（cuda:0）
os.environ['MINERU_DEVICE_MODE'] = "cuda"
# 或指定 GPU 编号，例如使用第二块 GPU（cuda:1）
os.environ['MINERU_DEVICE_MODE'] = "cuda:1"
```

#### 2、layout_config 版面解析参数说明如下：

|  参数名   |  说明   |      默认值       | 备注 |
| :-------: |:-----:|:--------------:|:--:|
| model_type |  模型   | PP_DOCLAYOUT_L |  |
| conf_thresh  |  阈值   |  0.4（_S为0.2）   |  |
| batch_num | 批处理大小 |       1        |  |
| model_dir_or_path | 模型路径  |      None       |  |
示例：
```python
from kitty_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType

layout_config = {
    "model_type": LayoutModelType.PP_DOCLAYOUT_L,
    "conf_thresh": 0.4,
    "batch_num": 1,
    "model_dir_or_path": "C:\ocr\models\ppmodel\layout\PP-DocLayout-L\pp_doclayout_l.onnx"
}
```

#### 3、ocr_config OCR识别参数说明如下：
在rapidocr配置基础上新增如下参数

|  参数名   |      说明      |                      默认值                       | 备注 |
| :-------: |:------------:|:----------------------------------------------:|:--:|
| engine_type  | det和rec的推理引擎 | OPENVINO（cpu）、TORCH（gpu） |  |
| Det.rec_batch_num |   rec批处理大小   |                       1                        |  |
> [ocr_config想更深入了解，请移步rapidocr config.yaml参数解释](https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/)

示例：
```python
from rapidocr import EngineType as OcrEngineType, OCRVersion, ModelType

ocr_config = {
    # rapidocr 已有的参数
    "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.onnx",
    "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.onnx",
    "Rec.rec_batch_num": 1,

    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    "Det.model_type": ModelType.SERVER,
    "Rec.model_type": ModelType.SERVER,

    # 新增的自定义参数
    "engine_type": OcrEngineType.TORCH, # 统一设置推理引擎
    "Det.rec_batch_num": 1, # Det批处理大小
}
```

#### 4、formula_config 公式识别参数说明如下：

|  参数名   |  说明   |         默认值          | 备注 |
| :-------: |:-----:|:--------------------:|:--:|
| model_type |  模型   | PP_FORMULANET_PLUS_S |  |
| formula_level  |  公式识别等级   |          0           | 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别 |
| batch_num | 批处理大小 |          1           |  |
| model_dir_or_path | 模型路径  |         None         |  |
示例：
```python
from kitty_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType

formula_config = {
    "model_type": FormulaModelType.PP_FORMULANET_PLUS_S,
    "formula_level": 1, # 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别
    "batch_num": 1,
    "model_dir_or_path": r"C:\ocr\models\ppmodel\formula\PP-FormulaNet_plus-S\pp_formulanet_plus_s.onnx",
}
```

#### 5、table_config 表格识别参数说明如下：

|               参数名                |           说明           |         默认值          | 备注 |
|:--------------------------------:|:----------------------:|:--------------------:|:--:|
|            model_type            |           模型           | UNET_SLANET_PLUS | 有线表格使用unet，无线表格使用slanet_plus |
|        model_dir_or_path         |          模型地址          |          None           | 单个模型使用。如SLANET_PLUS、UNITABLE |
|      cls.model_dir_or_path       |        表格分类模型地址        |         None           |  |
|      unet.model_dir_or_path      |       UNET表格模型地址       |         None         |  |
|      unitable.model_dir_or_path      |     UNITABLE表格模型地址     |         None         |  |
|      slanet_plus.model_dir_or_path      |   SLANET_PLUS表格模型地址    |         None         |  |
|   wired_cell.model_dir_or_path   | 有线单元格模型地址 |         None         | 配置SLANEXT时使用 |
| wireless_cell.model_dir_or_path  | 无线单元格模型地址 |         None         | 配置SLANEXT时使用 |
|  wired_table.model_dir_or_path   |       有线表结构模型地址        |         None         | 配置SLANEXT时使用 |
| wireless_table.model_dir_or_path |       无线表结构模型地址        |         None         | 配置SLANEXT时使用 |
示例：
```python
from kitty_doc.model.table.rapid_table_self import ModelType as TableModelType

table_config = {
    "model_type": TableModelType.UNET_SLANET_PLUS,  # （默认） 有线表格使用unet，无线表格使用slanet_plus
    #"model_type": TableModelType.UNET_UNITABLE, # 有线表格使用unet，无线表格使用unitable
    #"model_type": TableModelType.SLANEXT,  # 有线表格使用slanext_wired，无线表格使用slanext_wireless

    "model_dir_or_path": "", #单个模型使用。如SLANET_PLUS、UNITABLE

    "cls.model_dir_or_path": "", # 表格分类模型地址

    "unet.model_dir_or_path": "", # UNET表格模型地址

    "unitable.model_dir_or_path": "", # UNITABLE表格模型地址
    "slanet_plus.model_dir_or_path": "", # SLANET_PLUS表格模型地址

    "wired_cell.model_dir_or_path": "", # 有线单元格模型地址，配置SLANEXT时使用
    "wireless_cell.model_dir_or_path": "", # 无线单元格模型地址，配置SLANEXT时使用
    "wired_table.model_dir_or_path": "", # 有线表结构模型地址，配置SLANEXT时使用
    "wireless_table.model_dir_or_path": "", # 无线表结构模型地址，配置SLANEXT时使用
}
```

#### 6、checkbox_config 复选框识别参数说明如下：

|  参数名   |   说明   |  默认值  | 备注 |
| :-------: |:------:|:-----:|:--:|
| checkbox_enable |  是否识别复选框  | False | 基于opencv，有可能会误检 |
示例：
```python
checkbox_config = {
    "checkbox_enable": True, # 是否识别复选框，默认不识别，基于opencv，有可能会误检
}
```
