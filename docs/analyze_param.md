# 参数使用

```python
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze


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
# 在安装完 rapid_doc 之后，卸载 cpu 版的 onnxruntime
pip uninstall onnxruntime
# 这里一定要确定onnxruntime-gpu与GPU对应
# 可参见https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
pip install onnxruntime-gpu
```
```python
# 在 Python 中指定 GPU（必须在导入 rapid_doc 之前设置）
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
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType

layout_config = {
    "model_type": LayoutModelType.PP_DOCLAYOUT_L,
    "conf_thresh": 0.4,
    "batch_num": 1,
    "model_dir_or_path": "C:\ocr\models\ppmodel\layout\PP-DocLayout-L\pp_doclayout_l.onnx"
}
```

#### 3、ocr_config OCR识别参数说明如下：
在rapidocr配置基础上新增如下参数

|  参数名   |      说明      |                      默认值                       |                                             备注                                              |
| :-------: |:------------:|:----------------------------------------------:|:-------------------------------------------------------------------------------------------:|
| engine_type  | det和rec的推理引擎 | OPENVINO（cpu）、TORCH（gpu） |                                                                                             |
| use_det_mode  | 文本检测框模式：auto（默认）、txt、ocr | auto | 1、txt只会从pypdfium2获取文本框<br/>2、ocr只会从OCR-det获取文本框<br/>3、auto先从pypdfium2获取文本框，提取不到再使用OCR-det提取 |
| Det.rec_batch_num |   rec批处理大小   |                       1                        |                                                                                             |
> [ocr_config想更深入了解，请移步rapidocr config.yaml参数解释](https://rapidai.github.io/RapidOCRDocs/install_usage/api/RapidOCR/)

示例：
```python
from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType

ocr_config = {
    # "Det.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_det_infer\openvino\ch_PP-OCRv4_det_infer.onnx",
    # "Rec.model_path": r"C:\ocr\models\ppmodel\ocr\v4\ch_PP-OCRv4_rec_infer\openvino\ch_PP-OCRv4_rec_infer.onnx",
    # "Rec.rec_batch_num": 1,

    # "Det.ocr_version": OCRVersion.PPOCRV5,
    # "Rec.ocr_version": OCRVersion.PPOCRV5,
    # "Det.model_type": OCRModelType.SERVER,
    # "Rec.model_type": OCRModelType.SERVER,

    # 新增的自定义参数
    # "engine_type": OCREngineType.TORCH, # 统一设置推理引擎
    # "Det.rec_batch_num": 1, # Det批处理大小

    # "use_det_mode": 'auto' # 文本检测框模式：auto（默认）、txt、ocr
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
from rapid_doc.model.formula.rapid_formula_self import ModelType as FormulaModelType

formula_config = {
    "model_type": FormulaModelType.PP_FORMULANET_PLUS_S,
    "formula_level": 1,  # 公式识别等级，默认为0，全识别。1:仅识别行间公式，行内公式不识别
    "batch_num": 1,
    "model_dir_or_path": r"C:\ocr\models\ppmodel\formula\PP-FormulaNet_plus-S\pp_formulanet_plus_s.onnx",
}
```

#### 5、table_config 表格识别参数说明如下：

|                  参数名                   |                说明                |         默认值          |                   备注                    |
|:--------------------------------------:|:--------------------------------:|:--------------------:|:---------------------------------------:|
|               force_ocr                |          表格文字是否强制使用ocr           | False | 根据 parse_method 来判断是否需要ocr还是从pdf中直接提取文本 |
|         skip_text_in_image             |   是否跳过表格里图片中的文字                  | True |       如表格单元格中嵌入的图片、图标、扫描底图等，里面的文字       |
|             use_img2table              |       是否优先使用img2table库提取表格       | False |       需要手动安装（pip install img2table），基于opencv识别准确度不如使用模型，但是速度很快，默认关闭       |
|               model_type               |                模型                | UNET_SLANET_PLUS |      有线表格使用unet，无线表格使用slanet_plus       |
|           model_dir_or_path            |               模型地址               |          None           |      单个模型使用。如SLANET_PLUS、UNITABLE       |
|         cls.model_dir_or_path          |             表格分类模型地址             |         None           |                                         |
|         unet.model_dir_or_path         |            UNET表格模型地址            |         None         |                                         |
|       unitable.model_dir_or_path       |          UNITABLE表格模型地址          |         None         |                                         |
|     slanet_plus.model_dir_or_path      |        SLANET_PLUS表格模型地址         |         None         |                                         |
|      wired_cell.model_dir_or_path      |            有线单元格模型地址             |         None         |              配置SLANEXT时使用               |
|    wireless_cell.model_dir_or_path     |            无线单元格模型地址             |         None         |              配置SLANEXT时使用               |
|     wired_table.model_dir_or_path      |            有线表结构模型地址             |         None         |              配置SLANEXT时使用               |
|    wireless_table.model_dir_or_path    |            无线表结构模型地址             |         None         |              配置SLANEXT时使用               |
示例：

```python
from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

table_config = {
    "force_ocr": False,  # 表格文字，是否强制使用ocr，默认 False 根据 parse_method 来判断是否需要ocr还是从pdf中直接提取文本
    "skip_text_in_image": True, # 是否跳过表格里图片中的文字（如表格单元格中嵌入的图片、图标、扫描底图等）
    "use_img2table": False, # 是否优先使用img2table库提取表格，需要手动安装（pip install img2table），基于opencv识别准确度不如使用模型，但是速度很快，默认关闭
    "model_type": TableModelType.UNET_SLANET_PLUS,  # （默认） 有线表格使用unet，无线表格使用slanet_plus
    # "model_type": TableModelType.UNET_UNITABLE, # 有线表格使用unet，无线表格使用unitable
    # "model_type": TableModelType.SLANEXT,  # 有线表格使用slanext_wired，无线表格使用slanext_wireless

    "model_dir_or_path": "",  # 单个模型使用。如SLANET_PLUS、UNITABLE

    "cls.model_dir_or_path": "",  # 表格分类模型地址

    "unet.model_dir_or_path": "",  # UNET表格模型地址

    "unitable.model_dir_or_path": "",  # UNITABLE表格模型地址
    "slanet_plus.model_dir_or_path": "",  # SLANET_PLUS表格模型地址

    "wired_cell.model_dir_or_path": "",  # 有线单元格模型地址，配置SLANEXT时使用
    "wireless_cell.model_dir_or_path": "",  # 无线单元格模型地址，配置SLANEXT时使用
    "wired_table.model_dir_or_path": "",  # 有线表结构模型地址，配置SLANEXT时使用
    "wireless_table.model_dir_or_path": "",  # 无线表结构模型地址，配置SLANEXT时使用
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

#### 7、image_config 图片提取参数说明如下：

|  参数名   |   说明   |  默认值  |                    备注                     |
| :-------: |:------:|:-----:|:-----------------------------------------:|
| extract_original_image |  是否提取原始图片  | False | 使用 pypdfium2 提取原始图片。截图可能导致清晰度降低和边界丢失，默认关闭 |
| extract_original_image_iou_thresh |  原始图片和版面识别的图片，bbox重叠度  |  0.9  |  |
示例：
```python
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
image_config = {
    "extract_original_image": True, # 是否提取原始图片（使用 pypdfium2 提取原始图片。截图可能导致清晰度降低和边界丢失，默认关闭）
    "extract_original_image_iou_thresh": 0.5, # 原始图片和版面识别的图片，bbox重叠度，默认0.9
}
middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable, p_formula_enable, ocr_config=ocr_config, image_config=image_config)
```