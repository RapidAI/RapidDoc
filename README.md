# KittyDoc – 高速文档解析产线

**基于 [Mineru](https://github.com/opendatalab/MinerU) 二次开发，移除 VLM，专注于 Pipeline 产线下的高效文档解析，在 CPU 上也能保持不错的解析速度。**

## 😺 项目介绍

KittyDoc 是一个轻量级、专注于文档解析的开源框架，支持 **OCR、版面分析、公式识别、表格识别和阅读顺序恢复** 等多种功能。  
与原框架相比，本项目使用 **PP-StructureV3 系列模型**，并完全 **去除对 Paddle 的依赖**，所有模型均已转换为 ONNX，可直接通过 **ONNX Runtime** 或 **OpenVINO**（部分模型）进行高效推理。

---

> ✨如果该项目对您有帮助，您的star是我不断优化的动力！！！
>
> - [github点击前往](https://github.com/hzkitty/KittyDoc)
> - [gitee点击前往](https://gitee.com/hzkitty/KittyDoc)

## 👏 项目特点

- **OCR 识别**
  - 使用 [RapidOCR](https://github.com/RapidAI/RapidOCR)
  - 支持多种推理引擎
  - CPU 上优先使用 OpenVINO（若可用且用户未指定 engine_type）
  
- **版面识别**
  - 自定义 [rapid_layout_self](kitty_doc%2Fmodel%2Flayout%2Frapid_layout_self)（参考RapidLayout）
  - 模型使用 `PP-DocLayout` 系列 ONNX 模型（plus-L、L、M、S）
    - **PP-DocLayout_plus-L**：速度稍慢 
    - **PP-DocLayout-L**：速度快，效果也不错，默认使用  
    - **PP-DocLayout-S**：速度极快，可能存在部分漏检，会自动调低阈值，conf_thresh=0.2
  - 模型下载：[KittyDoc 模型集](https://www.modelscope.cn/models/hzkitty/KittyDoc)

- **公式识别**
  - 自定义 [rapid_formula_self](kitty_doc%2Fmodel%2Fformula%2Frapid_formula_self)
  - 使用 `PP-FormulaNet_plus` 系列 ONNX 模型（L、M、S）
    - **PP-FormulaNet_plus-L**：速度慢  
    - **PP-FormulaNet_plus-S**：速度最快，默认使用  
  - 支持配置只识别行间公式
  - cuda环境默认不使用gpu，公式模型onnx gpu推理会报错，暂时无人解决 [PaddleOCR/issues/15125](https://github.com/PaddlePaddle/PaddleOCR/issues/15125), [PaddleX/issues/4238](https://github.com/PaddlePaddle/PaddleX/issues/4238), [Paddle2ONNX/issues/1593](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1593)

- **表格识别**
  - 基于 [rapid_table_self](kitty_doc%2Fmodel%2Ftable%2Frapid_table_self) 增强，在原有基础上增强为多模型串联方案：  
    - **表格分类**（区分有线/无线表格）
    - **SLANeXt** 系列 [表结构识别](https://www.paddleocr.ai/main/version3.x/module_usage/table_structure_recognition.html) + [单元格检测](https://www.paddleocr.ai/main/version3.x/module_usage/table_cells_detection.html)
    - **[有线表格识别UNET](https://github.com/RapidAI/TableStructureRec)** + SLANET_plus/UNITABLE（作为无线表格识别）
  - 相关模型下载：
    - [RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable)

- **阅读顺序恢复**
  - 移除 layoutreader 和 xycut
  - 使用 PP-StructureV3 阅读顺序 `xycut++` 算法简化 [xycut_plus.py](kitty_doc%2Fmodel%2Freading_order%2Fxycut_plus.py)
  - 速度快且阅读顺序恢复效果良好

- **推理方式**
  - 所有模型通过 ONNXRuntime 推理，OCR可配置其他推理引擎
  - 除了 OCR 和 PP-DocLayout-M/S 模型，OpenVINO推理会报错，暂时难以解决。[PaddleOCR/issues/16277](https://github.com/PaddlePaddle/PaddleOCR/issues/16277)
---

## 🛠️ 安装KittyDoc

#### 使用pip安装KittyDoc （暂未发布）
```bash
pip install kitty_doc -i https://mirrors.aliyun.com/pypi/simple
```

#### 通过源码安装KittyDoc
```bash
# 克隆仓库
git clone https://github.com/hzkitty/KittyDoc.git
cd KittyDoc

# 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```
#### 使用gpu推理
```bash
# 在安装完kitty_doc之后，卸载cpu版的onnxruntime
pip uninstall onnxruntime
# 这里一定要确定onnxruntime-gpu与GPU对应
# 可参见https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements
pip install onnxruntime-gpu
```

---

## 📋 使用示例

参考 [demo.py](demo%2Fdemo.py)

---

## 模型下载
不指定模型路径，初次运行时，会自动下载
- [KittyDoc 模型集（版面/公式/表格）](https://www.modelscope.cn/models/hzkitty/KittyDoc)  
- [RapidOCR 模型](https://www.modelscope.cn/models/RapidAI/RapidOCR)  
- [部分表格模型RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable) 

---

## TODO

- [x] 复选框识别，包括勾选/未勾选
- [x] 跨页表格合并
- [x] 表格非OCR文本提取
- [ ] 公式识别支持gpu
- [ ] 版面、表格、公式支持openvino
- [ ] KittyDoc4j（Java版本）


## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR & PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## ⚖️ 开源许可

基于 [MinerU](https://github.com/opendatalab/MinerU) 改造而来，已**移除原项目中的 YOLO 模型**，并替换为 **PP-StructureV3 系列 ONNX 模型**。  
由于已移除 AGPL 授权的 YOLO 模型部分，本项目整体不再受 AGPL 约束。

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。