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
  - 使用 [RapidOCR](https://www.modelscope.cn/models/RapidAI/RapidOCR)
  - 支持多种推理引擎
  - CPU 上优先使用 OpenVINO（若可用且用户未指定 engine_type）
  
- **版面识别**
  - 自定义 [rapid_layout_self](kitty_doc%2Fmodel%2Flayout%2Frapid_layout_self)（参考RapidLayout）
  - 模型使用 `PP-DocLayout` 系列 ONNX 模型（plus-L、L、M、S）
    - **PP-DocLayout_plus-L**：效果最好，速度稍慢 
    - **PP-DocLayout-L**：速度快，效果也不错，默认使用  
    - **PP-DocLayout-S**：速度极快，可能存在部分漏检，会自动调低阈值，conf_thresh=0.2
  - 模型下载：[KittyDoc 模型集](https://www.modelscope.cn/models/hzkitty/KittyDoc)

- **公式识别**
  - 自定义 [rapid_formula_self](kitty_doc%2Fmodel%2Fformula%2Frapid_formula_self)
  - 使用 `PP-FormulaNet_plus` 系列 ONNX 模型（L、M、S）
    - **PP-FormulaNet_plus-L**：速度慢  
    - **PP-FormulaNet_plus-S**：速度最快，默认使用  
  - 支持配置只识别行间公式

- **表格识别**
  - 基于 [rapid_table_self](kitty_doc%2Fmodel%2Ftable%2Frapid_table_self) 增强，在原有基础上增强为多模型串联方案：  
    1. **表格分类**  
    2. **表格结构识别**：支持 `SLANeXt_wired` 和 `SLANeXt_wireless` 模型  
       - 注意：SLANeXt 系列模型预测的表格单元格信息无效，因此需要单元格检测模型配合使用  
    3. **单元格检测**：支持 `RT-DETR-L_wired_table_cell_det` 和 `RT-DETR-L_wireless_table_cell_det` 模型  
       - 单元格检测也可以直接作用于 `SLANeXt_plus` 模型进行增强 
  - 部分模型下载：
    - [RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable)

- **阅读顺序恢复**
  - 移除 layoutreader 和 xycut
  - 使用 PP-StructureV3 阅读顺序 `xycut++` 算法简化 [xycut_plus.py](kitty_doc%2Fmodel%2Freading_order%2Fxycut_plus.py)
  - 速度快且阅读顺序恢复效果良好

- **推理方式**
  - 所有模型通过 ONNXRuntime 推理，OCR可配置其他推理引擎
  - 除了 OCR 和 PP-DocLayout-M/S 模型，OpenVINO推理会报错，可能是opset_version ≥16，OpenVINO 部分算子不支持。
---

## 🛠️ 安装

```bash
# 克隆仓库
git clone https://github.com/hzkitty/KittyDoc.git
cd KittyDoc

# 安装依赖
pip install -r requirements.txt
```

---

## 📋 使用示例

参考 [demo.py](demo%2Fdemo.py)

---

## 模型下载

- [KittyDoc 模型集（版面/公式/表格）](https://www.modelscope.cn/models/hzkitty/KittyDoc)  
- [RapidOCR 模型](https://www.modelscope.cn/models/RapidAI/RapidOCR)  
- [部分表格模型RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable) 

---

## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR & PP-StructureV3](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## ⚖️ 开源许可

基于 [MinerU](https://github.com/opendatalab/MinerU) 改造而来，已**移除原项目中的 YOLO 模型**，并替换为 **PP-StructureV3 系列 ONNX 模型**。  
由于已移除 AGPL 授权的 YOLO 模型部分，本项目整体不再受 AGPL 约束。

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。