# RapidDoc - 高速文档解析系统

[English](README_EN.md) | [中文](README.md)

## 😺 项目介绍

**RapidDoc 是一个轻量级、专注于文档解析的开源框架，支持 **OCR、版面分析、公式识别、表格识别和阅读顺序恢复** 等多种功能，支持将复杂 PDF 文档转换为 Markdown、JSON、WORD、HTML 格式。**

**支持docx/doc、pptx/ppt、xlsx/xls的原生解析（不使用模型）。**

**框架基于 [Mineru](https://github.com/opendatalab/MinerU) 二次开发，移除 VLM，专注于 Pipeline 产线下的高效文档解析，在 CPU 上也能保持不错的解析速度。**

**本项目所使用的默认模型主要来源于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的 [PP-StructureV3](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PP-StructureV3.html) 系列（OCR、版面分析、公式识别、阅读顺序恢复，以及部分表格识别模型），并已全部转换为 ONNX 格式，支持在 CPU/GPU 上高效推理。**

**同时支持自定义OCR、公式、表格模型，需实现 CustomBaseModel 的 batch_predict 方法，目前内置 [PaddleOCRVL](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html) 系列模型的集成。**

**KittyDoc 已经成为 RapidAI 开源家族成员**

---

> ✨如果该项目对您有帮助，您的star是我不断优化的动力！！！
>
> - [github点击前往](https://github.com/RapidAI/RapidDoc)
> - [gitee点击前往](https://gitee.com/hzkitty/KittyDoc)

## 👏 项目特点

- **OCR 识别**
  - 使用 [RapidOCR](https://github.com/RapidAI/RapidOCR) 支持多种推理引擎
  - CPU 下默认使用 OpenVINO（速度快，内存占用较高），GPU 下默认使用 torch
  
- **版面识别**
  - 模型使用 `PP-DocLayout` 系列 ONNX 模型（v2、plus-L、L、M、S）
    - **PP-DocLayoutV3**：自带阅读顺序，支持异形框，默认使用
    - **PP-DocLayoutV2**：自带阅读顺序
    - **PP-DocLayout_plus-L**：效果好运行稳定
    - **PP-DocLayout-L**：速度快，效果也不错
    - **PP-DocLayout-S**：速度极快，存在部分漏检

- **公式识别**
  - 使用 `PP-FormulaNet_plus` 系列 ONNX 模型（L、M、S）
    - **PP-FormulaNet_plus-L**：速度慢，支持onnx  
    - **PP-FormulaNet_plus-M**：默认使用，支持onnx和torch    
    - **PP-FormulaNet_plus-S**：速度最快，支持onnx，复杂公式精度不够
  - 支持配置只识别行间公式
  - cuda环境，默认使用torch推理，公式模型onnx gpu推理会报错，暂时无人解决 [PaddleOCR/issues/15125](https://github.com/PaddlePaddle/PaddleOCR/issues/15125), [PaddleX/issues/4238](https://github.com/PaddlePaddle/PaddleX/issues/4238), [Paddle2ONNX/issues/1593](https://github.com/PaddlePaddle/Paddle2ONNX/issues/1593)

- **表格识别**
  - 基于 [rapid_table_self](rapid_doc/model/table/rapid_table_self) 增强，在原有基础上增强为多模型串联方案：  
    - **表格分类**（区分有线/无线表格）
    - **[有线表格识别UNET](https://github.com/RapidAI/TableStructureRec)** + SLANET_plus/UNITABLE（作为无线表格识别）

- **阅读顺序恢复**
  - PP-DocLayoutV2和PP-DocLayoutV3使用版面模型自带的阅读顺序
  - 其余版面模型，使用 PP-StructureV3 阅读顺序恢复算法，基于xycut算法和版面的结果

- **推理方式**
  - 所有模型通过 ONNXRuntime 推理，OCR可配置其他推理引擎
  - 除了 OCR 和 PP-DocLayout-M/S 模型，OpenVINO推理会报错，暂时难以解决。[PaddleOCR/issues/16277](https://github.com/PaddlePaddle/PaddleOCR/issues/16277)
---

## 基准测试结果

### 1. OmniDocBench

以下是RapidDoc在 OmniDocBench v1.6 上的评估结果。

Pipeline 模型使用 PP-DocLayoutV3、PP-OCRv6-small、PP-FormulaNet_plus-M、UNET_SLANET_PLUS。
<table style="width:100%; border-collapse: collapse;">
    <caption>Comprehensive evaluation of document parsing on OmniDocBench (v1.6_full)</caption>
    <thead>
        <tr>
            <th>Model Type</th>
            <th>Methods</th>
            <th>Size</th>
            <th>Overall&#x2191;</th>
            <th>Text<sup>Edit</sup>&#x2193;</th>
            <th>Formula<sup>CDM</sup>&#x2191;</th>
            <th>Table<sup>TEDS</sup>&#x2191;</th>
            <th>Table<sup>TEDS-S</sup>&#x2191;</th>
            <th>Read Order<sup>Edit</sup>&#x2193;</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>MinerU2.5-Pro</td>
            <td>Specialized VLMs</td>
            <td>1.2B</td>
            <td><strong>95.75</strong></td>
            <td><ins>0.036<ins></td>
            <td><strong>97.45</strong></td>
            <td><strong>93.42</strong></td>
            <td><strong>95.92</strong></td>
            <td><ins>0.120<ins></td>
        </tr>
        <tr>    
            <td>GLM-OCR</td>
            <td>Specialized VLMs</td>
            <td>0.9B</td>
            <td><ins>95.22<ins></td>
            <td>0.044</td>
            <td><ins>97.18<ins></td>
            <td><ins>92.83<ins></td>
            <td><ins>95.39<ins></td>
            <td>0.133</td>
        </tr>
        <tr>    
            <td>PaddleOCR-VL-1.5</td>
            <td>Specialized VLMs</td>
            <td>0.9B</td>
            <td>94.93</td>
            <td>0.038</td>
            <td>96.89</td>
            <td>91.67</td>
            <td>94.37</td>
            <td>0.130</td>
        </tr>
        <tr>    
            <td>PaddleOCR-VL</td>
            <td>Specialized VLMs</td>
            <td>0.9B</td>
            <td>94.18</td>
            <td>0.040</td>
            <td>95.91</td>
            <td>90.65</td>
            <td>93.74</td>
            <td>0.135</td>
        </tr>
        <tr>
            <td>Youtu-Parsing</td>
            <td>Specialized VLMs</td>
            <td>2.5B</td>
            <td>93.74</td>
            <td>0.044</td>
            <td>93.63</td>
            <td>92.02</td>
            <td>95.00</td>
            <td><strong>0.116<strong></td>
        </tr>
        <tr>
            <td>Qianfan-OCR</td>
            <td>Specialized VLMs</td>
            <td>4B</td>
            <td>93.90</td>
            <td>0.04</td>
            <td>95.08</td>
            <td>90.53</td>
            <td>93.31</td>
            <td>0.13</td>
        </tr>
        <tr>
            <td>Ovis2.6-30B-A3B</td>
            <td>General VLMs</td>
            <td>30B</td>
            <td>93.70</td>
            <td><strong>0.035<strong></td>
            <td>95.17</td>
            <td>89.44</td>
            <td>92.40</td>
            <td>0.135</td>
        </tr>
        <tr>
            <td>Logics-Parsing-v2</td>
            <td>Specialized VLMs</td>
            <td>4B</td>
            <td>93.33</td>
            <td>0.041</td>
            <td>95.65</td>
            <td>88.42</td>
            <td>91.98</td>
            <td>0.137</td>
        </tr>
         <tr>
            <td>ABot-OCR</td>
            <td>Specialized VLMs</td>
            <td>2B</td>
            <td>93.30</td>
            <td>0.037</td>
            <td>94.86</td>
            <td>88.69</td>
            <td>91.87</td>
            <td>0.137</td>
        </tr>
        <tr>
            <td>FireRed-OCR</td>
            <td>Specialized VLMs</td>
            <td>2B</td>
            <td>93.26</td>
            <td>0.037</td>
            <td>95.44</td>
            <td>88.04</td>
            <td>91.06</td>
            <td>0.131</td>
        </tr>
        <tr>
            <td>MinerU-2.5</td>
            <td>Specialized VLMs</td>
            <td>1.2B</td>
            <td>93.04</td>
            <td>0.045</td>
            <td>95.77</td>
            <td>87.88</td>
            <td>91.47</td>
            <td>0.130</td>
        </tr>
        <tr>
            <td>Gemini 3 Pro</td>
            <td>General VLMs</td>
            <td>-</td>
            <td>92.91</td>
            <td>0.064</td>
            <td>95.99</td>
            <td>89.15</td>
            <td>92.96</td>
            <td>0.165</td>
        </tr>
        <tr>
            <td>Gemini 3 Flash</td>
            <td>General VLMs</td>
            <td>-</td>
            <td>92.62</td>
            <td>0.066</td>
            <td>95.16</td>
            <td>89.29</td>
            <td>93.51</td>
            <td>0.172</td>
        </tr>
        <tr>
            <td>dots.ocr</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>90.77</td>
            <td>0.048</td>
            <td>89.95</td>
            <td>87.18</td>
            <td>90.58</td>
            <td>0.138</td>
        </tr>
        <tr>
            <td>OpenDoc-0.1B</td>
            <td>Specialized VLMs</td>
            <td>0.1B</td>
            <td>90.67</td>
            <td>0.049</td>
            <td>93.02</td>
            <td>83.88</td>
            <td>87.45</td>
            <td>0.140</td>
        </tr>
        <tr>
            <td>DeepSeek-OCR 2</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>90.25</td>
            <td>0.050</td>
            <td>91.84</td>
            <td>83.89</td>
            <td>87.75</td>
            <td>0.144</td>
        </tr>
        <tr>
            <td><strong>RapidDoc</strong></td>
            <td><strong>Pipeline Tools</strong></td>
            <td><strong>-</strong></td>
            <td><strong>90.157</strong></td>
            <td><strong>0.047</strong></td>
            <td><strong>93.777</strong></td>
            <td><strong>81.394</strong></td>
            <td><strong>88.402</strong></td>
            <td><strong>0.136</strong></td>
        </tr>
        <tr>
            <td>HunyuanOCR</td>
            <td>Specialized VLMs</td>
            <td>1B</td>
            <td>89.95</td>
            <td>0.088</td>
            <td>87.68</td>
            <td>91.01</td>
            <td>93.23</td>
            <td>0.171</td>
        </tr>
        <tr>
            <td>Qwen3-VL-235B</td>
            <td>General VLMs</td>
            <td>235B</td>
            <td>89.78</td>
            <td>0.063</td>
            <td>92.55</td>
            <td>83.07</td>
            <td>86.75</td>
            <td>0.166</td>
        </tr>
        <tr>
            <td>Dolphin-v2</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>89.50</td>
            <td>0.069</td>
            <td>91.01</td>
            <td>84.40</td>
            <td>87.44</td>
            <td>0.150</td>
        </tr>
        <tr>
            <td>OCRVerse</td>
            <td>Specialized VLMs</td>
            <td>4B</td>
            <td>88.60</td>
            <td>0.063</td>
            <td>89.61</td>
            <td>82.44</td>
            <td>86.27</td>
            <td>0.163</td>
        </tr>
        <tr>
            <td>MonkeyOCR-pro-3B</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>88.57</td>
            <td>0.074</td>
            <td>88.74</td>
            <td>84.35</td>
            <td>88.62</td>
            <td>0.189</td>
        </tr>
        <tr>
            <td>GPT-5.2</td>
            <td>General VLMs</td>
            <td>-</td>
            <td>86.59</td>
            <td>0.114</td>
            <td>88.21</td>
            <td>82.95</td>
            <td>87.93</td>
            <td>0.193</td>
        </tr>
        <tr>
            <td>Dolphin-1.5</td>
            <td>Specialized VLMs</td>
            <td>0.3B</td>
            <td>86.52</td>
            <td>0.094</td>
            <td>87.49</td>
            <td>81.43</td>
            <td>84.82</td>
            <td>0.167</td>
        </tr>
        <tr>
            <td>MinerU-Pipeline</td>
            <td>Pipeline Tools</td>
            <td>-</td>
            <td>86.47</td>
            <td>0.055</td>
            <td>83.07</td>
            <td>81.88</td>
            <td>88.68</td>
            <td>0.153</td>
        </tr>
        <tr>
            <td>olmOCR</td>
            <td>Specialized VLMs</td>
            <td>7B</td>
            <td>85.74</td>
            <td>0.139</td>
            <td>88.10</td>
            <td>83.00</td>
            <td>87.17</td>
            <td>0.216</td>
        </tr>
        <tr>
            <td>Mistral OCR</td>
            <td>Specialized VLMs</td>
            <td>-</td>
            <td>85.66</td>
            <td>0.097</td>
            <td>89.91</td>
            <td>76.78</td>
            <td>80.93</td>
            <td>0.171</td>
        </tr>
        <tr>
            <td>Kimi K2.5</td>
            <td>General VLMs</td>
            <td>1T</td>
            <td>84.53</td>
            <td>0.107</td>
            <td>83.50</td>
            <td>80.76</td>
            <td>84.00</td>
            <td>0.211</td>
        </tr>
        <tr>
            <td>InternVL3.5-241B</td>
            <td>General VLMs</td>
            <td>241B</td>
            <td>83.76</td>
            <td>0.130</td>
            <td>89.95</td>
            <td>74.35</td>
            <td>79.78</td>
            <td>0.215</td>
        </tr>
        <tr>
            <td>Nanonets-OCR-s</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>83.61</td>
            <td>0.108</td>
            <td>81.46</td>
            <td>80.18</td>
            <td>84.51</td>
            <td>0.213</td>
        </tr>
        <tr>
            <td>POINTS-Reader</td>
            <td>Specialized VLMs</td>
            <td>3B</td>
            <td>83.37</td>
            <td>0.096</td>
            <td>85.72</td>
            <td>73.98</td>
            <td>77.40</td>
            <td>0.198</td>
        </tr>
        <tr>
            <td>Marker</td>
            <td>Pipeline Tools</td>
            <td>-</td>
            <td>78.44</td>
            <td>0.157</td>
            <td>85.24</td>
            <td>65.77</td>
            <td>73.24</td>
            <td>0.243</td>
        </tr>
    </tbody>
</table>

## 🛠️ 安装RapidDoc

#### 使用pip安装
```bash
pip install rapid-doc[cpu] -i https://mirrors.aliyun.com/pypi/simple
或
pip install rapid-doc[gpu] -i https://mirrors.aliyun.com/pypi/simple
```

#### 通过源码安装
```bash
# 克隆仓库
git clone https://github.com/RapidAI/RapidDoc.git
cd RapidDoc

# 安装依赖
pip install -e .[cpu] -i https://mirrors.aliyun.com/pypi/simple
或
pip install -e .[gpu] -i https://mirrors.aliyun.com/pypi/simple
```
#### 使用gpu推理
```python
# rapid-doc[gpu] 默认安装 onnxruntime-gpu 最新版
# 需要确定onnxruntime-gpu与GPU对应，参考 https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

# 在 Python 中指定 GPU（必须在导入 rapid_doc 之前设置）
import os
# 使用默认 GPU（cuda:0）
os.environ['MINERU_DEVICE_MODE'] = "cuda"
# 或指定 GPU 编号，例如使用第二块 GPU（cuda:1）
os.environ['MINERU_DEVICE_MODE'] = "cuda:1"
```
#### 使用PaddleOCRVL系列推理
vl模型的部署，参考[官方文档](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html#31-vlm) 
```python
import os
os.environ['PADDLEOCRVL_VERSION'] = "v1.6"
os.environ['PADDLEOCRVL_VL_REC_BACKEND'] = "vllm-server"
os.environ['PADDLEOCRVL_VL_VL_REC_SERVER_URL'] = "http://localhost:8118/v1"

from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
from rapid_doc.model.custom.paddleocr_vl.paddleocr_vl import PaddleOCRVLTableModel, PaddleOCRVLOCRModel, PaddleOCRVLFormulaModel
layout_config = {
    "model_type": LayoutModelType.PP_DOCLAYOUTV3,
}
ocr_config = {
    "custom_model": PaddleOCRVLOCRModel()
}
formula_config = {
    "custom_model": PaddleOCRVLFormulaModel()
}
table_config = {
    "custom_model": PaddleOCRVLTableModel()
}
```

#### 使用docker部署RapidDoc
RapidDoc提供了便捷的docker部署方式，这有助于快速搭建环境并解决一些棘手的环境兼容问题。

您可以在文档中获取 [Docker部署说明](docker/README.md)，镜像已推送至 [Docker Hub](https://hub.docker.com/r/hzkitty/rapid-doc)。

---
### 📋 使用

```python
import os
from pathlib import Path
from rapid_doc import RapidDoc
__dir__ = Path(__file__).resolve().parent.parent
output_dir = os.path.join(__dir__, "output")

doc_path_list = [
    __dir__ / "demo/pdfs/示例1-论文模板.pdf",
    __dir__ / "demo/docx/test.docx",
]
engine = RapidDoc()
outputs = engine(doc_path_list, output_dir=output_dir)
for output in outputs:
    print(output.markdown)
```
---

## 在线体验

### 基于Gradio的在线demo
基于gradio开发的webui，界面简洁，仅包含核心解析功能，免登录

- [![ModelScope](https://img.shields.io/badge/Demo_on_ModelScope-purple?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIzIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCiA8Zz4KICA8dGl0bGU+TGF5ZXIgMTwvdGl0bGU+CiAgPHBhdGggaWQ9InN2Z18xNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTAsODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTUiIGZpbGw9IiM2MjRhZmYiIGQ9Im05OS4xNCwxMTUuNDlsMjUuNjUsMGwwLDI1LjY1bC0yNS42NSwwbDAsLTI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTYiIGZpbGw9IiM2MjRhZmYiIGQ9Im0xNzYuMDksMTQxLjE0bC0yNS42NDk5OSwwbDAsMjIuMTlsNDcuODQsMGwwLC00Ny44NGwtMjIuMTksMGwwLDI1LjY1eiIvPgogIDxwYXRoIGlkPSJzdmdfMTciIGZpbGw9IiMzNmNmZDEiIGQ9Im0xMjQuNzksODkuODRsMjUuNjUsMGwwLDI1LjY0OTk5bC0yNS42NSwwbDAsLTI1LjY0OTk5eiIvPgogIDxwYXRoIGlkPSJzdmdfMTgiIGZpbGw9IiMzNmNmZDEiIGQ9Im0wLDY0LjE5bDI1LjY1LDBsMCwyNS42NWwtMjUuNjUsMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzE5IiBmaWxsPSIjNjI0YWZmIiBkPSJtMTk4LjI4LDg5Ljg0bDI1LjY0OTk5LDBsMCwyNS42NDk5OWwtMjUuNjQ5OTksMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIwIiBmaWxsPSIjMzZjZmQxIiBkPSJtMTk4LjI4LDY0LjE5bDI1LjY0OTk5LDBsMCwyNS42NWwtMjUuNjQ5OTksMGwwLC0yNS42NXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIxIiBmaWxsPSIjNjI0YWZmIiBkPSJtMTUwLjQ0LDQybDAsMjIuMTlsMjUuNjQ5OTksMGwwLDI1LjY1bDIyLjE5LDBsMCwtNDcuODRsLTQ3Ljg0LDB6Ii8+CiAgPHBhdGggaWQ9InN2Z18yMiIgZmlsbD0iIzM2Y2ZkMSIgZD0ibTczLjQ5LDg5Ljg0bDI1LjY1LDBsMCwyNS42NDk5OWwtMjUuNjUsMGwwLC0yNS42NDk5OXoiLz4KICA8cGF0aCBpZD0ic3ZnXzIzIiBmaWxsPSIjNjI0YWZmIiBkPSJtNDcuODQsNjQuMTlsMjUuNjUsMGwwLC0yMi4xOWwtNDcuODQsMGwwLDQ3Ljg0bDIyLjE5LDBsMCwtMjUuNjV6Ii8+CiAgPHBhdGggaWQ9InN2Z18yNCIgZmlsbD0iIzYyNGFmZiIgZD0ibTQ3Ljg0LDExNS40OWwtMjIuMTksMGwwLDQ3Ljg0bDQ3Ljg0LDBsMCwtMjIuMTlsLTI1LjY1LDBsMCwtMjUuNjV6Ii8+CiA8L2c+Cjwvc3ZnPg==&labelColor=white)](https://www.modelscope.cn/studios/RapidAI/RapidDoc)

---

## 📋 使用示例

- [代码示例](./demo/demo.py)

- [参数介绍](./docs/analyze_param.md)

- [FastAPI 示例](./docker/README_API.md)
---

## 模型下载
不指定模型路径，初次运行时，会自动下载
- [RapidDoc 模型集（版面/公式/表格）](https://www.modelscope.cn/models/RapidAI/RapidDoc)  
- [RapidOCR 模型](https://www.modelscope.cn/models/RapidAI/RapidOCR)  
- [部分表格模型RapidTable](https://www.modelscope.cn/models/RapidAI/RapidTable) 

---

## 📌 TODO

- [x] 跨页表格合并
- [x] 复选框识别，使用opencv（默认关闭、opencv识别存在误检）
- [x] 提供 fastapi，支持cpu和gpu版本的docker镜像构建
- [x] 文本型pdf，表格非OCR文本提取
- [x] 文本型pdf，使用pypdfium2提取文本框bbox
- [x] 文本型pdf，支持0/90/270度三个方向的表格解析
- [x] 文本型pdf，使用pypdfium2提取原始图片（默认截图会导致清晰度降低和图片边界可能丢失部分）
- [x] 表格内公式提取，表格内图片提取
- [x] 优化阅读顺序，支持多栏、竖排等复杂版面恢复
- [x] 公式支持torch推理，可用GPU加速
- [x] 版面、表格模型支持openvino
- [x] markdown转docx、html
- [x] 支持 PP-DocLayoutV2 版面识别+阅读顺序
- [x] OmniDocBench评测
- [x] 支持自定义的ocr、table、公式。支持PaddleOCR-VL系列
- [x] 支持docx/doc、pptx/ppt、xlsx/xls的原生解析（不使用模型）
- [x] 支持印章文本检测
- [x] 文档方向90°、270°矫正（默认关闭），表格方向90°、270°矫正（默认开启）

## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## Star History

<a href="https://www.star-history.com/?repos=RapidAI%2FRapidDoc&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=RapidAI/RapidDoc&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=RapidAI/RapidDoc&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=RapidAI/RapidDoc&type=date&legend=top-left" />
 </picture>
</a>

## ⚖️ 开源许可

基于 [MinerU](https://github.com/opendatalab/MinerU) 改造而来，已**移除原项目中的 YOLO 模型**，并替换为 **PP-StructureV3 系列 ONNX 模型**。  
由于已移除 AGPL 授权的 YOLO 模型部分，本项目整体不再受 AGPL 约束。

该项目采用 [Apache 2.0 license](LICENSE) 开源许可证。
