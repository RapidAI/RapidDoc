# RapidDoc - Fast Document Parsing System

[English](README_EN.md) | [中文](README.md)

## Introduction

RapidDoc is a lightweight open-source framework focused on document parsing. It supports OCR, layout analysis, formula recognition, table recognition, reading-order recovery, and conversion from complex documents to Markdown, JSON, Word, and HTML.

RapidDoc also supports native parsing for `docx/doc`, `pptx/ppt`, and `xlsx/xls` files without using neural models.

The project is based on [MinerU](https://github.com/opendatalab/MinerU), with VLM components removed. It focuses on efficient pipeline-based document parsing and keeps good performance even on CPU.

Most default models come from the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) [PP-StructureV3](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PP-StructureV3.html) family, including OCR, layout analysis, formula recognition, reading-order recovery, and some table models. These models are converted to ONNX for efficient CPU/GPU inference.

RapidDoc also supports custom OCR, formula, and table models by implementing the `CustomBaseModel.batch_predict` method. Integration for the [PaddleOCR-VL](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html) series is included.

## Features

- OCR recognition powered by [RapidOCR](https://github.com/RapidAI/RapidOCR), with multiple inference engines.
- Layout analysis with PP-DocLayout models, including PP-DocLayoutV3 as the default.
- Formula recognition with PP-FormulaNet_plus models.
- Table recognition with table classification, wired-table recognition, and SLANET_plus/UNITABLE for wireless tables.
- Reading-order recovery using layout model output or PP-StructureV3-style XY-cut logic.
- Native Office parsing for Word, PowerPoint, and Excel files.
- Markdown, Word, HTML, JSON, and image output workflows.
- Docker and FastAPI deployment examples.

## Installation

Install the CPU version:

```bash
pip install rapid-doc[cpu]
```

Install the GPU version:

```bash
pip install rapid-doc[gpu]
```

Install from source:

```bash
git clone https://github.com/RapidAI/RapidDoc.git
cd RapidDoc
pip install -e .[cpu]
```

For GPU development:

```bash
pip install -e .[gpu]
```

## GPU Inference

Set the device before importing `rapid_doc`:

```python
import os

os.environ["MINERU_DEVICE_MODE"] = "cuda"    # cuda:0
# os.environ["MINERU_DEVICE_MODE"] = "cuda:1"
```

The GPU package installs `onnxruntime-gpu`. Make sure your CUDA/cuDNN environment matches the ONNXRuntime CUDA execution provider requirements.

## Quick Start

```python
import os
from pathlib import Path

from rapid_doc import RapidDoc

root = Path(__file__).resolve().parent
output_dir = root / "output"

doc_path_list = [
    root / "demo/pdfs/示例1-论文模板.pdf",
    root / "demo/docx/test.docx",
]

engine = RapidDoc()
outputs = engine(doc_path_list, output_dir=output_dir)

for output in outputs:
    print(output.markdown)
```

## Docker

RapidDoc provides Docker deployment examples for quickly setting up CPU/GPU environments.

- [Docker guide](docker/README.md)
- [FastAPI guide](docker/README_API.md)
- [Docker Hub image](https://hub.docker.com/r/hzkitty/rapid-doc)

## Online Demo

Try the Gradio demo on ModelScope:

[RapidDoc on ModelScope](https://www.modelscope.cn/studios/RapidAI/RapidDoc)

## Examples and Parameters

- [Code example](demo/demo.py)
- [Parameter guide](docs/analyze_param.md)
- [FastAPI example](docker/README_API.md)

## Model Downloads

If model paths are not specified, RapidDoc downloads models automatically on first run.

- [RapidDoc models: layout, formula, table](https://www.modelscope.cn/models/RapidAI/RapidDoc)
- [RapidOCR models](https://www.modelscope.cn/models/RapidAI/RapidOCR)
- [RapidTable models](https://www.modelscope.cn/models/RapidAI/RapidTable)

## Benchmark

RapidDoc is evaluated on OmniDocBench v1.6 using a pipeline with PP-DocLayoutV3, PP-OCRv6-small, PP-FormulaNet_plus-M, and UNET_SLANET_PLUS. See the Chinese [README](README.md#基准测试结果) for the detailed benchmark table.

## Acknowledgements

- [MinerU](https://github.com/opendatalab/MinerU)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## License

RapidDoc is derived from [MinerU](https://github.com/opendatalab/MinerU). The original YOLO model components under AGPL have been removed and replaced with PP-StructureV3 ONNX models.

This project is released under the [Apache 2.0 license](LICENSE).
