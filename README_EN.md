# RapidDoc - Fast Document Parsing System

[English](README_EN.md) | [中文](README.md)

## Introduction

RapidDoc is a lightweight open-source framework focused on document parsing. It supports OCR, layout analysis, formula recognition, table recognition, reading-order recovery, and conversion from complex documents to Markdown, JSON, Word, and HTML.

RapidDoc also supports native parsing for `docx/doc`, `pptx/ppt`, and `xlsx/xls` files without using neural models.

The project is based on [MinerU](https://github.com/opendatalab/MinerU), with VLM components removed. It focuses on efficient pipeline-based document parsing and keeps good performance even on CPU.

Most default models come from the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) [PP-StructureV3](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PP-StructureV3.html) family, including OCR, layout analysis, formula recognition, reading-order recovery, and some table models. These models are converted to ONNX for efficient CPU/GPU inference.

RapidDoc also supports custom OCR, formula, and table models by implementing the `CustomBaseModel.batch_predict` method. Integration for the [PaddleOCR-VL](https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html) series is included.

## Choose an Engine and Language SDK

RapidDoc provides the original Python implementation and a shared Rust engine. The original Python package remains `rapid-doc`. The new Python, Java, Go and Node.js bindings call the same Rust parsing core without installing the original Python package. Rust, Java, Go and Node.js applications do not require Python at runtime.

| Interface | Package / import name | Usage |
| --- | --- | --- |
| Original Python | `rapid-doc` / `rapid_doc` | [Installation](#installation) |
| Python binding to Rust | `rapid-doc-rust` / `rapid_doc_rust` | [Python SDK](bindings/python/README.md) |
| Rust | `rapiddoc` | [Rust SDK](rust/rapiddoc/README.md) |
| Java | `io.github.rapidai:rapid-doc` | [Java SDK](bindings/java/README.md) |
| Go | `github.com/RapidAI/RapidDoc/bindings/go` | [Go SDK](bindings/go/README.md) |
| Node.js / TypeScript | `rapid-doc` on npm | [Node.js SDK](bindings/node/README.md) |

**Release status: the Rust SDK and its language bindings are development previews at version `0.1.0`, not yet published to public package registries.** These are the intended package names. Follow the individual SDK guides for source builds or local artifacts; future registry installation examples are not available releases yet. See the [publishing guide](docs/publishing.md) for build and validation workflows.

The Rust engine supports PDF, DOC, DOCX, XLS, XLSX, PPT and PPTX, with native Office parsing and ONNX Runtime for PDF model inference. Delivery of PDFium, ORT, GEOS and CUDA operators depends on the language and platform package; deploying a single shared library is insufficient. Missing models can be downloaded automatically or supplied through an existing model directory; see [model configuration](docs/rust-models.md).

Feature parity is not complete. The Rust PDF pipeline supports V2/V3 layout, S/M/L formula models, UNET/SLANet and Table Transformer. Full Plus-L PDF reading order, enhanced reading order and character-box splitting with `use_word_box=True` remain deferred. Python's Torch, alternative OCR engines and `CustomBaseModel` extensions are not automatically available in the Rust bindings. Consult the SDK guides for supported options and defaults.

The existing [Docker / FastAPI deployment](docker/README.md) uses the original Python implementation; a Rust image has not replaced it. Compare output, time and memory with identical models, parameters and documents. A language change alone does not guarantee a fixed performance multiplier.

## Features

This section describes the original Python implementation. See the SDK guides above for Rust model and parameter support.

- OCR recognition powered by [RapidOCR](https://github.com/RapidAI/RapidOCR), using ONNX Runtime by default on CPU/CUDA, with other engines available through explicit configuration.
- Layout analysis with PP-DocLayout models, including PP-DocLayoutV3 as the default.
- Formula recognition with PP-FormulaNet_plus models, using ONNX Runtime by default on CPU/CUDA. Explicit Torch configuration requires installing the corresponding Torch dependencies.
- Table recognition with table classification, wired-table recognition, and SLANET_plus/UNITABLE for wireless tables.
- Reading-order recovery using layout model output or PP-StructureV3-style XY-cut logic.
- Native Office parsing for Word, PowerPoint, and Excel files.
- Markdown, Word, HTML, JSON, and image output workflows.
- Docker and FastAPI deployment examples.

## Installation

These commands install the original Python implementation, `rapid-doc`, not the Rust engine. For the Rust-backed Python package, see the [Python SDK guide](bindings/python/README.md).

Install the CPU version:

```bash
pip install "rapid-doc[cpu]"
```

Install the GPU version:

```bash
pip install "rapid-doc[gpu]"
```

The default pipeline models (OCR, layout, formula, and table) use ONNX Runtime. The CPU extra installs `onnxruntime`; the GPU extra installs `onnxruntime-gpu` without adding the CPU runtime distribution or Torch by default. Choose one runtime distribution: both expose the same `onnxruntime` import name and should not be installed together. Native Office and DocVortex rule parsing do not use neural inference.

Install from source:

```bash
git clone https://github.com/RapidAI/RapidDoc.git
cd RapidDoc
pip install -e ".[cpu]"
```

For GPU development:

```bash
pip install -e ".[gpu]"
```

## GPU Inference

Set the device before importing `rapid_doc`:

```python
import os

os.environ["MINERU_DEVICE_MODE"] = "cuda"    # cuda:0
# os.environ["MINERU_DEVICE_MODE"] = "cuda:1"
```

The GPU extra installs `onnxruntime-gpu` within the project dependency constraint (currently `<=1.23.0`). Match CUDA/cuDNN to the [CUDA Execution Provider requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements). Default OCR and formula inference use ONNX Runtime and do not require switching to Torch.

OCR enables `onnx_ops` by default, and the formula ONNX path also integrates the bundled CUDA operator library. See the [operator library documentation](native/onnx_ops/README.md) for native-library CUDA/cuDNN requirements, build instructions, and platform validation coverage. The operator library is separate from `onnxruntime-gpu`; installing the runtime alone does not establish that the native library and its system dependencies are ready.

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

This project is released under the [Apache 2.0 license](LICENSE).

## Built-in native parsing

PDF rule layout and tables use the bundled DocVortex core. DOC/DOCX, XLS/XLSX and PPT/PPTX use native parsers without LibreOffice conversion. The core uses Python kernels and no fasttext language model. Install `rapid-doc[cpu]` or `rapid-doc[gpu]` for the inference runtime; the GPU extra does not install the CPU runtime distribution. See [migration notes](docs/migrations/docvortex-baseline.md).
