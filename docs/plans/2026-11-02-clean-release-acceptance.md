# 干净环境发布验收（2026-11-02）

## 范围

使用独立 venv、不继承 system-site-packages，从空工作目录用 `python -I`
运行。项目 `.venv` 和服务器业务环境不改。测试环境不安装 rapid_doc、
RapidOCR、pdftext、OpenCV、onnx、Shapely、pypdfium2；校验这些模块找不到，
也没有在解析时导入。只安装 wheel 声明的依赖。

本轮为安装与运行库加载验收，不替代上一轮同模型/同 ORT 的完整精度、
速度与内存验收。旧生产 Rust 二进制和 Python 解析代码保持不变。

## Windows CPU

本地新环境位于 build/release-acceptance/windows-cpu/env，普通 pip 安装
此前交付的 Windows wheel，自动解析为 Python 3.14.5、onnxruntime 1.31.1、
Pillow 12.3.0、NumPy 2.5.4。`pip check` 通过。

包内 PDFium/GEOS 和依赖中的 ORT 自动发现；未设置任何运行库路径。
DOC/DOCX/XLS/XLSX/PPT/PPTX 各一份完整文档通过；扫描、公式、密集表格、
印章各一个真实 PDF 页通过，输出非空且返回页数正确。实际下载一次
默认 orientation 模型，SHA 与清单一致，第二次解析路径相同且无 .part
残留。为避免重复下载大型原始模型，其余固定模型以硬链接放入独立
测试模型目录；这不是“全部十个模型从网络完整下载”的声明。
十个默认模型地址额外 HEAD 检查均为 200，不代替完整内容哈希验证。

## Linux CUDA 打包修正

此前 Linux SDK 以验收脚本传入外部 PDFium、GEOS，原测试 wheel 没有携带
Linux 的这两个库，不能称为无配置独立部署。本轮仅修复发布资产：
从已验证 pypdfium2 5.10.1 取原始 PDFium 二进制，改私有文件名，携带对应
构建的原始许可证；GEOS 取配对的 3.13.1 库，携带原分发许可及已有匹配
LGPL 源码归档。生成 Linux 平台清单、资产 SHA 和正确 wheel RECORD。
不编译或修改 Rust/模型，不要求用户安装 pypdfium2、Shapely。

CUDA 专用 wheel 使用 build tag `1cuda`，包名仍 rapid-doc-rust、导入名仍
rapid_doc_rust；该构建的 METADATA 声明 onnxruntime-gpu 及已验收的 CUDA
运行库依赖，替代 CPU onnxruntime 依赖，不同时安装两种 ORT。默认 Windows
CPU 包依赖保持不变。这是本地预览构建配置，不表示已发布 PyPI。

GPU 发布依赖固定：ORT GPU 1.23.0、cuBLAS 12.4.5.8、CUDA Runtime/NVRTC
12.4.127、cuDNN 9.1.0.70，和完成精度验收的服务环境一致。
服务器网络下载太慢，独立部署测试使用离线 wheelhouse：从已安装依赖
逐文件核对原 RECORD 哈希后重新装为 wheel，保留原许可和入口点。
它不是上游原始 wheel 文件，manifest 明确记录来源；pip 将依赖完整装入
新 venv，不用 .pth、PYTHONPATH、system-site-packages 或旧环境包路径。

最终使用另一全新 venv `env-delivery` 重新执行普通 pip 离线依赖安装，
没有 --no-deps 或继承业务包目录，`pip check` 通过。实际环境为 Python
3.10.12、ORT GPU 1.23.0、Pillow 12.2.0、NumPy 2.2.6。六种 Office
完整文件和四个 PDF 场景全部通过；默认运行不设置任何 ORT/PDFium/
GEOS/算子库路径覆盖，也清空 LD_LIBRARY_PATH 和 PYTHONPATH。

采样时当前解析进程有 3,548 MiB 显存分配；进程映射中的 cuBLAS、
cuDNN、CUDA Runtime 全部来自新 venv 的 nvidia 包目录。模型描述符
均为 CUDA；SLANet 原设计的 CPU decoder 不视为 CUDA 回退。
原 Python 框架及七个被禁止的模型/文档 Python 包均未安装或导入。

GEOS 初次独立加载暴露库间定位问题：Shapely 自身会预加载 GEOS 核心，
它分发的 GEOS C 库没有相邻库搜索路径。本包不导入 Shapely，因此在
打包副本上用构建工具 patchelf 设置 RUNPATH=$ORIGIN；核心库和算法
代码不变，记录原始/交付库哈希。不通过设置用户环境变量绕开问题。
构建工具只在单独 release-tooling 中，不是解析依赖。另修复离线导出
工具对 py2/py3 多 Tag 轮子的文件名选择；最终选择当前解释器兼容 Tag。

为检查运行库改动影响，在相同干净环境和模型下，四个 PDF 场景分别
使用新包内 GEOS 和此前显式系统 GEOS，Markdown、完整 middle_json /
content_list 和图片编码内容的联合 SHA 均完全一致。此为四页定向
对照，不宣称已用新打包 GEOS 重跑 183 页全套。

服务器实际模型下载因网络过慢取消，.part 已清理。Linux 复用已有
SHA 验证的原始模型硬链接；网络下载和复用能力以 Windows 的真实
默认 URL 下载为准，不把 Linux 取消的下载记作通过。

新 GPU wheel 的 Rust 扩展和所有 Python 解析文件与已验收基础包逐
字节一致，不携带任何 ONNX 模型；CUDA 依赖元数据和所有 wheel RECORD
已校验。未向当前业务环境安装该新包、未重启既有服务。


## 复现脚本

- scripts/migration/accept_clean_install.py：独立环境模型下载、运行库和解析检查。
- scripts/migration/prepare_linux_release_wheel.py：阶段二进制资产及 CUDA 元数据。
- scripts/migration/export_verified_wheelhouse.py：校验已安装依赖、导出离线包。

所有发布资产为测试/交付文件，未推送仓库或发布到外部服务。


## 最终交付位置与边界

- Windows CPU 包：`build/cuda-memory-final-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`。
- Linux CUDA 包：`build/release-acceptance/linux-cuda-wheels/rapid_doc_rust-0.1.0-1cuda-cp310-abi3-manylinux_2_35_x86_64.whl`。
- Windows 可读记录：`build/release-acceptance/windows-cpu/`，其中 env 是测试解释器，不是发布内容。
- Linux 可读记录：`build/release-acceptance/linux-cuda-records/`。

Linux 包、许可/哈希清单、依赖来源记录、安装/解析/运行库对照日志及脚本已归档并下载校验：`build/release-acceptance/clean-release-acceptance-results.tar.gz`，32,705,737 字节；SHA256 `8f661cfaa3323d42524bc1f6ef8b1d77ee6b108fff28b400e8b2e8882722b744`。

15 项受影响配置/下载/验收脚本检查通过，新增脚本语法检查通过；未跑
全量代码测试。本轮没有向 PyPI/GitHub/Gitee 发布。验证范围限于本机
Windows x64 CPU 和服务器 Linux x64 CUDA；Linux wheel 最低 glibc 2.35，
不适用于 musl，macOS/ARM/Windows CUDA 仍未做本轮干净环境验收。
Linux CUDA 机器需已有兼容 NVIDIA 驱动；用户态 CUDA 库由该 wheel 的
依赖自动安装。离线 wheelhouse 只留在服务器，不随 32 MB 结果归档重复
打入约 GB 级运行库依赖；其来源和原始 RECORD 验证清单已包含在归档。


隔离测试基于现有 Windows/Linux 主机新建 venv，不是重新安装空白操作
系统；未验证系统驱动、字体和 C/C++ 运行时从零部署。发布资产准备完成，
实际外部分发/索引发布仍需后续明确安排。
