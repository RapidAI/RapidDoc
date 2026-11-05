# Linux 多语言安装验收

验收目录：服务器 `/web/hzkitty/rapiddoc-release-acceptance/20261104`。
本机原始记录：`build/release-acceptance-20261104-receipts.tar.gz`，解压于
`build/release-acceptance-receipts/`。本次不发布包，不替换业务环境。

## 修改

- Rust 共享 SDK 接受 `table_config.use_word_box=false/null`，与省略参数一致。
  字符串值仍报参数错误；`true` 的字符框拆分尚未接入生产路径，明确拒绝。
- Node 移除全局 mimalloc，使用宿主分配器，修复 Linux 关闭首个 worker 后
  重建 worker 在分配器初始化时发生的 SIGSEGV。保留模块驻留措施。
- 增加连续八次创建、解析、关闭 Node engine 的回归，重新生成 CPU/CUDA npm 包。

## 验证

环境：Linux x86_64、RTX 4090、ORT 1.23.0、Python 3.10、Java 11、
Go 1.20.5、Node 24.12.0。工具和依赖放在独立目录，不替换系统 Node/Java。

| 绑定 | CPU | CUDA | 范围 |
| --- | --- | --- | --- |
| Python wheel | 通过 | 通过 | 独立 venv，7 格式及扫描/混合 PDF，共 9 个输入 |
| Java 平台 JAR | 通过 | 通过 | 每组 11 项，含 7 格式、OCR、并发实例、异常及输出 |
| Go C ABI | 通过 | 通过 | 7 格式、OCR、回调、并发实例、关闭后结果、输出路径 |
| Node npm 包 | 通过 | 通过 | 独立安装消费者及每组 7 项生命周期/进度测试 |

Python venv 只提供声明的 Python 依赖及平台原生运行库，不继承旧 Rust 包，
不依赖 Python `onnxruntime` 模块执行推理。CUDA 原生库通过 NVIDIA 库目录加载。
原有两个 GPU 服务 PID 和显存占用与验收前一致。

Windows 补跑新 Node 包的 7 项受影响测试，通过；独立 Node demo 已安装新包。
显式 `use_word_box=false` 的 Rust 回归和本机 Python 输出等价检查通过。

## 对照范围和差异

Windows 固定参考不能直接用于 Linux 图片逐字节比较。两个系统都使用
PDFium 151.0.7891.0；各系统内 Python pypdfium2 与 Rust 渲染的 RGB 像素
完全一致，但系统之间的文字抗锯齿像素存在差别，导致裁图哈希和 OCR 输入不同。
检测/识别模型 SHA256 已核对一致。混合页的旋转文字分别得到 Windows
`Rotd ttext`、Linux `Rod text`，不能宣称跨平台输出完全相同。

原 Windows 参考未在本机修改，已另存到服务器报告。Linux 验收副本使用同系统
C ABI 的完整输出检查各语言适配结果，属于跨绑定一致性验收，不是新的独立
Python 精度基准。修改字段清单保存在 `reports/platform-baseline-audit.json`。

Linux CPU/CUDA 三个 PDF 的 Markdown、内容列表、坐标和图片均相同；只有
混合 PDF 的 10 个 OCR score 字段（含重复存储）相差 0.001。脚本逐字段确认
仅有这些差异后，为 CUDA 使用对应的完整 middle_json 作跨绑定对照，未放宽
其他字段断言。记录：`reports/cpu-cuda-exact-comparison.json`。

## 边界

本次是平台包安装、基础 OCR 和绑定行为验收，未重跑全量评测、性能/峰值内存
对比，也未逐一复测所有版面、公式和表格模型。宿主分配器对 Node 的性能与
峰值内存影响尚未单独量化。包尚未发布；`use_word_box=true` 仍待实现。
