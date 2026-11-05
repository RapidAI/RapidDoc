# 七种格式性能与内存检查

本轮以已安装的 office-public-delivery-20261101-r2 为基准，不改变解析阈值、模型、线程参数或文档内容。以下是本机指定样本的测量，不是对任意文档的速度保证。

## 默认 Office 完整解析

每种格式使用 demo/ms_office_docs 对应原始文档，输入冻结并校验 SHA256。两轮独立进程、交替两端顺序，每轮一次冷调用和三次热调用。耗时、冷解析内存、完整输出校验分别起进程。计时含公开 API 完整返回，不含文件读取、导入初始化、验证和文件输出。

| 格式 | Python 热中位数 ms | Rust 热中位数 ms | 加速比 | Python 进程峰值 MiB | Rust 进程峰值 MiB |
|---|---:|---:|---:|---:|---:|
| DOC | 2512.37 | 390.35 | 6.44 | 332.0 | 73.3 |
| DOCX | 498.13 | 29.13 | 17.10 | 343.9 | 49.9 |
| PPT | 62.40 | 28.16 | 2.22 | 327.5 | 65.3 |
| PPTX | 63.48 | 22.06 | 2.88 | 334.4 | 57.8 |
| XLS | 21.98 | 8.27 | 2.66 | 310.1 | 43.0 |
| XLSX | 57.75 | 33.06 | 1.75 | 327.6 | 62.1 |

全部完整 middle、Markdown、content list、资源像素/SVG 对照一致。Windows 保存的进程生命周期峰值包含导入依赖；它与算法新增内存不同。父进程外部采样目标间隔 2 ms，仍可能漏短峰，详细报告保留间隔和私有提交。

解析新增采样工作集 Python/Rust（MiB）：DOC 21.7/36.7、DOCX 45.0/15.6、PPT 13.8/30.5、PPTX 21.3/23.2、XLS 10.7/8.7、XLSX 12.9/27.6。不能用较低总进程峰值声称每个增量指标也更低。

原始结果：build/office-performance-20261101/{manifest,reports,summary}.json，60 个隔离 worker。脚本 scripts/migration/accept_office_performance.py 复用已验证的 DOC 外部内存采样方法。

## 默认 PDF 完整解析

两份完整 PDF，同样两轮、六次热调用，耗时/内存/校验隔离，默认 auto、公式和表格开关不变，未手动指定 DLL/模型。记录实际请求的 session 模型路径；Rust 按需加载使用到的模型，Python 初始化更广的模型集合，这是默认路径本身的区别。冷调用含惰性模型加载，导入初始化另计。

| 文件 | Python 冷 s | Rust 冷 s | Python 热 s | Rust 热 s | Python 峰值 MiB | Rust 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 49－长表格-密集型.pdf | 7.813 | 3.348 | 2.740 | 2.576 | 1911.1 | 1343.2 |
| small_ocr.pdf | 23.149 | 15.585 | 14.147 | 15.126 | 2422.3 | 1711.8 |

完整输出对照一致。密集表格热运行 Rust 约快 6.4%；扫描件 Rust 热耗时约高 6.9%（Python/Rust 加速比 0.935）。PDF 仍未达到“所有场景快于 Python”的目标，不能把较快冷启动或较低内存当成热运行也更快。

原始结果：build/pdf-performance-20261101/{manifest,reports,summary}.json，20 个隔离 worker。它只覆盖这两份完整文件，不是重跑所有 PDF 或 CUDA。

## 规模递增与结果释放

scripts/migration/accept_document_memory.py 生成 18 份不同规模输入，另复制 1 份真实 PPT 作重复调用。构造输入不是新增真实语料。每格式独立进程依次增加输入规模，在最大规模上同一引擎再解析/释放六次，每次 gc.collect；不保留结果，不在内存测量进程里验证/解码资源。完整输出校验另起进程。

| 格式/负载 | 规模 | Python 进程峰值 MiB | Rust 进程峰值 MiB |
|---|---|---:|---:|
| DOC Unicode 段落 | 2000 / 8000 / 32000 | 498.4 | 323.4 |
| DOCX 段落、样式、表格 | 200 / 800 / 3200 | 374.9 | 72.9 |
| XLS 六列单元格行 | 500 / 2000 / 8000 | 438.3 | 133.5 |
| XLSX 六列单元格行 | 500 / 2000 / 8000 | 427.3 | 141.3 |
| PPTX 文字、列表、表格页 | 25 / 100 / 400 | 340.6 | 58.8 |
| PPT 真实文件连续调用 | 6 页、固定输入 | 329.5 | 84.8 |
| PDF 原生文本规则窗口 | 16 / 65 / 129 页 | 2162.1 | 349.7 |

所有规模的完整输出一致。PDF 压力场景显式配置 docvortex_rule、txt、关闭公式/表格模型、pdf_pages_batch=64，覆盖跨窗口处理，但不代表默认 OCR/模型链路内存。进程峰值包含先前较小输入和导入，不能据此推算单页内存。

释放后的内存不是必须回到导入基线：运行库、分配器、缓存和堆保留均可能影响。Rust PPT/PPTX 的六次私有提交基本平稳；DOCX/XLSX 有小幅波动；DOC 约 149–154 MiB；XLS 从约 110 降到 82 MiB；PDF 在约 116–207 MiB 波动后下降，没有本次六次观察中的单调持续增长。Python DOCX/XLS/XLSX/PDF 也出现不同幅度上移。六次观察不能证明不存在长期泄漏，不将波动直接归因为泄漏。

原始结果：build/document-memory-20261101/{manifest,summary}.json，各格式 trend/verify 报告保存每个增长阶段和六次释放后的工作集/私有提交。28 个隔离 worker。

## 诊断发现

独立诊断脚本 scripts/migration/profile_pdf_pipeline.py 不改变推理参数，但计时边界不同，不能直接把 Python ORT run 等同于 Rust 整阶段或相加全部嵌套阶段。

- 扫描件：Python 热调用诊断 13.61 s，其中 ORT run 合计 10.16 s；layout 4.322 s、检测 1.906 s、识别 3.932 s。Rust 同次配置诊断热调用 14.658 s，layout 4.329 s、渲染 1.987 s、区域裁图 0.634 s、OCR 检测/裁图阶段 2.907 s、识别裁图准备 1.345 s（部分阶段嵌套）、批量识别 4.290 s。没有证据表明换成 Rust 会让相同 ONNX 模型自动快几倍；图像转换和组装仍有开销。
- 129 页原生规则诊断：Python 12.55 s、Rust 39.48 s（压力运行中约 10.49/40.47 s，只作诊断）。Rust native_analysis 28.335 s、render 7.535 s、assemble 2.715 s；native 内部的页面编排和几何/角色处理约 22.12 s，其中表格相关容器约 7.40 s。这是明确待优化的规则场景，尚未修复，不据此给出完成性能目标的结论。
- 原生规则细分来自现有 RAPIDDOC_NATIVE_PROFILE，阶段存在嵌套，禁止直接求和。报告 build/pdf-native-rule-profile-{python,rust}-20261101.json、Rust 同名日志；扫描件报告 build/pdf-scan-profile-{python,rust}-20261101.json。

后续只采用保持完整结果、像素和规则的改动。

## PNG 编码实验：未采用

开发示例 rust/rapiddoc-pdf/examples/png_filters.rs 对原生文本页、扫描页、论文页的 200 DPI RGB 像素，使用相同 Fast 压缩分别测试 Adaptive/Up/Sub/NoFilter，五次编码与解码并逐像素断言一致。记录 build/png-filters-20261101.jsonl；输入位于 build/png-filter-inputs-20261101。

Up 相比 Adaptive 的编码中位数：文本页 2.49→1.77 ms，扫描页 15.04→10.62 ms，论文页 3.29→2.16 ms；编码尺寸分别增加约 10%、31%、35%。单次解码亦有变化，但这不是端到端流水线测量，无法将其累计为解析收益。仅毫秒级编码节省不足以解释当前秒级瓶颈，故保留默认编码，没有为此更改运行包或解析规则。NoFilter 在本轮样本更慢且明显增大体积。

本轮交付是完整测量、差分与瓶颈归因，没有宣称已修复原生规则的性能差距。后续优先给页面编排/几何阶段增加更细的分配与阶段归因，再消除重复构建，不能通过跳过表格/文字规则来换速度。

## 复现

```powershell
.venv/Scripts/python.exe scripts/migration/accept_office_performance.py --output build/office-performance-new
.venv/Scripts/python.exe scripts/migration/accept_office_performance.py --pdf-only --output build/pdf-performance-new
.venv/Scripts/python.exe scripts/migration/accept_document_memory.py --output build/document-memory-new
```

输出目录必须不存在，支持相同二进制/配置下的 --resume。测量期间不要同时跑构建、其他解析或性能脚本；不清除系统文件缓存。参数、输入和二进制指纹都在 manifest 保存。本轮未重新构建或安装运行包，测量对应 `.venv` 的 R2 二进制。
