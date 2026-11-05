# 用户实测：速度、03.pdf OCR 跨批失败及日志

2026-10-29 后续已完成公式内存像素传递、表格逐格整页拷贝消除及缩放访问优化，输出严格回归和组件交替计时见 [通用开销优化](2026-10-29-native-overhead-optimization.md)。下方历史计时保留，不与新的不同时刻记录直接计算整篇提速。

依据用户 2026-10-28 22:41–22:49 的完整日志，比较成功解析的起止时间；Python 为 demo.py 打印的总运行时间，Rust 为 parsing 到 completed。不是同配置基准：原 Python CLI 还输出调试 PDF，缓存状态未控制，先前 Rust 默认单线程，Python 版面/OCR 为 ORT 自动线程、表格/公式为 4 线程。

| 文件 | Rust 原耗时 | Python 原耗时 | Rust / Python |
| --- | ---: | ---: | ---: |
| 论文模板，14 页 | 41.115 s | 19.496 s | 2.109 |
| demo1，13 页 | 100.120 s | 38.132 s | 2.626 |
| 密集长表格，1 页 | 5.269 s | 3.447 s | 1.528 |
| 03，10 页 | 131.952 s 后报错 | 65.281 s 完成 | 不比较失败运行的完成速度 |

### 已定位

1. `recognition_batch` 先用 Rust 稳定排序，再主动拒绝同宽高比跨批改变 padding 的情况。这是迁移时未支持的分支，不是 PDF 输入损坏，也不是 CUDA/磁盘导致。此前首页验收不覆盖整篇跨页正文 OCR 的排序边界。
2. 正式 CPU 默认线程先前固定 1，未与原 Python 的实际默认设置对齐。显式单线程冻结对照仍保留，不能将它当成用户默认速度。
3. 正文 OCR 同一裁图来源被反复 PNG 解码；在 Rust 内缓存当前来源的解码像素即可消除重复工作，不改变透视、旋转或输出裁图像素。
4. IDE 的 stderr 通常不是 TTY，原进度条自动关闭，Python 外层又只保留开始/结束日志，导致长时间没有页数/任务数信息。

### 本轮修复

- 排序调用与当前原 NumPy 2.4.6 一致的 x86-simd-sort 原生内核，固定原上游 revision。AVX2/AVX512 在运行时按 CPU 分派；其他 CPU 使用原 NumPy 标量 introsort。不是改 batch_size、统一放大 padding 或改稳定排序来绕过报错。生产不导入 NumPy，后端为本地静态编译 C++ 库，BSD 许可/来源随 SDK 保留。
- 默认线程对齐原配置：版面/OCR 为 ORT 自动选择，公式/表格为 4；显式 1 线程仍可复现原固定线程参考。
- 正文 OCR 当前来源的像素只解码一次，最多保留一个来源，识别结果仍按原输入顺序还原。
- Rust 发出批次、逐页任务数量、OCR 裁图完成数及阶段总耗时；Python 日志仅转发显示、模型路径和线程数。TTY 原位进度条保留，IDE 仍可见 INFO 阶段信息。

### 证据与边界

独立 native 后端已与本机 NumPy 360 组输入顺序完全一致；AVX512 仅完成编译，本机只有 AVX2。新增 228 组原 NumPy 排序/批 padding 冻结参考，并保留原 12 组 CTC 参考文件（其中原历史 unsupported 项现在纳入结果对照）。用户原 Python 的 03 整篇输出另行保存为新参考，没有修改旧参考。

完整 release 已构建，四个文件共 38 页全部完成。最终 wheel 已安装现有 .venv，安装文件哈希与已验证二进制/适配器/审计一致，pip check 正常。只构建 release、只跑受影响测试，避免清理后重新生成庞大 debug/test 产物。PDFium 共存、印章、CUDA/增强阅读序不在此次修改内；全局质量、峰值内存和正式同条件性能基准仍待第 5 项。

## 最终四文件复测

沿用用户文件顺序与单实例模型复用，使用修正后的默认线程配置；没有与编译任务同时运行。Python 列仍是用户原日志，尚非反复采样、统一输出选项的正式基准。

| 文件 | 原 Python | 修复前 Rust | 修复后 Rust |
| --- | ---: | ---: | ---: |
| 论文模板 14 页 | 19.496 s | 41.115 s | 18.781 s |
| demo1 13 页 | 38.132 s | 100.120 s | 41.731 s |
| 密集长表格 1 页 | 3.447 s | 5.269 s | 4.045 s |
| 03 10 页 | 65.281 s | 131.952 s 后失败 | 62.658 s 完成 |

修复后合计 127.214 s，Python 日志合计 126.357 s；本轮已接近，不能据此宣称 Rust 全面更快。四份整篇 Markdown、middle JSON、content list 均与用户原 Python 已保存输出完全相同，64 个图片资产的模式/尺寸/解码像素也完全相同。03 完成 10 页、271 个正文识别裁图、10,907 个 Markdown 字符、39 个资产。

03 的 Rust 阶段计时：公式 74 项 41.301 s、版面 10 页 8.044 s、检测/裁图 118 项 2.471 s、正文识别 271 项 5.775 s、表格 3 项 3.473 s。原 Python 日志公式约 36 s、版面约 10 s、OCR 检测约 4 s、识别约 8 s。阶段边界并非完全相同，不能简单相减作严格归因；公式处理仍是明显的耗时重点，后续正式性能优化不以降低精度换速度。

受影响验证：360 组原 native 排序对照、228 组排序/批 padding 及实际标量 NumPy 对照、原 12 组真实 CTC 批识别（含原 unsupported 跨批项）的逐图张量位模式/文字、73 个后 OCR 合同和 11 条链、5 个原固定单线程 API 案例、进度与 119 模块兼容审计均通过。未跑全量/CUDA。

早期 scalar_indices 参考只禁用 AVX2，NumPy wheel 的 X86_V3 分组仍会启用 SIMD，因此不能作为标量 oracle。该原文件保留不改；原生顺序/批次字段有效，真正的标量对照另存 recognition-scalar-order-x86v3-disabled.json。最终单元对照通过，未通过改写冻结数据掩盖差异。

日志：build/reported-four-full-test.txt、build/reported-four-summary.json、build/reported-sort-source-test-final.txt、build/reported-legacy_post_ocr-test.txt、build/reported-api_compat-test.txt、build/reported-progress-test.txt、build/reported-compatibility-final-test.txt、build/reported-install.txt。完整结果位于 build/reported-four/。本轮只重建 release，主构建清空缓存后耗时 15 分 37 秒；之后只调整了测试 oracle 与审计元数据，最终包装复用逐字相同的已验证生产二进制，没有重复整库编译。

## 继续定位：为什么仍未全面更快

已测到的重点：demo1 的 Rust 公式阶段 25.087 s，原 Python 进度条约 19 s；03 的公式阶段 41.301 s，原 Python 约 36 s。密集表格的 Rust 表格整体阶段 2.542 s，Python 表格进度条约 1.31 s。Python 为进度条约值且计时边界不完全一致，差额不能直接当作纯推理损耗。

源码能确认的额外工作：公式区域在 crops::prepare 编码为 PNG；pipeline::formulas 调用 validate_png 解码一次，Model::recognize_formula_batch_png 的预检再次完整解码，formula_preprocess::from_png 处理前第三次解码。原 Python 的 BatchAnalyze 跨页收集 NumPy 公式裁图后传入模型，无这些 PNG 往返。Rust 当前还在页内交错调用布局/公式/OCR 检测，原 Python 以跨页阶段编排；这可能影响执行和缓存开销，但尚未做独立计时，不将其写成已证实的具体秒数来源。表格链仍有 serde_json::Value 字符/几何树构造、克隆与重复校验，具体占比也待细分。

Python 的主要模型推理本来由 ONNX Runtime 原生代码执行，PDFium/图像库也是原生实现，换入口语言不会自动加速这些部分。当前额外的格式转换和数据搬运可抵消 Rust 的调度优势。下一步应优先细分公式验证/解码/预处理/ORT 推理/文本解码，改为有界内存图像传递并消除重复解码，再处理表格大树拷贝；以已保存的四整篇严格输出为门槛，不改变算法、图像精度或 OCR padding 来取速度。
