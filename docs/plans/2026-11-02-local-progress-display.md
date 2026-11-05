# 本机阶段日志与原生进度条

阶段累计时间的后续修正见 [计时说明](2026-11-02-progress-active-time.md)，
最新本机安装包为 `build/progress-wall-clock-windows-wheels/`；进度条显示
真实经过时间，累计处理耗时单独汇总。

旧显示存在两个问题：只拿页索引画进度，而公式、OCR、表格的任务单位
不同；非 TTY 的 IDE 控制台会关闭进度条，退化为逐页 INFO 日志。

## 调整

- 原生流水线事件提供独立的 `completed` / `total`，表示已经完成的工作。
  Layout 按窗口页数、公式按公式裁图、正文 OCR 按文字裁图、表格按候选
  表格、印章按区域、页面组装按页计数。未开始的任务不冒充已经完成。
- Rust 保持各阶段计数与起始时间，输出接近 tqdm 的百分比、进度条、
  完成数/总数、已用时间、预计剩余时间和处理速率。阶段完成行保留，
  不会在切换阶段时擦掉；跨窗口重新计数。
- TTY、PyCharm 控制台支持原地更新；普通重定向输出使用独立文本行，
  起始/结束必显，中途最多约每五秒一条，避免日志文件大量回车字符。
- 模型加载日志前暂停进度条，之后继续原阶段，避免条形图与 INFO 串行。
- 批次显示 `Batch i/n`、页数、实际 TXT/OCR 路线；原生分析单独提示。
  后续按原 Python 显示方式，不再单独打印 `Post-processing...` / `write...`，
  完成日志包含总耗时和本地输出目录。
- 默认去掉逐页 prepared 行与几十条细分计时；需要诊断时可设置
  `RAPIDDOC_DETAILED_LOGS=1`，通过 DEBUG 日志保留这些详细信息。

不添加 Python tqdm 依赖；显示与计数在 Rust，Python 仅转交事件与普通
日志。公式增加观察回调，不改变模型、张量、排序、batch_size、阈值
或解码方式；表格和 OCR 同样只报告已有工作。

## 可选控制

默认无需配置。`RAPIDDOC_PROGRESS=always` 强制原地更新，`plain` 强制
独立文本行，`off` 关闭进度条但保留正常日志。这是显示模式，不影响
识别参数。ETA 是按已完成工作估算，不是固定承诺。

本轮仅在 Windows 本机验证及安装，不修改服务器环境。安装包与本机
验证结果位于 `build/progress-ui-windows-wheels/`、`build/logging-after.log`。

## 本机结果

- 原生计数/ETA/速率格式测试通过；四项 Python 回归通过，覆盖 PyCharm
  管道原地更新、普通重定向不含回车、阶段完成行只保留一次、多窗口重置、
  未完成任务不显示 100%、关闭显示及既有评测失败日志。
- 论文第 6–7 页以窗口 1 解析，验证两个批次和 5 个公式；长表格验证
  `Table Predict`。两组修改前后完整 snapshot 一致，包括 Markdown、
  中间结构、内容列表和 5 项资产，见 `build/logging-output-parity.json`。
- 模拟 PyCharm 的真实两页方向矫正 OCR 显示 `OCR-rec Predict` 的
  `26/26` 完成行，正文保留，见 `build/logging-ocr-ide.log`。
- wheel 已安装项目 `.venv`，`pip check` 通过；不需要修改调用参数。
  这轮没有重跑全量回归、CUDA 或服务器测试。此为显示调整，不是速度评测。

后续对齐原 Python 绘图策略：`_drawing` 将 pypdf 日志级别设为 ERROR，
隐藏 `Ignoring wrong pointing object...` 等警告，保留错误。使用含错误
xref 的小型 PDF 验证警告确实被抑制、页面仍可读取、错误仍被记录。
五项日志/绘图回归及真实 PDF 的 layout/span 两种可视化导出通过；
实际输出不再包含上述两行阶段提示，进度条和输出目录仍保留。
最新本机安装包位于 `build/progress-python-style-windows-wheels/`，
验证日志为 `build/progress-python-style-smoke.log`。本次复用原生编译
产物，只重新打包 Python 兼容层；未改变 PDF 算法或服务器环境。
