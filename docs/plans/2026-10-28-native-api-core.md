# 第 2 项：正式解析限制与 Python 核心代码清理

用户已授权直接实现。保持原 Python 算法及精度，不能删除功能、改用矩形或简化分类来减少代码。开发只跑受影响验证，完成后统一 release 安装；当前 release wheel 已更新项目现有 `.venv`（Python 3.14）；先经过独立开发 wheel 定位，再统一 release 构建与安装。

| 内容 | 当前结果 |
| --- | --- |
| 正式调用固定字符门槛 | 正式 PDF Limits 使用无字符计数上限的 `pdf_host()`；宿主记录容量不再固定 100,000，诊断默认保留。真实多对象 PDF 的 100,058 字符已完整采样。 |
| 独立资源控制 | Rust `HostResources` 可经 `RAPIDDOC_RESOURCE_LIMITS` JSON 配置 work、JSON 节点/字节、输出字节及候选成员预算；作用域、异常恢复和线程隔离通过。资源配置不接受 max_chars。其他像素/输入/对象遍历限制没有声称全部取消。 |
| 文本/扫描分类 | 完整判断链迁入 `rapiddoc-pdf::classification`，含原乱码信号、CID/Latin 字体资源与真实文本绘制探测。140 规则、36 合成 PDF、6 用户文件的分类/41 页统计通过；不导入 pypdf、NumPy、OpenCV、Shapely 或原解析器。 |
| 多边形后处理 | 迁入 `rapiddoc-inference::layout_polygons/layout_contour`，原 OpenCV 闭合近似、定制顶点和 FP32 最小矩形算法保留；与原 Shapely 相同的 GEOS 3.13.1 动态库由 Rust 直接调用。56 auto/poly/quad/rect 输出严格匹配。完整掩码留在 Rust，不再经大 JSON 转给 Python。 |
| 解析编排 | 正式 `RapidDoc` 已调用共享 Rust `pipeline::Engine`，公式/正文 OCR 跨页批处理、原生表格守卫与恢复、方向/文字/双模型仲裁、图片资产和后置处理均在 Rust；模型由 Rust 注册表持有、按需加载与复用。Python 回调只提供资源描述符和进度。低层 `parse_pdf_models` 预览接口暂保留，不由正式入口调用。 |

从发布源码移出 `_classification_rules.py`、`_classification_probe.py`、`_classifier_bridge.py`、`_layout_rules.py`，参考副本留在 `scripts/migration/references/`。分类 Python 文件仅保留字节、范围及原生结果适配。SDK 取消对 NumPy/OpenCV/Shapely 的直接依赖；ONNX Runtime 自身的传递依赖独立存在。调试绘图及 Python API 输入/输出适配尚保留 Python 实现，没有宣称所有 Python 文件都应删除。

GEOS 运行库位于 `runtime/geos/`，保留原二进制名字/哈希、LGPL 与 MSVC 许可、对应原始源码归档及替换路径 `RAPIDDOC_GEOS_LIBRARY`。不导入 Shapely，也不把 GEOS 算法改写为近似相交。当前已验证 Windows x64；其他平台运行库需按该平台打包，不能据本机测试宣称全部平台通过。

独立开发包验证：6 文件分类统计及 5 个实际默认 API 案例全部通过，Markdown/JSON/图片像素严格匹配原 Python，包括 demo1 前 2 页、密集表格、small_ocr/03/论文模板各首页。日志：`native-classification-source-tests.txt`、`layout-polygons-source-test.txt`、`host-resources-source-test.txt`、`native-core-classifier-install.txt`、`native-core-resources-install.txt`、`native-core-api-documents.txt`（均在 build/）。没有改写已有冻结参考，没有全量/CUDA或速度/内存提升声明。

本轮开发包已通过正式 Rust 编排的 5 个默认 API 案例（6 页）及 6 文件分类统计，测试显式禁止 Python 预览编排和 LazySession 模型执行。原生布局论文首页（207 字符/1 资产）与整页扫描首页（841 字符）严格对照通过。扫描整页暴露的识别裁图源图预算已与输出裁图预算分离，透视坐标、图像尺寸、像素和 400 万输出裁图限制保留；定向单元验证源图/输出限制独立。

分页的原生文档统计改为所选页集合，与原 Python 的先选页再分析顺序一致；原页索引用于图像和文字绘制采集。模型资源和进度回调的 Python 异常类型保持。最终 release 已安装；4 个原生 API 案例共 5 页严格通过：原生论文首页、扫描首页、demo1 前两页逐页批次、论文非首页。包含资源/进度回调原异常类型验证。

资源配置示例（可选，不设置则使用默认独立资源预算）：

```powershell
$env:RAPIDDOC_RESOURCE_LIMITS='{"max_json_nodes":8000000,"max_json_bytes":64000000,"max_output_bytes":64000000}'
.venv/Scripts/python.exe demo/rust_pdf_demo.py
```

该配置作用于单次宿主调用并在返回/异常后恢复，不是字符截断开关；字形及原生采集不会因字符计数达到固定门槛停止。已有固定诊断接口默认限制保留。模型注册表以运行库、模型路径、哈希、设备、线程数及张量预算为键共享会话，描述符只在首次实际需要该角色时解析，不按每页重复加载。

## 最终安装验证

- 现有 `.venv` 从 `site-packages/rapid_doc_rust` 加载新 release，`pip check` 正常；4 个临时 Python 算法文件均已从安装包移除。
- 正式 API 的 5 个默认案例（6 页）与 4 个原生案例（5 页）通过原 Python 冻结参考的 Markdown、JSON、资产像素严格对照；正式入口禁用 Python 编排/模型执行仍通过。
- 36 份分类 PDF、真实 100,058 字符采样、资源配置/异常恢复、进度条、119 模块来源审计均通过。257 适配器参考、8 页脱离句柄、13 个原生表格恢复正例及 2 篇 10 页原生宿主文档通过。
- 安装与验证日志：`build/native-api-core-install.txt`、`build/native-api-core-pip-check.txt`、`build/native-api-core-{api_compat,native_pipeline,host_resources,native_classifier,native_host,compatibility,progress}-test.txt`。仅受影响验证，未跑全量/CUDA。

本轮交付后按约定停下等待用户实际测试。PDFium 同进程共存、印章、质量/速度/峰值内存对比仍是第 3/4/5 项；不提前进入增强阅读顺序或 CUDA。低层开发/预览 Python 编排保留兼容接口，正式 `RapidDoc` 已不调用；尚未验收的自定义模型配置继续明确拒绝。Windows x64 以外的平台不在本次验收范围内。

用户报告的 demo1 整篇完成性回归也通过：13 页、51,140 个 Markdown 字符、18 个图片资产，全文 Markdown 与此前已安装 Rust 版本逐字一致；本轮全篇运行禁用 Python 预览编排和 LazySession 执行仍通过。日志 `build/native-api-core-demo1-full-test.txt`，结果 `build/native-api-core-demo1-full/{document.md,summary.json}`。这是报告问题的完成性/迁移稳定性验证，不替代第 5 项的原 Python 整文档质量与同条件性能对比。

用户随后提供了 03 整篇跨批失败和性能日志：该漏项、默认线程不一致、重复 PNG 解码及 IDE 缺日志已修复并更新安装；四文件 38 页的三类文本/结构输出与原 Python 完全一致，64 图片像素一致。详细复测与尚未达到全面性能优势的边界见 [用户实测与修复](2026-10-28-reported-speed-and-ocr.md)。
