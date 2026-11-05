# PDF 交付收尾：逐项实现与用户测试

按用户指定顺序推进，每完成一项停下，待用户测试反馈后再继续下一项。

| 顺序 | 内容 | 当前状态 |
| --- | --- | --- |
| 1 | 项目现有 Python 环境直接使用 Rust 包，无需切换到专用测试环境 | 环境已完成；用户已测试，解析问题并入第 2 项 |
| 2 | 原 Python API 兼容封装、自动运行库/模型加载 | 默认 PDF API 与共享 Rust 核心已构建安装；5 默认/4 原生案例严格对照通过，待用户测试验收，详见 native-api-core 记录 |
| 3 | PDFium 与 pypdfium2 同进程共存兼容 | 已实现、安装到原 `.venv`，Windows x64 八组同进程及正式双引擎混合验证通过；待用户测试 |
| 4 | 印章识别接入 | 已完成共享 Rust 实现并安装到原 `.venv`；105 组差分、4 页 PDF 印章对照、开关/安装检查通过，待用户测试 |
| 5 | Python/Rust 输出效果、速度、峰值内存对比和相关回归 | 首轮错误及效果差异已修复，随后轻量结构整理也已完成并安装原 `.venv`；12份101页与原Python及整理前Rust的Markdown、两个JSON、图片严格一致，API兼容与相关回归通过。速度/峰值内存未重测，不承诺已达性能目标。下一步可接入Office。详见[完整效果复验](2026-10-30-pdf-parity-fixes.md)及[结构整理验证](2026-10-30-pdf-structure-cleanup.md) |

复选框不集成。增强阅读顺序和 CUDA 的开发/验证仅在用户明确要求时启动，可以放在 Office 集成之后。

## 第 4 项最新交付（2026-10-30）

印章默认开启，专用检测、多边形扩框、曲线展开、识别和回填在共享 Rust 完成；没有印章候选时不加载专用模型。现有 `含印章.pdf` 的印章文字与原 Python 一致，关闭后不加载印章检测器。已安装到原 `.venv` 并核对 148 个包文件，8 组 PDFium 共存回归仍通过。详见 [印章交付记录](2026-10-30-seal-recognition.md)。按约定先停下供用户测试，再做第 5 项完整效果、速度、峰值内存对比；未跑全量或 CUDA。

## 第 3 项交付记录

PDFium 共存已完成本机实现与安装，不再要求两个包分进程使用。详见 [同进程共存交付记录](2026-10-29-pdfium-coexistence.md)。使用独立 PDFium 实例，默认不需要配置；自定义外部 DLL 自动准备私有副本。两种导入顺序、raw、显式路径、环境变量、并行调用、Python 销毁/重建和中文渲染均通过；原 Python 与 Rust 的正式混合解析各自与独立输出一致。

本项按约定停下给用户测试，随后再实现第 4 项印章。下文早期“不能同时导入”的说明属于历史限制，现已解除。最终效果、速度与峰值内存验收仍为第 5 项，不用本轮小范围共存验证替代。

## 第 1 项：现有项目环境运行 Rust

已将本地 Windows x64 abi3 wheel `rapid-doc-rust 0.1.0` 安装到项目原有 `.venv`（Python 3.14.5 x64，原 `rapid-doc 0.9.11` 保留）。仅安装本地 wheel，没有重编译 Rust、联网安装或修改其他依赖。

在项目根目录运行：

```powershell
.venv/Scripts/python.exe demo/rust_pdf_demo.py
```

IDE 继续使用项目原来的 `.venv/Scripts/python.exe`，不再要求切换至 `build/rust-wheel-env`。如果 IDE 原来使用的是其他环境，则需把 wheel 安装到那个解释器环境；安装包不会自动出现在所有 Python 环境中。

可重复安装命令（在需使用的 Python 环境执行）：

```powershell
python -m pip install --no-index --no-deps build/wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl
```

修改 `demo/rust_pdf_demo.py` 中的 `PDF_FILES`、`START_PAGE_ID`、`END_PAGE_ID` 即可测试，结束页为包含边界。示例现已改为 `from rapid_doc_rust import RapidDoc`，自动定位本地运行库和模型；第 2 项仍在修复与验收。本步安装共处不代表 PDFium 同进程兼容已完成：现阶段测试脚本不能同时导入 `pypdfium2`，该限制留到第 3 项处理。

验证：安装前后 `pip check` 均无依赖错误；SDK 确认从 `.venv/Lib/site-packages/rapid_doc_rust` 导入；同一现有解释器加载真实 CPU 模型，运行示例的论文第 1 页并生成 Markdown、JSON 和图片目录，未加载原解析器。日志为 `build/current-python-rust-install.txt`、`build/current-python-rust-smoke.txt` 和 `build/current-python-pip-check-{before,after}.txt`。未重复整篇回归、未重新构建、未运行 CUDA。

## 第 2 项补充验收：用户实际文件和原默认值

早期预览显式启用 `native_only=True`，跳过布局模型，不能代表原默认行为。当前兼容示例使用 `RapidDoc()`：`parse_method="auto"`、公式/表格开启、默认 PP-DocLayoutV3 多边形布局，原生布局只有显式配置时启用。自动分类沿用原规则；部分规则暂在 Python，按用户要求还需迁入 Rust，不能把现有封装视为完整移植。

首次诊断记录如下；各项最新修复/验证状态以 [第 2 项实现记录](2026-10-28-python-api-compatibility.md) 为准，不能据此标记第 2 项完成：

- `demo/pdfs/demo1.pdf`：原 Python 分类为 txt，13 页，前两页分别 3,562/4,111 个原始字符，不是单纯扫描件。Rust 全文原生入口在 `legacy_model/native_lines.rs:237` 触发单页 4,096 字符诊断限制；宿主捕获此错误后，native-only 路由将所有页面作为无原生布局处理，导致空 Markdown 却打印完成。必须让实际失败可见，并按原配置决定模型回退；不能把失败当作成功空白文档。
- `demo/pdfs/49－长表格-密集型.pdf`：原 Python 分类为 txt，1 页、20,198 个字符。原生字符采集先碰到 20,000 字符限制；原生 OCR 规划请求含 882,065 个 JSON 节点、13,568,975 字节（诊断序列化），超过旧单阶段节点预算，在 `legacy_page.rs:585` 报错。文件/整页调用需有明确、受限且独立的容量，不改变原分组、识别、表格算法，不删字符、不截断、不跳过表格；旧小型诊断限制保留。错误载荷保存在 `build/dense-native-ocr-failure.json`，不是有效冻结参考。
- 扫描件必须通过原 auto 分类与版面/OCR 路由实际验收；已有 OCR 内核不等于默认宿主已正确启用扫描流程。
- 默认进度日志需覆盖开始/输入分类、模型定位与加载、当前页/总页数、版面/原生回退、OCR、公式、表格与输出阶段；异常包含文件、页号和阶段，不可仅在整个文档结束时输出“完成”。保留配置日志级别的方式。

诊断日志：`build/api-original-classify-diagnostic.txt`、`build/api-demo1-native-diagnostic.txt`、`build/api-dense-diagnostic.txt`。后续已通过 demo1 完整 13 页与延迟识别器专项，密集表格原生恢复专项通过；新的 Rust 遮罩、进度显示、容量及原 3500 像素渲染修复已统一构建安装。6 文件分类统计及 5 个实际 API 案例严格对照通过（含两份扫描件的首页），详见 `build/api-current-documents-test.txt`。没有运行全量或 CUDA。第 2 项还需完成共享 Rust 核心迁移，再按约定停下交给用户做整项验收。

## 第 2 项最新交付

分类、多边形后处理与正式 PDF 编排均已迁入共享 Rust 核心，正式解析取消固定字符计数上限，独立资源预算可配置。最终安装与验证以 [Rust 核心交付记录](2026-10-28-native-api-core.md) 为准；上文“临时 Python 规则/待下沉”属于较早诊断阶段，不代表目前正式入口仍调用这些算法。开发预览接口继续保留。按约定本项完成安装验证后停下，由用户测试，再开始第 3 项。
