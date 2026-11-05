# 原生 PDF 与默认宿主链整合

当前按用户指定的五项顺序收尾：①现有 Python 环境运行 Rust；②原 Python API 兼容封装与自动运行库/模型加载；③PDFium 同进程共存；④印章；⑤Python/Rust 效果、速度和峰值内存对比及回归。每项完成后停下供用户测试，进度见[逐项交付记录](2026-10-28-pdf-delivery-steps.md)。复选框排除。增强阅读顺序、CUDA 开发和实测均仅在用户明确要求时启动，可以安排在 Office 集成之后，不作为当前 PDF 收尾或进入 Office 的前置条件。

继续迁移现有 `docvortex_adapter.py` 和 `RapidLayoutModel.batch_predict`，不另写解析规则。原生块像素转换、图表/公式补充的稳定替换、完整回退判定和区域文字方向在 Rust 执行；SDK 只编排模型和资产。保留原 `off/rule/rule_plus_s` 语义、native-only 的空页行为，以及 OCR 模式移除原生文字/表格预填的规则。

同一 `parse_pdf_models` 入口接入整篇原生结果，原生或模型选择后均继续现有裁图、公式、OCR、表格、页面组装、段落/跨页表格及 Markdown/资产。默认 off 保持既有 Python 配置，原生通过参数启用；原生-only 可不传布局模型，模型回退和补充模型按实际任务延迟要求。内置表格恢复器复用现有完整原生表格引擎，按页缓存脱离句柄的可见字符/矢量原语，保留方向推断、恢复错误、置信度和公式/图片守卫。

验证先冻结现有 Python 适配器/宿主连续输出，开发只跑受影响测试；本阶段结束统一构建安装。真实两份示例需从同一入口产出 Markdown、content_list 和可解码资产；不得把冻结布局模型响应当作实际模型推理。质量对照与性能测量分开记录，增强阅读顺序和 CUDA 继续后置。

| 原 Python 逻辑 | Rust/SDK 接入 |
| --- | --- |
| `convert_docvortex_layout`、`_block_text`、图注类型/旋转/忽略标签、HTML 与文字 owner | `native_adapter::convert`；保留原像素裁剪、空框跳过和原始 order |
| `docvortex_layout_fallback_reason` | `native_adapter::fallback`；保留空布局、过大丢弃框、80 字符门槛和单图反证 |
| `merge_docvortex_supplements` | `native_adapter::supplements`；稳定分数/位置平局、原槽替换、对象身份消费、公式编号与插入次序 |
| `_table_text_angle` | `native_adapter::direction`；视觉角、四方向接近度、Counter 首次平局 |
| `prepare_table_page` / `recover_docvortex_table` | 原生可见字符/矢量采集、原规则/矩形转换、完整表格恢复及语义 HTML；`NativePdfTableRecoverer` 逐页缓存并保护并发 |
| `RapidLayoutModel.batch_predict` 模式/回退/补充及 OCR 清除原生预填 | SDK `_native_pdf.choose_native_layout` 与 `parse_pdf_models`，算法调用上述 Rust 函数 |
| `get_res_list_from_layout_res` 原生文字和 HTML 预填 | 现有 `crops.rs` 开放原 `native_text`，后续已有 OCR 直接注入和 HTML 跳过表格模型分支继续复用 |

新接口示例：

```python
result = parse_pdf_models(
    "input.pdf", docvortex_mode="rule", native_only=True,
    detector=detector, recognizer=recognizer, formula_model=formula_model,
)
# result: pages / markdown / content_list / assets，资产仍由调用方决定写出位置。
```

仍按实际任务提供模型：native-only 不要求布局模型，原生 HTML 表格不调用表格模型；不可用文字会继续现有 OCR。`rule` 的模型回退需要 `layout_model`，`rule_plus_s` 公式补充需要提供适配既有 `layout_png` 协议的 `supplement_layout_model`。本轮保留 SDK 原来的印章/复选框范围，不把补充仲裁规则当作实际 S 模型/印章推理验收。

预览脚本支持 `--docvortex-mode rule --native-only`，会加载实际 CPU OCR、公式与表格会话，跳过未使用的布局模型。`native-host-document.json` 的五页文件正反文字模式使用真实 CPU OCR/SLANet，公式固定 x+y；`native-host-samples.json` 则对论文/财报完整 23 页使用真实 CPU 公式和 OCR，另有 19 个原资产像素参考。后者用于完整 Markdown/JSON/资产对照，不能把前者的固定公式响应解释成实际公式推理。

模型专用旧冻结参考在原 Python 中明确关闭了 rule-table；受影响回归传同样的 `use_rule_table=False`。新的默认 `use_rule_table=True` 对照使用内置 Rust 恢复器，不再用冻结响应提供器替代后端。所有冻结 JSON 保留。

整篇严格对照发现并修复三个差异，未修改参考：像素坐标裁剪保留 Python `min/max` 选择的整数/浮点类型；不透明 RGB 页面使用原 BGR 渲染格式，避免 BGRA 抗锯齿舍入改变像素 MD5 和全部图片路径；正文 OCR 按原宿主跨页收集后一次批量识别，避免逐页批次改变补齐宽度和 CTC 文本。财报第 3 页原 BGRA 路径仅差 1 个像素，仍会影响图片路径，不能通过忽略文件名解决。跨页准备增加 128 MB 压缩输入预算，超过明确报错，不返回部分文档。

重新 release 构建（8 分 51 秒）并安装至 `build/rust-wheel-env` 后，以下受影响验收全部通过，不依赖源码目录加载 SDK，也不导入原 Python 解析器。

| 安装验收 | 结果 |
| --- | --- |
| 原适配器及区域表格 | 257 个严格类型/所有权对照、8 个脱离句柄页面、13 个实际表格恢复正例，含共享缓存并发 |
| 完整原生宿主 | 五页文件分别原生文字/强制 OCR，共 10 页的 JSON、Markdown、content_list 和解码资产一致；实际 CPU OCR/SLANet，公式响应固定 x+y；内置默认表格后端、回退/补充及错误门禁通过 |
| 两份用户示例 | 论文 14 页/7 资产/8465 字符 Markdown；财报 9 页/12 资产/21790 字符 Markdown。共 23 页/19 资产，实际 CPU 公式/OCR，JSON、Markdown、content_list、资产名、模式、尺寸与像素全部一致 |
| 页面渲染 | 18 RGBA + 18 RGB，原像素及限制/错误恢复通过 |
| 受影响旧路径 | legacy_model、model_document、model_ocr_execution、native_table_media_png 全部通过；保留真实 V3 PDF 冒烟、模型/原生表格回退及表内图片/ICC/透明色测试 |
| 兼容性来源/默认门禁 | 118 模块 hash 和原默认门禁通过；不提升全局精度/性能声明 |

最终日志：`build/native-host-final-wheel-build.txt`、`build/native-host-final-wheel-install.txt`、`build/native-host-final-native_host-install.txt`、`build/native-host-final-native_host_samples-install.txt`、`build/native-host-final-render-install.txt`，其余受影响套件均为 `build/native-host-final-<suite>-install.txt`。两项相关源码库测试也通过；早期差异/中断编译日志保留，以 final 日志为本阶段安装证据。未跑全量或 CUDA。

可查看结果：`build/rust-native-host-preview/示例1-论文模板/document.md` 和 `build/rust-native-host-preview/比亚迪财报/document.md`，同目录有 `result.json` 和 `images/`。本轮宿主接入与这两份文档的整篇验收已完成；后续仍需更广 PDF/扫描件语料、实际补充模型/印章宿主范围和相同条件耗时/峰值内存测量。增强阅读顺序及服务器 CUDA 实测继续延期，不能宣称所有 PDF 均已等价或性能更优。

用户确认的剩余范围：复选框不集成、不作为 PDF 交付阻塞项；印章必须按原 Python 流程接入。Python/Rust 速度与内存对比为必做验收，先在本机 CPU 进行，使用相同 PDF、模型哈希、DPI、线程、原生/OCR/表格/公式开关和输出范围，分别记录初始化、解析与总耗时，冷启动与会话复用结果分开；独立进程记录整个运行的峰值工作集/私有内存，不能仅用 Python 分配器追踪代替原生内存。先确认输出一致，再解释速度比、内存差异及重复运行波动，不预设 Rust 更优。其余主要工作为已有模型分支/错误回退与整篇语料验收，不重复把已完成的正文、表格、公式或 Markdown 列为待移植。

两份示例也可使用同一实际模型入口运行，输出默认写入新的时间戳目录，不覆盖已有结果：

```powershell
build/rust-wheel-env/Scripts/python.exe -I scripts/migration/preview_pdf.py "demo/pdfs/示例1-论文模板.pdf" "demo/pdfs/比亚迪财报.pdf" --docvortex-mode rule --native-only
```

`off` 继续原模型布局模式，仍是 API 默认值；完整原生链不是默认打开实验性排序。`rule_plus_s` 的适配器和补充仲裁有原参考/协议测试，本轮两份示例实际验收的是 `rule + native_only`，不把协议测试当作真实 S 模型或印章识别验收。
