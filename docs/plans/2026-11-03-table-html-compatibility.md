# 表格 HTML 容错与 Unsupported 检查

## 实际问题

`page-50ffe5d2-cd27-449a-a813-d4a8c44b12ea.pdf` 的 SLANet 原始输出以 `<td</table></body></html>` 结束。UNET 候选为空，组合模型比较器进入 BeautifulSoup 语义的空单元格统计。Python 接受这个不规范标签（解析成 `td<`，附带空的 `table` 属性）；Rust 原本要求标签名只含字母数字等字符，提前返回 Unsupported，因此没有机会进入 TATR 备用。

这不是 PDF 内容不支持或 GPU 故障，也不能要求用户修改文件规避。

## 修复范围

- 仅在 BeautifulSoup 兼容模式处理起始标签里的异常 `<`、分隔斜线和不完整尾部、`< 2` 等字面小于号，以及残缺结束标签。严格跨页合表模式保持原校验。
- 候选 HTML 解释阶段的 Unsupported/InvalidInput 标记为候选拒绝，配置的备用模型仍能执行；没有扩大到模型推理、会话加载、设备或资源错误。
- 生产元数据保存候选拒绝原因，Python 兼容层打印告警。关闭备用模型与文字回退时，不擅自启用这两种功能，无法解释的候选按未识别处理。
- 不修补某个文件名、文字内容或某一张表；使用独立 Python 的 BeautifulSoup、原 normalize/selector 作行为对照。

## Unsupported 检查边界

静态检查了 `rapiddoc`、`rapiddoc-inference`、`rapiddoc-pdf` 中的 Unsupported 调用点；原始清单位于 `build/pdf-unsupported-audit.txt`。调用点数量不是缺失功能数量，多个分支用于同一个输入合同，且包含测试及未进入生产的诊断接口。

| 类别 | 示例 | 处理 |
| --- | --- | --- |
| 本次确认的普通输出兼容缺陷 | 不规范 SLANet 标签在候选比较时导致整页失败 | 修复容错及候选拒绝处理 |
| 模型接口校验 | 缺少 character 字典、张量形状错误、错误 tokenizer 类型 | 保留，不能用空结果掩盖配置或模型错误 |
| 明确未启用/延期能力 | Plus-L 完整流程依赖 XY-cut；VL OCR；额外输出模式 | 保留，并继续明确支持边界 |
| 内部像素/运行库合同 | OCR 内部 RGB PNG、GEOS 缺失 | 保留；生产会先转换像素，若正常文档触发，按调用路径诊断 |
| 诊断入口限制 | 非矩形 poly、带 polygon_points 的旧诊断请求 | 生产 host 已有多边形路径；不应仅凭报错文字认为生产缺功能 |
| 仍需关注的文档容错 | 跨页合表的严格 HTML、异常跨度、嵌套行，少见内联 WMF/EMF | 未证明全部兼容；后续按真实触发补测试，不能声称已清零 |

不得将所有 Unsupported 全局 catch 后返回空值，否则会把模型缺失、结果丢失伪装成成功。现有全量对照的历史报告包含本次失败，不因修复改写旧评分；后续重测单页与必要的受影响用例。

## 验证结果

- 95 项受影响测试通过：包含 11 类不规范 HTML 的 Python 真实 normalize/selector 对照，以及表格模型、OCR 规则、文字回退与安装后输出回归。
- 1 项原生错误边界测试通过：候选解释错误可恢复，ResourceLimit / ModelMissing / DeviceUnavailable / RuntimeUnavailable 必须传播。
- 实际采集的 SLANet 原始 HTML 和两模型候选比较结果，Rust 与独立 Python 的处理一致。
- 报错 PDF 在最终安装包的 CPU 完整链路成功解析，公式关闭、UNET+SLANet 主模型、TATR 备用、docvortex_ocr；输出 4323 Markdown 字符和一张表。证明本次报错修复，不声称整份 Markdown 与 Python 逐字一致，也不替代识别精度评测。
- release wheel 已安装到项目 `.venv`；未跑全量或 CUDA，未重写服务器既有 A/B/C/D 结果。服务器仍是此前评测版本，本次尚未同步。

证据：`build/table-html-selection.json`、`build/table-html-raw.json`、`build/table-html-fixed-final.log`、`build/table-html-rust-fixed-final.md`、`build/table-html-candidate-tests.log`。

## 后续 CUDA 单页修复对照

用户随后授权同步服务器并补测 A/B/C 的失败页。复用现有源码及构建缓存安装修复版后，该 PDF 在三组均成功：A 生成 3818 字节 Markdown，B/C 各 4327 字节。公式开启，其他参数与原三组相同，未重跑其余 384 页。

服务器证据在 `table-abc-20261103/htmlfix/`，本机位于 `D:/Download/OmniDocBench/table_abc_20261103/htmlfix/`。已逐文件核对每组仅原失败 PDF 发生变化；新预测为 `table_abc_20261103_{A,B,C}_htmlfix`，新评分为 `result_table_abc_20261103_{A,B,C}_htmlfix`。保留原始 A/B/C 评分及 D 组结果；新评分使用原固定 385 页标注。
