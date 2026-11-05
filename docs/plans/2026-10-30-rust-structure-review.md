# Rust 包与模块结构审查

实施更新：用户已授权并完成第一轮轻量整理。规则/推理目录归组、页面组装规范入口、绑定分文件、旧路径兼容和自有源码排版已落地；12份101页与原Python及整理前Rust的Markdown/两个JSON/图片严格一致，Python公开API未变。详见[实施与验证记录](2026-10-30-pdf-structure-cleanup.md)及[当前目录导航](../../rust/README.md)。下文是整理前的审查快照；整条JSON强类型化、跨格式公共结果模型和包边界重划没有在此轮混做。

日期：2026-10-30。范围：现有 Cargo 清单、源码组织和正式入口调用关系；不修改算法、包或文件位置，不进行构建/性能测试。以下为审查建议，不是已批准的重构决定。

## 结论

顶层分包基本合理，包内组织仍有明显迁移期痕迹。无需照抄 Python 目录，也无需把所有包合成一个；优先解决领域代码集中在 legacy_model、动态 JSON 内部契约、职责交叉和绑定文件集中。结构问题影响维护、跨语言 API 和构建依赖面，但不能据此断言现有解析慢的原因，更不能承诺移动文件后会提速。

## 实际分布

统计仅包含各包 src 下 Rust 文件，排除 vendor、tests 和 experiments；行数包括注释、数据表及条件编译内容，不代表算法复杂度。

| 包 | 文件数 | 行数 | 当前职责 | 判断 |
| --- | ---: | ---: | --- | --- |
| rapiddoc-core | 6 | 693 | 事实/几何类型、错误、限制、进度、Pillow 像素算术 | 基础层方向合理，但 PDF 专属事实及图像实现混在通用命名下 |
| rapiddoc-pdf | 26 | 7,959 | PDFium、PDF 字符/矢量/字体/图片读取渲染 | 保留独立包合理 |
| rapiddoc-native | 17 | 8,480 | 原生 PDF/文字/表格规则、DocVortex 校验适配 | 有独立价值，native 命名需说明是原生事实规则，不是“所有 Rust 实现” |
| rapiddoc-inference | 43 | 10,253 | ONNX 会话、模型前后处理、OCR/版面/公式/印章及表格处理 | 包边界可以保留，包内应按 runtime、OCR、layout、formula、table、seal 归组 |
| rapiddoc-render | 4 | 697 | HTML 表格输出、结构组装及候选质量选择 | 很轻；纯输出和业务选优混合，且不覆盖全部 Markdown/JSON 输出 |
| rapiddoc | 219 | 82,639 | 正式流水线和大量文字、版面、图形、表格、输出规则 | 承担过多细节，最需要包内按领域整理 |
| rapiddoc-cli | 1 | 85 | 当前为原生文字提取/能力查询 CLI | 独立二进制包合理；目前不等同于完整模型 PDF 流程 |
| bindings/python（rapiddoc-python） | 3 | 2,681 | PyO3 适配、底层差分/预览接口、正式 PdfEngine 入口 | 正式 pipeline 适配很薄，但 lib.rs 集中了 2,570 行和 85 个 pyfunction |

项目内直接依赖形成有向无环关系：core 为基础；pdf/native/inference 在其上；render 还依赖 native；rapiddoc 汇聚这些包；CLI 和 Python 绑定向外提供入口。vendor 中 DocVortex、lopdf 等属于第三方来源边界，不能按文件数量视为需要合并的业务包。

## 主要问题

### 1. legacy_model 已经超出“模型兼容层”的职责

其目录包含 194 个文件、73,474 行，约占 rapiddoc 包源码行数的 89%。模块同时覆盖标题、段落、参考文献、跨页文档、原生表格、图形仲裁、公式和最终输出。`legacy` 表示移植原 Python 行为，并不表示这些代码已经过时或可删除。

建议先在原包内整理为 layout、text、table、graphics、document、output 等领域目录，保留旧模块重导出和操作名，逐步迁移调用者。不能按单个文件数量机械合并，也不要新建一批对应这些目录的 crate。

### 2. 正式流水线仍以动态 JSON 作为内部数据契约

pipeline 的 Page.facts、Page.plan 和最终 metadata 都是 serde_json::Value；operations::run 通过字符串操作名分发。这样方便与原 Python 中间 JSON 做逐项对照，但字段错误在运行期暴露，公共契约不容易由编译器和 IDE 检查，也妨碍明确所有权和跨语言稳定 API。

并非每次 op 调用都序列化字符串：op 直接传 Value；但 wire 明确执行序列化后再解析，以保持旧 FP32/十进制往返语义。不能简单删除 wire 或把所有字段改成 f32 来追求速度。

建议先给新稳定接口引入明确的页面、版面区域、OCR 任务、表格、资产和输出类型，在边界保留 JSON 适配。既有行为逐段迁移并验证数值契约，不一次性替换整条流水线。效果和耗时收益需要单独测量。

### 3. 推理、表格规则、输出的边界不够清晰

inference 中 table_match、wired_match、table_lines 等包含不依赖 ONNX 会话的后处理；render 的 quality 包含候选选优，实际 Markdown/内容 JSON 输出又在 rapiddoc/legacy_page/output.rs。render 因 Table 类型定义在 native/grid.rs 而依赖具体规则包。

短期先明确包内目录与文档，不急于移动所有实现。后续可以让通用表格结果类型进入 core 的独立领域模块，规则/模型产生该类型，输出层消费该类型；候选选优留在业务规则/调度层。是否保留独立 render 包，以形成统一输出职责后再决定。

### 4. 生产入口与调试/预览接口混在绑定层

bindings/python/src/pipeline.rs 只有 91 行，已将正式调度交给共享 Rust Engine，这是正确方向；但 lib.rs 同时集中底层模型调用、几何检查、预览和差分入口。Python SDK 也仍保留较大的旧预览入口。

建议先将绑定实现按 pdf、inference、pipeline、diagnostics 拆成文件，保持导出名称、参数和行为兼容。调试协议不应成为未来其他语言绑定都必须实现的正式业务 API。无需再写其他语言版本的 OCR 或后处理算法。

### 5. 为 Office 复用准备的数据类型还不够明确

core/document.rs 的 Document/Page 主要是 PDF 事实，包含 PDFium 字符、原生坐标、路径等，不是最终格式通用的文档结果。Office 不宜被迫伪造 PDF 字符/矢量来复用它。

建议区分格式专属事实与通用文档结果，可先在同一个 core 包内用模块表达。PDF、Office 各自产生事实/区域，复用后续通用布局、段落、表格和输出类型；不为尚未实现的每种格式预先增加 crate。

## 建议顺序与约束

1. 按原交付顺序完成第 5 项效果、速度、峰值内存验收，冻结 PDF 基线。
2. 先做文件归组、命名说明、绑定分文件及兼容重导出，保持算法和数据完全不变。
3. 在 Office 接入前确定格式事实/通用结果边界；动态 JSON 转强类型按领域分步完成，不与算法优化混做。

重构验收应继续检查 Markdown/JSON、资产像素、模型 token/数值边界及峰值内存；只跑受影响测试，阶段末构建/回归。移动文件本身不会自动提升模型推理速度，也不保证缩短依赖变更后的上层重编译时间。本轮没有执行任何重构或新增测试。

## 二次核查：小文件、单文件目录与 Office

### 文件大小不是当前的主要问题

7 个 Rust 包 src 目录共 316 个 `.rs` 文件，其中少于 60 行的只有 11 个（约 3.5%），少于 100 行的有 28 个。统计排除 vendor、独立 tests 目录和绑定层，但包含 src 内的数据及测试代码。这不足以支持“普遍拆得太碎”的判断。

检查后，小文件主要有明确用途：

- `rapiddoc-core/src/ocr.rs` 10 行，定义独立的 OCR TextLine 公共契约。
- `rapiddoc-inference/src/pillow_resize.rs` 2 行，转导出 core 中共享实现，没有复制算法。
- native 包多个 2～11 行文件提供固定上游规则的转导出/适配边界。
- `legacy_model/table_native_vector/pipeline/digit.rs` 12 行，是生成的 Unicode 数字分类常量，不是 12 行业务逻辑再套一层调度。
- `filled_grid/number.rs` 76 行，集中处理 Python int/float 面积比较与有理数边界，有独立数值契约，不宜为了减少文件数并入普通表格流程。

没有依据批量合并这些文件。单次调用且没有独立职责的辅助函数，可以在后续逐域整理时考虑内联或合并，但不应仅按行数定名单。

### 目录只有一个文件，不等于只有一个模块文件

共找到 11 个仅含一个 Rust 文件的目录，它们都在目录外有对应的父模块 `.rs` 文件。另有一个只存 JSON 数据的目录，单独作为资源看待。

例如 `seal.rs` 是 287 行的印章主模块，其 `mod crop` 对应 `seal/crop.rs`（712 行）；`form_structure.rs` 是 546 行，子目录中 `declarations.rs` 是 186 行。这些是父模块文件与子模块目录并列的正常组织，不是空目录层级，也不能只看文件夹内部就判为过度拆分。

### 更明显的问题是排版和领域定位

部分短文件使用了压缩排版：`layout_contour.rs` 虽然仅 62 行，却有约 6,807 个字符；`layout_geos.rs` 87 行、约 7,533 个字符；PDF classification/probe.rs 的 34 行中也包含多段循环、几何计算和探测流程。它们不等于“只有一点代码”。单行多分支、多循环使阅读、差异审查和报错定位更困难。

应在受控的格式整理中恢复常规排版，避免因视觉行数少而继续合并。配合前述 legacy_model 按领域归组、缩短冗长迁移名称；对于只剩一份数据文件的很深路径，可在兼容重导出下调整位置，不值得单独发起大重构。排版、移动文件和算法变更应分开验证。

### Office 尚未集成，不能把 PDF 阶段结构当成最终结构

已核对原 Python 的 `backend/office/office_analyze.py`：现有主入口按格式调用 Doc/Docx/Xls/Xlsx/Ppt/Pptx 原生分析器，再经 Office 自己的中间数据转换和输出流程处理。不能为复用 Rust PDF 流水线，改成全部 Office 先转 PDF 或 OCR，也不能认为现有 PDF Document/Page 已经覆盖 Office 的契约。

Office 接入应继续对照当前 Python 各格式实现，保留其列表、段落、表格、图片、公式等行为。PDF 事实读取和 Office 结构读取保持独立；只对已有等价语义的类型、资产、工具及输出能力复用，Office 特有行为不强行并入 PDF 页面组装器。

建议依旧是：PDF 验收后冻结参考，在 Office 入口设计前做一轮轻量领域/契约整理；不等全部 Office 写完才处理耦合，也不现在重划所有 crate 或批量合并小文件。未来是否需要独立 `rapiddoc-office` 包，应根据真实依赖与实现体量决定，目前不创建空包。
