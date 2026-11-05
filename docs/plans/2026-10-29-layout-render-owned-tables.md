# 渲染像素复用及表格恢复请求所有权

目标：移除已经确认的内部搬运，不修改模型、阈值、批量、像素运算或表格判定。

- PDF 渲染共享原有完整参数校验、白底 BGR PDFium 渲染、RGBA→RGB 丢弃 alpha、3500 长边上限和 PNG 编码。新增入口同时返回编码前的原 RGB Vec，公共 PNG 接口继续沿用同一实现。
- 正式版面调用直接借用当前页的 RGB Vec，交给共享 RGB 版面入口；原 cubic/FMA、800×800、FP32 缩放、三命名输入、原 masks/GEOS 后处理不变。原 PNG 入口的公共解码器返回 BGR，随后原有通道交换将它还原为 RGB，不能在新 RGB 入口额外交换通道。PNG 仍用于其他后续资产与裁图，不能称整页 PNG 已全部消除。
- 原始像素在当前页版面阶段结束立即释放，不进入 16 页窗口 Page 缓存。复用的是原渲染缓冲区，可能保留 RGBA 的容量；本轮没有测定进程峰值内存，不宣称峰值下降。
- 表格恢复按借用/拥有两种存储进入同一内核。公共 Rust 借用入口保留写时复制；正式调度及已经拥有 JSON 值的共享操作分发使用拥有入口，省去一次完整请求深拷贝。恢复结果通过 take 移出，省去再复制 glyph/cell 结果树。初次将整页几何装入每表请求的复制仍存在。
- 原树校验、host_page 预算、同一 Work、引擎错误包装、渲染错误及 HTML/恢复字段不变。Python 包没有加入算法或文档分支。

已有 `build/migration/*.json` 参考保持不变。不跑全量或 CUDA，只做受影响组件、实际安装链路及有失败原因的阶段 PDF 回归。

初轮整篇回归在论文第一页发现版面坐标不一致：最初的新入口误将渲染 RGB 交换成 BGR，尽管 Markdown 相同，严格 middle 比较正确阻止了交付。原三个模型样本近灰度，初步输入比较未暴露颜色顺序问题；补充由原模型样本生成的非对称彩色输入，并修正为直接 RGB。失败产物保留在 `build/layout-render-owned-seven/`，没有安装失败 wheel，也没有修改参考或放宽比较。初轮构建 11 分 42 秒，不能当作最终验证通过的构建。

## 最终验证

- `rust/rapiddoc-inference/tests/layout_pixels.rs`：三个原 PNG 与一个非对称彩色样本，原 PNG/新 RGB 入口的 FP32 输出位模式、整型输出及矩形结果一致；auto/poly/quad 结果一致；尺寸、字节数、预算和非法阈值检查通过。最终记录 `build/layout-render-pixels-colored-final-inference-test.txt`，不能只引用初轮近灰度验证。
- `rust/rapiddoc-pdf/tests/render_pixels.rs`：文字/扫描 PDF、72/200 DPI 共四组原 RGB 像素与 PNG 解码逐字节一致，公共 PNG 压缩字节/宽高/scale 相同；非法 DPI 错误类别相同。记录 `build/layout-render-pixels-pdf-test.txt`。
- 原生表格共享拥有入口已通过 526 完整 HTML（含 168 自动上下标正例）、724 完整恢复、32 实际 PDF/角度/样式链，含精确类型、原结果保留、输入所有权、并发、分阶段错误和预算。该表格实现随后未再修改，未因 RGB 修正重复同一组件测试。记录 `build/layout-render-owned-native_table_materialization-test.txt`。
- 最终 wheel：4 页原 API 渲染 cap/像素，15 版面规则/3 实际 PNG 模型通过；记录 `build/layout-render-owned-fixed-{render_caps,layout}-test.txt`。
- 最终七份 PDF 共 60 页、92 图片，与修改前 Rust 的完整输出一致；Markdown/content list/图片模式、尺寸、像素与原 Python 一致。6 份 middle 与原 Python 完全一致；small_ocr 原有两处分数差异继续显式保留，与修改前 Rust 完全一致。记录 `build/layout-render-owned-fixed-seven-test.txt`、`build/layout-render-owned-fixed-seven-summary.json`、`build/layout-render-owned-fixed-seven/`。初轮失败之后只执行这一次修正后阶段回归。
- 119 模块来源审计通过。未跑全量、CUDA 或正式进程峰值内存基准。

## 本轮实际耗时

| 文档 | 用户修改前日志 | 最终本轮 |
| --- | ---: | ---: |
| 论文模板 | 18.327 秒 | 17.867 秒 |
| demo1 | 31.895 秒 | 33.209 秒 |
| 密集长表格 | 3.283 秒 | 3.774 秒 |
| 03 | 52.125 秒 | 49.018 秒 |
| small_ocr | 17.097 秒 | 16.317 秒 |
| 财报 | 10.525 秒 | 9.735 秒 |
| 研究报告 | 10.038 秒 | 9.921 秒 |
| 合计 | 143.291 秒 | 139.841 秒 |

本轮单次总耗时少约 2.4%，但不是反复同条件受控基准。版面阶段合计 37.981→35.516 秒；未修改的公式推理也从 52.561→49.919 秒波动，表格从 17.131→17.811 秒。不能将全部差值归因于本次代码，不能承诺固定提速，也不能宣称密集表格已经加速。修改前用户日志路径及 SHA 保存在 `build/layout-render-owned-user-before.json`，未覆盖旧记录。

## 构建与安装

最终修正版 wheel 构建 5 分 47 秒，记录 `build/layout-render-owned-fixed-build-20261029.txt`。已安装现有 `.venv`，正式 wheel 与测试 wheel 一致，全部 wheel RECORD 哈希/长度通过；144 个安装原生/源码/数据文件逐字节匹配（排除 pip/import 再生成的 pyc），`pip check` 和来源审计通过。旧正确版本保存在 `build/pre-layout-render-owned-wheels-20261029/`。安装日志 `build/layout-render-owned-fixed-{install,pip-check}.txt`。

仍存在整页 PNG 编码（供后续裁图/资产）、后续整页解码、表格遮罩/结构 PNG、每表首次几何请求复制，以及版面 masks 输出复制。原模型推理未替换或降低精度，不能期待这些搬运优化消除模型本身的耗时。
