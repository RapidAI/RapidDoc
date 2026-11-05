# 原生 PDF 行处理复制优化

承接七格式性能验收发现的原生文本规则慢点。本轮不改变表格候选、文字/脚注分类、几何阈值、跨页窗口或识别模型。

## 修改

- 文字角色、脚注和末尾辅助文本处理增加内部接收 `Vec<Value>` 的入口；页面调用者转移已校验的行数组，不再先通过 `json!` 复制整组含字符数据的行、再由分类器复制一次。原公开/诊断请求入口保留参数校验及原有行为。
- 行簇合并增加借用输入切片的内部入口；完整页准备直接传递行、页面尺寸和表格区域，结果中的行数组通过 `take` 转移，不再往返序列化和复制。
- 上下标脚本准备直接借用行、字符、绘图、紧框和 origin；不再为内部请求复制全页字符与几何映射，也不再在准备器内复制同一字符列表。
- 行簇的行字节计数和表格字符准备的 clone-byte 计数，在无大小预算的兼容作用域内跳过序列化。有限预算时仍使用同样的计数及限制；不移除形状、数值、坐标和异常输入校验。
- 现有 `RAPIDDOC_NATIVE_PROFILE` 下增加几何重拆、图像认领、文字角色、脚注、末页准备等细分耗时。默认不输出新增日志。开发例程 `rust/rapiddoc/examples/native_work.rs` 直接执行原生规则并保存完整结果，不加载模型。

## 基线与验证方法

保存修改前的独立 release 诊断程序 `build/native-work-baseline-20261101.exe`。先捕获 16 页完整原生输出，再交替执行修改前/后程序，避免把机器状态变化当成代码收益。该程序直接使用低层原生入口，不等同于默认模型 PDF 流程，也不是含渲染/组装的公开 API 全耗时。

初步交替 A/B（仅所有权改动、尚不含后续逐字 clone-byte 计数优化）：

| 输入 | 修改前 s | 修改后 s | 方法 |
|---|---:|---:|---|
| 16 页生成原生文本 | 2.4931 | 2.3435 | 三轮交替中位数，约降低 6.0% |
| 129 页生成原生文本 | 20.1673 | 18.9273 | 一轮，约降低 6.1%，仅诊断 |
| 论文前两页 | 1.1652 | 1.1952 | 一轮，未证明提速 |
| 印章前两页 | 0.0889 | 0.0871 | 一轮，差异很小 |

全部 A/B 完整原生 JSON 一致，含文字、几何、块、顺序及数据类型；原始记录 `build/native-owned-ab-20261101.json`。最初非交替单次的 3.50→2.85 s 不用于正式收益结论，机器状态会影响绝对时间。不同阶段边界也不能与公开 API 39 秒直接相减。

密集表格文件的低层独立例程在修改前即因 source_lines 诊断上限拒绝，未把它计为成功对照。此项不代表正式默认模型 API 报错；最终公开路径另行验收。也未在这次性能修改中顺手放宽该低层诊断行为。

## 参考测试

- 持久参考 `tests/fixtures/native_owned_line_reference.json.gz`：92 组文字角色、143 组脚注、32 组行簇、4 个实际生成 PDF 旋转角度的上下标脚本，共 271 个阶段用例，包含原 Python 的预期错误。保存原 Python 关键源码 SHA256，脚本 `export_owned_line_reference.py` 生成且拒绝覆盖已存在的固定参考。
- 新原生测试逐项比较 owned/borrowed 路径与 Python 完整预期，不仅比较两种 Rust 调用相互一致。有限字节计数、兼容作用域跳过计数及作用域恢复另有断言。
- 初始历史测试缺少本地 build/migration 参考文件导致失败，不是解析差分。相关 Python 参考重建后，92 组原文字角色和 164 组早期脚注参考通过；新增 271 用例固定压缩保存，避免依赖被清理的 build 文件。
- 本轮开发仅运行受影响测试；最终一次构建安装后运行指定 PDF 公开路径及 16/65/129 页完整输出回归，不重跑无关 Office、全 workspace 或 CUDA。

## 复现

```powershell
cargo test --locked -p rapiddoc --lib owned_line_stages_preserve_python_reference_outputs
cargo test --locked -p rapiddoc --lib clone_byte_counter_keeps_finite_limits_and_skips_unlimited_scope
cargo test --locked -p rapiddoc --lib output_byte_count_preserves_exact_finite_limits
```

只有重新生成缺失参考时才运行 `.venv/Scripts/python.exe scripts/migration/export_owned_line_reference.py`；已有参考不覆盖。原生例程接受 PDFium DLL、源 PDF、选定页数和输出 JSON 四个参数，计时不含输出文件序列化/写盘，未宣称覆盖全部 PDF 语义或精度。

最终安装版本、公开耗时和输出差分以本报告后续安装验收及收据为准。原生规则仍有大量阶段开销，本轮不能解释为已解决此前相对 Python 的全部速度差距。

## 最终公开接口计时

最终版本含上述所有权/借用改动及逐字 clone-byte 计数优化。沿用同一 129 页构造文件、docvortex_rule/txt、关闭公式/表格模型、64 页窗口，版本间顺序执行，各一次冷/热调用：

| 项目 | 修改前 s | 修改后 s |
|---|---:|---:|
| 首次完整调用 | 38.408 | 38.901 |
| 第二次完整调用 | 39.489 | 38.914 |
| 热调用原生分析阶段 | 28.356 | 27.343 |
| 热调用渲染阶段 | 7.326 | 7.837 |
| 热调用组装阶段 | 2.831 | 2.777 |

Markdown 哈希保持一致。单次热调用总耗时约减少 1.5%，原生分析阶段约减少 3.6%，但首次调用略慢，渲染波动也抵消部分收益。这不是交替多轮公开接口基准，不能宣称整体已稳定提速，仍明显慢于此前 Python 对应规则场景。代码移除了可确认的复制/无效计数，但未完成此场景的性能追平目标；本轮没有重新做峰值内存比较，不据删除复制推算内存下降百分比。

原始计时：build/native-owned-public-{before,after}-20261101.json。后续应继续定位表格候选、页面编排/几何和渲染中的剩余重复工作；不以取消规则或降低精度来追平。

## 最终安装验收

`build/native-owned-lines-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` 已安装项目 `.venv`，已核对 wheel 原生二进制与实际安装文件 SHA256 一致，现有 demo 可直接运行。

- 3 项最终受影响原生检查通过：271 组固定 Python 参考，clone-byte 计数有限/无限作用域，有限输出预算检查。此前重建的 2 项原文字角色/脚注测试也通过。
- `tests/migration/test_production_resource_policy.py` 的 5 项安装测试通过。
- 16/65/129 页公开路径完整快照与 R2 基线严格一致；比较保留整数/浮点区别、全部未知字段、文本、结构、资产身份及顺序，没有放宽精度。
- 默认模型下扫描、论文、印章共 6 组公开选页/单页窗口检查与 Python 对照通过。结果 `build/native-owned-pdf-public-20261101/summary.json`。

最终收据：`build/native-owned-delivery-receipt-20261101.json`，含当前二进制、wheel、修改源码和固定参考哈希。可运行 `.venv/Scripts/python.exe scripts/migration/report_native_owned_delivery.py` 从已保存结果复核，不重新解析。日志保留于 build/native-owned-{final-tests,resource-tests,window-verify,pdf-public}-20261101 对应文件/目录。

本轮结果是去除复制并完成精度回归，公开端到端速度收益仍有限，未完成原生规则性能追平，也未宣称扫描件提速或内存下降。
