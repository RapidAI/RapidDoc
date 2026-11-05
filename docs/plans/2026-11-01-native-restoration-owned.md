# 原生 PDF 图内文字与分式恢复的所有权传递

基于上一最终 `native-page-borrow-20261101` 包再次采样：原生分析约占主线程样本 75.8%；树 clone 与 Value serializer 分别约 29.6%/20.0%，重叠且包含等待，不相加也不等同 CPU 百分比。原始数据 `build/native-page-borrow-stack-20261101.tsv`、`build/native-page-borrow-stack-summary-20261101.json`。

源码核查发现几何收尾的行恢复链仍有多份完整字符树副本：图内文字拆行前复制水平行，拆行/补认领的请求与结果重复构造行数组，分式恢复同时复制 remaining/source 两份行。调用方最终只读取 source_index 和 median_glyph_width 更新。

## 实现边界

- 分式恢复内部使用 Cow 暂借原始行，发生真实恢复时才生成新行；正式路径最后按原始对象身份索引移交未改变的行，避免按相等内容或 source_index 合并对象。
- 公开诊断路径继续返回完整 source_lines；正式页面链只传递调用方实际读取的来源号/字宽更新。更新顺序、缺失/空字宽、分式合并与新来源身份不变。
- 图内文字拆行借用水平行/分组，已有行数组直接移交；图内文字补认领在自有数组上执行原处理，只把必要更新摘要传回页面源。保留公开借用接口、完整诊断结果与原处理步骤。
- 图形容器、所属关系、表格排除、字符过滤、旋转、来源顺序和预算规则不变。不根据文件名或“看起来是普通文本”跳过逻辑。
- 缺失/非对象页面继续返回原错误，不因内部 take 操作导致 panic。没有新字符/公式上限，没有更改模型、阈值或数字精度。

## 固定参考

- `tests/fixtures/native_leading_owned_reference.json.gz`：从原 Python 导出器的前 27 组独立场景取得完整预期，含 8 组实际恢复正例；旋转、语义保护、来源顺序、多分母、字符/字框、画线位置等分支。保存 Python formulas.py 源 SHA。
- `tests/fixtures/native_late_image_owned_reference.json.gz`：原 Python 导出器的 29 组独立拆行/补认领场景，含多图、不同 run、表格排除、字体、图形归属竞争、背景图、数字和旋转等；保存 graphics.py、pipeline.py 源 SHA。
- 两组均在有限预算/生产作用域检查输出、更新/身份、错误与 Work 计数；并通过已有文字角色/脚注/行簇/脚本、原生诊断/辅助文字/来源行参考。`cargo test` 的 `owned_` 过滤共 6 项通过。原 Python 后续依赖的连续 PDF 场景没有伪称已加入这 56 组固定参考。

第一份构建 `native-restore-owned-20261101` 只用于开发，不交付；最终构建使用 `native-restore-owned-r2-20261101`，包括非对象输入保护及全部改动。正式安装、PDF 回归和计时使用最终包。只跑受影响测试，阶段结束再构建/回归；不跑无关 Office、全 workspace、全 PDF、CUDA 或新内存基准。

## 最终安装验收

- 16、65、129 页以及密集长表格完整快照与上版严格一致；默认模型扫描/论文/印章 6 组公开选页对照通过。36 组资产/错误和 5 项资源策略检查通过。
- 原始记录：`build/native-restore-{window-verify,public,dense-verify,assets,resource-tests}-20261101` 对应目录/文件。
- 构建/回归结束后，同一 129 页原生规则输入、64 页窗口、txt、关闭公式/表格模型，使用独立进程按上版/新版/Python顺序各两次调用；轻量阶段日志一致，没有 cProfile 或堆栈采样。计时期间不编译或回归。

| 调用 | 上版 Rust 秒 | 新版 Rust 秒 | 耗时下降 | 同期 Python 秒 |
| --- | ---: | ---: | ---: | ---: |
| 首次 | 12.959 | 11.865 | 8.4% | 11.798 |
| 复用 | 12.939 | 11.981 | 7.4% | 10.325 |

热原生分析 9.772→8.831 秒，渲染约 1.57 秒、组装约 0.70 秒。各版本 Markdown 哈希一致；完整结构和资产回归在独立进程校验。首次接近 Python，热调用仍慢约 16%，未完成追平；这是少量顺序调用，不是多轮交替基准，也不代表全部 PDF 或扫描速度。

最终 wheel 已安装并保留项目 `.venv`，现有 demo 直接运行。`build/native-restore-delivery-receipt-20261101.json` 保存源码、Python 固定参考与源 SHA、wheel/安装指纹、全部回归和计时；安装二进制与 wheel、公开回归 manifest 一致。
