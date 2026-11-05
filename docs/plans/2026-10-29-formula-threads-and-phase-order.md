# 公式默认线程及阶段顺序修正

用户七文件实测：Rust 251.827 秒，原 Python 168.335 秒，耗时约 1.50 倍。前一轮 PNG/拷贝组件优化没有解决主要差距，不能把组件百分比当作整篇收益。

## 已定位问题

原 Python 的实际公式会话默认值为 intra_op=0、inter_op=0、CPU arena=False、ORT_ENABLE_ALL。Rust Python 适配层误将公式默认 intra_op 设置为 4（版面/OCR 已是自动线程）。之前“公式默认 4 线程已对齐”的记录有误，本次明确更正。用户显式提供线程数仍应优先。

从同一模型、同一批三组 FP32 输入张量进行独立进程诊断，预热后重复三轮，输出 token 和输入 SHA 完全相同：

| 入口 | 线程 | 三组输入中位耗时 |
| --- | --- | ---: |
| Python ORT | 4 | 1.443407 s |
| Rust ORT 开发入口 | 4 | 1.469158 s |
| Python ORT | 自动 | 1.091116 s |
| Rust ORT 开发入口 | 自动 | 1.086869 s |

Rust 诊断入口含预先序列化 JSON 的解析及输出转换，并非只有 ORT 内核计时；正式管线不经该开发 JSON 路径。三组样本是定位配置差异的证据，不代表全部文档或平台。原 Python 的 CPU arena 另有显式设置差异，本轮没有凭猜测更改其分配策略。

原 Python 在窗口内先完成各页版面，再执行公式、OCR 检测；Rust 之前逐页交错调用这些会话。本次改为窗口内按阶段执行，单个模型的输入顺序、batch/padding、模型与阈值保持不变。窗口大小不变，公式原始像素仍逐页产生并在使用后释放，不缓存整个窗口的公式 RGB。

## 验证与证据

原 Python 实际会话设置：`build/formula-original-options-local-20261029.txt`。相同输入诊断：`scripts/migration/probe_formula_execution.py`、`build/formula-execution-20261029/`。

用户现成的七篇 Python 输出另存为新的不可变参考 `build/migration/user-seven-pdfs-20261029.json`，包含 60 页、92 个图片资产的像素证据；没有用新 Rust 输出覆盖参考。修改前用户 Rust 输出也独立冻结到 `user-seven-rust-before-phase-20261029.json`。

生产 release 构建完成（5 分 18 秒），结束编译后单独运行七篇。没有并行测试或编译负载：

| 文档 | 用户 Python 日志 | 用户修正前 Rust 日志 | 本轮 Rust |
| --- | ---: | ---: | ---: |
| 论文模板 | 20.141 s | 25.834 s | 19.234 s |
| demo1 | 40.044 s | 64.015 s | 35.789 s |
| 密集表格 | 2.774 s | 6.929 s | 4.416 s |
| 03 | 63.642 s | 98.418 s | 55.064 s |
| small_ocr | 21.153 s | 27.872 s | 19.524 s |
| 比亚迪财报 | 9.954 s | 12.732 s | 10.382 s |
| 研究报告 | 10.626 s | 16.027 s | 11.891 s |
| 合计 | 168.335 s | 251.827 s | 156.300 s |

三个运行不是同一时刻反复交替的正式基准，本轮无写盘，用户脚本含写盘，因此不能承诺固定提速比例或普遍优于 Python。密集表格、研究报告和财报仍慢于这份 Python 日志，表格链是剩余重点。demo1 公式推理从用户日志的 40.108 秒到本轮 18.318 秒，03 从 64.143 秒到 33.449 秒；这是同时修正默认线程与阶段顺序的联合结果，不能独立分摊成每一项的收益。

七篇 Markdown、content list 与原 Python 完全一致，92 个图片资产的模式、尺寸、像素一致。六篇 middle JSON 与原 Python 完全一致；small_ocr 存在两处既有 score 差异（Python 0.998、Rust 0.999），其余相同。本轮七篇全部三类输出与修改前用户 Rust 完全相同，没有通过调整文字、置信度、阈值或参考来取得速度。

首次七篇严格 Python 对照在末尾检查报错，原因就是上述两处旧差异；失败日志保留为 `build/phase-order-seven-test.txt`，不声明该次严格检查通过。随后逐项对照修改前真实 Rust 输出，完整非回退及原 Python 文本/图片验证保存在 `build/phase-order-seven-verified.json`。通用安装脚本可显式传入修改前 Rust 参考进行非回退检查，仍保留所有 Python 一致性字段和既有差异报告，不针对某个文档放宽生产行为。

5 个固定单线程 API 原参考案例、分类统计及真实阶段顺序专项通过，另核对默认公式自动线程和显式单线程覆盖；原生变体和来源审计的最终安装结果见下方交付记录。没有重跑不受影响的算法测试或完整/CUDA 套件。

本轮输出和计时：`build/phase-order-seven/`、`build/phase-order-seven-summary.json`；构建日志 `build/phase-order-build-20261029.txt`。

## 最终交付

原生 API 变体及 119 模块来源审计通过。已安装现有 `.venv`，原生二进制、适配器、兼容审计与已测试 wheel 逐字节相同，wheel RECORD 全部哈希/长度检查及 `pip check` 正常；默认模型线程实际检查为版面/检测/识别/公式自动、表格 4。用户原测试脚本不需要改参数即可使用修正默认值，已加载扩展的 IDE 进程需重新运行。旧 wheel 保留于 `build/pre-phase-wheels-20261029/`。安装/专项日志：`build/phase-order-install.txt`、`build/phase-order-pip-check.txt`、`build/phase-order-api-test.txt`、`build/phase-order-native-test.txt`、`build/phase-order-audit-test.txt`。同一验证二进制不再重复完整回归。
