# 性能回退：恢复 09:31 已测试安装包

## 当前交付状态

2026-10-29 10:56 已将现有 `.venv` 和 `build/wheels` 回退至 `build/pre-layout-render-owned-wheels-20261029/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`。该包对应用户 09:31 的 143.291 秒七篇记录，包含此前识别像素/FMA/CTC 输出借用，不含后续渲染 RGB 复用、恢复请求拥有入口及候选分配实验。

wheel SHA256：`1f62e79e8b396adcc9b9cfaa74a8e8b57945300602d7fca5be6140fa6925f20e`。安装原生二进制、`__init__.py`、`_compat.py`、`_runtime.py`、审计元数据与回退 wheel 逐字节一致，`pip check` 通过。日志 `build/performance-regression-rollback-install.txt`。

刚安装过的候选分配包保留在 `build/quarantined-native-table-allocation-20261029/`，`build/dev-wheels`/开发 SDK 仍是实验包。源代码中的后续实验没有删除；**源码与当前稳定安装包不同**。后续先定位端到端回退，不要将普通重新构建安装误当作已验证的修复。组件收益不能作为整篇性能验收通过的替代。

## 用户最新反例

日志 `C:/Users/huazhen/.codex/attachments/5cbbb5a1-e19b-461b-9b8c-c9dcb6873316/已粘贴的文本.txt`：七篇 248.523 秒，对比用户 09:31 的 143.291 秒，增加约 73.4%。全部正常结束，但多个阶段变慢，包括未被候选分配改动涉及的版面、公式推理和渲染。

上一轮在财报整篇交替中位数 15.994→16.841 秒时仍安装新包，交付判断不稳妥；不能再只报告恢复/渲染组件的有利中位数来解释用户反例。现有反例和之前全部基准文件保留，不改写 `build/migration/*.json`。

## 回退后核对，根因尚未确认

无 Cargo/编译进程时，独立进程、相同 small_ocr 输入/参数/首次加载范围进行一次旧/实验包核对。两边完整输出和图片均符合原参考：

| 包 | 当前整篇 | 版面 | OCR 识别 |
| --- | ---: | ---: | ---: |
| 09:31 回退包 | 23.148 秒 | 8.467 秒 | 6.769 秒 |
| 暂停交付的实验包 | 21.809 秒 | 7.635 秒 | 5.818 秒 |

记录 `build/performance-regression-{rollback,quarantined}-control-{old,new}.json`（实际分别为 rollback/old、quarantined/new），以及相应 `*-small-ocr.txt`。两者都未恢复此前用户 small_ocr 约 17.097 秒，不能声称回退已经解决性能问题，也不能把全部 73% 增长归因于最后三个 Rust 文件。

确认两包 Python `_runtime.py`、`_compat.py`、`__init__.py` 相同；最新日志的模型路径和自动/表格 4 线程配置相同。用户 demo 仅导入 Rust 包，同一实例复用模型，未发现同进程导入原 Python PDFium 的情况。

只读现场快照显示：32 GB 物理内存可用约 4 GB，提交内存使用约 93%，存在两个后台进程各消耗约一个逻辑核；电源方案为高性能。这仅是后续定位线索，不是已确认的历史回退根因。没有停止其他应用、修改电源方案或进程调度设置。

当前保留回退包，暂停新的性能改动安装。后续需要解释共同运行条件及整篇回退，再决定哪些源代码实验可重新交付；不能仅凭一次较快运行认定已解决。
