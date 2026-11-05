# 算法块 Magika 接入与评测日志（2026-11-02）

## 实际问题

OmniDocBench 的 PPT_PresentationSpecification_page_010.pdf 在页处理成功后，
Markdown 输出阶段触发 algorithm 分支的旧 unsupported。此前小语料验收未
覆盖这个分支；早期输出参考明确排除过算法链，不能据已测 PDF 推导所有
模型标签都已闭合。不是这份 PDF 损坏，也不是表格推理出错。

官方 Magika Rust 1.1.0 确实存在，但固定依赖 ort = 2.0.0-rc.12；本项目
统一 ORT ABI 为 rc.15/api-23，直接添加会引入 links/版本冲突。因此这里只
迁移原 Python guess_language_by_text 的实际代码语言路径，复用当前 Rust
ORT，而不是引入另一套推理运行时或调用 Python Magika。

来源：[官方 Rust 绑定说明](https://securityresearch.google/magika/cli-and-bindings/overview/)。
参考 crate 的 Cargo.toml 已保存 build/magika-reference/Cargo.toml。

## 改动

- 原 Python standard_v3_3 ONNX 和 config 不改，作为 Rust 原生资产嵌入；
  Google Apache-2.0 原许可、模型 README、来源及 SHA 一并保留。
- 只在算法块首次需要时创建 CPU Magika 会话，并复用。特征为原始 bytes
  前/后各 1024 字节，按各 4096 字节窗口做 ASCII bytes lstrip/rstrip，
  padding=256，输入 int32；保留原高置信阈值、overwrite_map、首次最大值
  tie 规则和语言封装的 unknown/低置信度/分类异常 -> txt 行为。
- 算法正文保留逐行拼接、缩进、空行及原全半角规则，生成带语言的 fenced
  code block；不是无条件标 txt 或删掉算法内容来绕过异常。
- 非算法渲染和页面解析算法不改。语言无关 JSON 操作与 Python 公共输出
  均可使用原生分类，不导入原 Python 解析器或 Python magika 包。

## 验证

本地冻结原 Python 46 组语言和 2048 个原始特征向量，Rust 逐项一致；
原字节空白/padding 单元检查通过。服务器原 ORT 1.23.0 冻结 45 组语言、
405 组完整 Markdown/content_list 输出，Rust 全部一致；405 组包含算法
标签在块/单独 span，以及普通段落对照。没有把已清理掉的历史 171 组
夹具算作本轮验证；也没有依赖全量测试来凑覆盖。

服务器实际报错 PDF 重新完整解析成功：1 页、668 个 Markdown 字符、
2 个图片资产和 1 个代码块。将同一次解析的 pages 交给原 Python union_make，
Markdown 和 content_list 均逐项一致。只复测这一份报错文件，没有自动
启动 1651 份整套评测。结果位于服务器 results/magika-failing-page/。

## 评测日志

只修改 demo/RapidDoc_rust_end2end.py，不改变评测预测目录/文件名。
每次启动会在预测目录旁的 `<预测目录名>_logs/` 写入：

- `<时间>.log`：完整 DEBUG 及以上阶段日志。
- `<时间>.errors.log`：ERROR 及完整异常堆栈。
- `<时间>.failures.jsonl`：发生失败时追加文件名、异常类型和消息。

终端保留 INFO 进度，关闭变量值展开以免异常时打印整页结构。
单份失败继续下一份，结束打印成功/跳过/失败/总秒数；有失败返回非零。
Markdown 在图片成功后原子提交，重跑跳过已有 `.md`。模拟成功、跳过、
失败的定向测试验证日志、失败清单、最终汇总和不生成失败预测文件。

## 部署

Linux 修复包已安装用户原命令所用的
`/web/hzkitty/rapiddoc-cuda-validation/20261101/release-acceptance/linux-cuda/env-delivery/`。
服务器同名评测脚本也已更新。重启原评测命令才能加载新代码；成功的
预测保持跳过，失败文档会补跑。原 Python 业务环境不改。

本地 Windows release wheel 已构建并安装项目 `.venv`；46 组原始特征/
语言检查、414 组完整输出对照、原许可声明检查与 23 文档 DOCX 安装/API
及 PDF 文本冒烟通过。日志成功/跳过/失败测试、字节特征单元测试及编译
检查通过。未跑全量或整套评分；全 workspace fmt 检查发现原 examples
已有格式差异，未为此修改无关示例。

当前 Windows wheel：`build/magika-windows-wheels/`；当前 Linux CUDA
wheel：`build/magika-fix/linux-wheels/`。此前不含此修复的构建仍为历史
产物，勿用旧包重新覆盖。服务器原手动命令和输出目录保持不变。

原始模型/配置来源、源文件 SHA、Linux wheel、405 组参考、报错页 pages /
Markdown / 原 Python 对照和运行日志已归档并下载校验：
`build/magika-fix/magika-fix-results.tar.gz`，40,929,975 字节；SHA256
`0d913a61773e19d3544124f01169a3e8452aad0ed60d9310aa172a2a6694ecf8`。

